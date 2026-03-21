import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import json
import time
import argparse
import pandas as pd
import numpy as np
import torch
import cv2
import open_clip
from pathlib import Path
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from concurrent.futures import ProcessPoolExecutor
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR        = Path("/home/build/martin/spotify_v2")
COVERS_DIR      = BASE_DIR / "data/covers"
METADATA_PATH   = BASE_DIR / "spotify_411k.parquet"
CHECKPOINT_DIR  = BASE_DIR / "checkpoints_album"

CHECKPOINT_DIR.mkdir(exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────────
CPU_WORKERS         = 12
COLOUR_BATCH_SIZE   = 5000
CLIP_BATCH_SIZE     = 256
COLOUR_RESIZE       = 64
TEXTURE_RESIZE      = 128
N_PALETTE_COLOURS   = 5
CLIP_MODEL          = "ViT-B-32"
CLIP_PRETRAINED     = "openai"

# ── Stage 1: Colour & Texture ──────────────────────────────────────────────────

def _extract_one_ct(args):
    img_path, n_palette, colour_sz, texture_sz = args
    track_id = Path(img_path).stem
    result = {"track_id": track_id, "error": None}
    try:
        # Single read with CV2
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None: raise ValueError("Could not read image")
        
        # Colour features (Resize early)
        img_rgb_small = cv2.resize(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), (colour_sz, colour_sz))
        arr = img_rgb_small.reshape(-1, 3).astype(np.float32)
        
        km = MiniBatchKMeans(n_clusters=n_palette, n_init=3, random_state=0, max_iter=50).fit(arr)
        palette_flat = (km.cluster_centers_.flatten() / 255.0).tolist()
        counts = np.bincount(km.labels_, minlength=n_palette)
        palette_weights = (counts / counts.sum()).tolist()

        # HSV stats (Small version)
        img_hsv_small = cv2.cvtColor(img_rgb_small, cv2.COLOR_RGB2HSV).astype(np.float32) / 255.0
        h_mean, s_mean, v_mean = img_hsv_small.reshape(-1, 3).mean(axis=0)
        h_std, s_std, v_std = img_hsv_small.reshape(-1, 3).std(axis=0)
        
        # Warm/cool proxy (R - B channel)
        colour_temp = float((arr[:, 0].mean() - arr[:, 2].mean()) / 255.0)
        
        # Hue entropy
        hue_hist = np.histogram(img_hsv_small[:, :, 0], bins=16, range=(0, 1))[0].astype(float)
        hue_hist /= hue_hist.sum() + 1e-9
        colour_entropy = float(-np.sum(hue_hist * np.log2(hue_hist + 1e-9)))

        # Texture features (Greyscale small)
        gray_small = cv2.resize(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), (texture_sz, texture_sz))
        
        sharpness = float(cv2.Laplacian(gray_small, cv2.CV_64F).var())
        contrast = float(gray_small.std())
        
        lum_hist = cv2.calcHist([gray_small], [0], None, [64], [0, 256]).flatten()
        lum_hist /= lum_hist.sum() + 1e-9
        lum_entropy = float(-np.sum(lum_hist * np.log2(lum_hist + 1e-9)))
        
        sx = cv2.Sobel(gray_small, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(gray_small, cv2.CV_64F, 0, 1, ksize=3)
        edge_orientation = float(np.abs(sx).mean() / (np.abs(sy).mean() + 1e-9))

        result.update({
            **{f"palette_r{i}": palette_flat[i*3] for i in range(n_palette)},
            **{f"palette_g{i}": palette_flat[i*3+1] for i in range(n_palette)},
            **{f"palette_b{i}": palette_flat[i*3+2] for i in range(n_palette)},
            **{f"palette_w{i}": palette_weights[i] for i in range(n_palette)},
            "h_mean": h_mean, "s_mean": s_mean, "v_mean": v_mean,
            "h_std": h_std, "s_std": s_std, "v_std": v_std,
            "colour_temp": colour_temp, "colour_entropy": colour_entropy,
            "sharpness": sharpness, "contrast": contrast,
            "lum_entropy": lum_entropy, "edge_orientation": edge_orientation
        })
    except Exception as e:
        result["error"] = str(e)
    return result

def run_ct_extraction(cover_paths, gpu_id):
    ct_batch_dir = CHECKPOINT_DIR / f"gpu{gpu_id}_ct"
    ct_batch_dir.mkdir(exist_ok=True)
    
    done_ids = set()
    for p in ct_batch_dir.glob("*.parquet"):
        try: done_ids.update(pd.read_parquet(p, columns=["track_id"])["track_id"].tolist())
        except Exception: pass

    todo = [(p, N_PALETTE_COLOURS, COLOUR_RESIZE, TEXTURE_RESIZE) for p in cover_paths if p.stem not in done_ids]
    print(f"[GPU {gpu_id}] Stage 1: {len(done_ids):,} done, {len(todo):,} remaining")
    if not todo: return

    with ProcessPoolExecutor(max_workers=CPU_WORKERS) as pool:
        batch = []
        for i, result in enumerate(tqdm(pool.map(_extract_one_ct, todo, chunksize=50), total=len(todo), desc=f"GPU {gpu_id} CT")):
            if result["error"] is None:
                del result["error"]
                batch.append(result)
            
            if len(batch) >= COLOUR_BATCH_SIZE or i == len(todo) - 1:
                if batch:
                    df = pd.DataFrame(batch)
                    ts = int(time.time() * 1000)
                    df.to_parquet(ct_batch_dir / f"batch_{ts}.parquet", index=False)
                    batch = []

# ── Stage 2: CLIP Embeddings ───────────────────────────────────────────────────

class CoverDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        path = self.paths[i]
        track_id = Path(path).stem
        try:
            img = Image.open(path).convert("RGB")
            return self.transform(img), track_id
        except Exception:
            return None, track_id

def collate_fn(batch):
    batch = [(img, tid) for img, tid in batch if img is not None]
    if not batch: return None
    imgs, ids = zip(*batch)
    return torch.stack(imgs), list(ids)

def run_clip_extraction(cover_paths, gpu_id):
    clip_batch_dir = CHECKPOINT_DIR / f"gpu{gpu_id}_clip"
    clip_batch_dir.mkdir(exist_ok=True)
    
    done_ids = set()
    for p in clip_batch_dir.glob("*.json"):
        try:
            with open(p) as f: done_ids.update(json.load(f))
        except Exception: pass

    todo = [p for p in cover_paths if p.stem not in done_ids]
    print(f"[GPU {gpu_id}] Stage 2: {len(done_ids):,} done, {len(todo):,} remaining")
    if not todo: return

    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(CLIP_MODEL, pretrained=CLIP_PRETRAINED)
    model = model.to(device).eval()

    dataset = CoverDataset(todo, preprocess)
    loader = DataLoader(dataset, batch_size=CLIP_BATCH_SIZE, num_workers=4, pin_memory=("cuda" in device), collate_fn=collate_fn)

    new_embs, new_ids = [] , []
    save_every = 20 # batches
    
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=("cuda" in device)):
        for i, batch in enumerate(tqdm(loader, desc=f"GPU {gpu_id} CLIP")):
            if batch is None: continue
            imgs, ids = batch
            imgs = imgs.to(device, non_blocking=True)
            feats = model.encode_image(imgs)
            feats /= feats.norm(dim=-1, keepdim=True)
            new_embs.append(feats.cpu().float().numpy())
            new_ids.extend(ids)

            if (i + 1) % save_every == 0 or i == len(loader) - 1:
                if new_embs:
                    ts = int(time.time() * 1000)
                    emb_arr = np.vstack(new_embs)
                    np.save(clip_batch_dir / f"batch_{ts}.npy", emb_arr)
                    with open(clip_batch_dir / f"batch_{ts}.json", "w") as f:
                        json.dump(new_ids, f)
                    new_embs, new_ids = [], []

# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--total-gpus", type=int, default=1)
    args = parser.parse_args()

    print(f"[GPU {args.gpu}] Scanning directory...")
    all_covers = {}
    with os.scandir(COVERS_DIR) as it:
        for entry in it:
            if entry.name.endswith(".jpg") and entry.is_file():
                all_covers[entry.name[:-4]] = Path(entry.path)
    
    print(f"[GPU {args.gpu}] Reading metadata...")
    meta = pd.read_parquet(METADATA_PATH, columns=["track_id"])
    valid_ids = [tid for tid in meta["track_id"].astype(str) if tid in all_covers]
    
    # Sharding
    sharded_ids = valid_ids[args.gpu :: args.total_gpus]
    if args.limit:
        sharded_ids = sharded_ids[:args.limit]

    cover_paths = [all_covers[tid] for tid in sharded_ids]
    print(f"[GPU {args.gpu}] Tracks to process: {len(cover_paths):,}")
    
    run_ct_extraction(cover_paths, args.gpu)
    run_clip_extraction(cover_paths, args.gpu)
    
    print(f"[GPU {args.gpu}] Complete.")
