# Spotify Top 5000 Dataset
These are the 5000 most popular songs around March 2023 on Spotify.

-> eda.ipynb : Find a rudimentary EDA, which may need some fixing.
-> read_dataset.ipynb : Simple notebook to read the dataset.
-> recommendations.ipynb : Several rudimentary recommendation systems, which need improvement.
-> spectrogram.ipynb : Generate spectrograms for the songs.
-> spotify_5000.ipynb : (Not for use) to generate a dataset from the 5000 top songs (of the full dataset).


### Missing Data
Beware of the following rows, which contain 'False' values:

| row_index | track_id | cover_downloaded | preview_downloaded |
|---|---|---|---|
| 4101 | 5W7DOVGQLTigu09afW7QMT | False | False |
| 4635 | 4AGPU9325LRTIBsBSJ5v75 | True | False |
| 1865 | 5sqNqbyVJ8RF2E1qeX4sw9 | True | False |
| 2830 | 4w273WCBXwM4P3jTX5HkB2 | True | False |
| 902 | 7cZoAjj5vufdREBJZGW2lH | True | False |
| 433 | 5EaVikAxSRponyifRP6Fhp | True | False |
| 846 | 3IgjZRkRFAWflrCxRimOoL | True | False |
| 3521 | 7MRU4vOkywuhZ9kbFiPuiu | True | False |
| 2300 | 3H3x4NhR3wJh6IvRHmPvkd | True | False |
| 201 | 3B60AOG78SDAmK5kLozoqo | True | False |
| 3828 | 6rZVy6FIG7lSJQMFXHo12z | True | False |
| 3538 | 45Ugm9xuEUtnItECxHghGx | True | False |

___

# Spotify Dataset Augmentation with Lyrics and Metadata
This notebook demonstrates how to augment a Spotify dataset with additional metadata and lyrics using the `spotifyscraper`, `syrics`, and `lyricsgenius` libraries. It also includes steps for data deduplication and caching to optimise the extraction process.

### `extractors/spotify_extractor_v9.ipynb`
The ninth iteration of the Spotify extractor notebook, with a larger sample size for processing. It now incorporates a sample size of 1,159,764 tracks, up from the previous 114,000. The notebook also includes environment variable settings for Spotify and Genius API keys, and demonstrates how to load the dataset from a Parquet file instead of CSV for improved performance. The LRCLib library is used for handling lyrics data, with fallbacks from Spotify, Genius, and Musixmatch. The notebook is structured to facilitate efficient data extraction and augmentation.

### `spotify_data.parquet`
A Parquet file containing the Spotify dataset, which is more efficient for loading and processing large datasets compared to CSV. It originates from [Kaggle](https://www.kaggle.com/datasets/amitanshjoshi/spotify-1million-tracks?source=post_page-----5780cabfe194---------------------------------------).

##### What does `'popularity'` mean?
| Popularity Score | Daily Streams (Approx.) | Algorithmic Milestones |
|------------------|------------------------|------------------------|
| 0 – 10           | < 50                   | Brand new or inactive catalog. |
| 11 – 20          | 100 – 300              | 20% is the threshold for Release Radar. |
| 21 – 30          | 400 – 1,000            | 30% often triggers Discover Weekly. |
| 31 – 40          | 1,500 – 5,000          | Solid "mid-tier" indie momentum. |
| 41 – 50          | 6,000 – 15,000         | Likely appearing on larger editorial playlists. |
| 51 – 60          | 20,000 – 50,000        | Significant viral or genre-wide traction. |
| 61 – 70          | 60,000 – 150,000       | "Hit" territory within a specific region. |
| 71 – 80          | 200,000 – 600,000      | Major label priority / Global "Breaking" hits. |
| 81 – 90          | 700,000 – 2M+          | Top 200 Global Chart territory. |
| 91 – 100         | 3M – 10M+              | The biggest songs in the world (e.g., Taylor Swift). |


### `extractors/parquet_batch_inspector.ipynb`
A utility notebook designed to inspect and validate the contents of Parquet files in batches. This notebook allows users to load Parquet files in manageable chunks, display sample records, and verify the integrity of the data. It is particularly useful for large datasets, enabling users to quickly identify any issues or inconsistencies in the data before proceeding with further analysis or processing. Additionally, it allows you to see how many lyrics were successfully extracted and how many tracks have metadata, providing insights into the coverage of the dataset.

## Motivation
This project was created to augment a large Spotify dataset with additional metadata and lyrics, which can be valuable for various applications such as music analysis, recommendation systems, and natural language processing tasks. By leveraging APIs from Spotify and Genius, we can enrich the dataset with information that is not readily available in the original CSV file. The use of Parquet format allows for efficient storage and faster processing of the large dataset, making it feasible to work with over a million tracks.

Example use cases include:
1. Popularity Prediction
2. Building a Music Recommendation System
3. Sentiment Analysis of Lyrics