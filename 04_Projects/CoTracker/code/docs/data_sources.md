# Data Sources

## Best sources for this project

### 1. Official CoTracker repository assets

Use first for smoke tests and quick demos.

- Local files already in this repo:
  - `assets/apple.mp4`
  - `assets/apple_mask.png`
  - `assets/bmx-bumps.gif`

Why use them:

- zero setup
- already known to work with the repo
- ideal for debugging scripts and visualization

### 2. TAP-Vid DAVIS

Use for the first real benchmark comparison.

Official sources:

- DeepMind TAPNet repo: https://github.com/google-deepmind/tapnet
- TAP-Vid paper: https://proceedings.neurips.cc/paper_files/paper/2022/file/58168e8a92994655d6da3939e7cc0918-Paper-Datasets_and_Benchmarks.pdf
- DAVIS dataset site: https://davischallenge.org/

Why use it:

- standard point-tracking benchmark
- already integrated into this CoTracker repo
- gives report-ready quantitative results

Status in this repo:

- already downloaded and extracted to `datasets/tapvid_davis/tapvid_davis.pkl`

### 3. CoTracker3_Kubric

Use if we want controlled, synthetic, point-tracking stress tests that come from the CoTracker3 ecosystem itself.

Official sources:

- dataset card: https://huggingface.co/datasets/facebook/CoTracker3_Kubric
- model card: https://huggingface.co/facebook/cotracker3

What makes it useful:

- official CoTracker3 training data release
- around 6,000 sequences
- 512 x 512 resolution
- 120 frames per sequence

Best use in our project:

- controlled qualitative inspection
- synthetic edge cases
- understanding training data characteristics

### 4. Dynamic Replica

Use as an optional secondary benchmark if time allows.

Official source:

- https://dynamic-stereo.github.io/

Why it is interesting:

- dynamic scenes
- complementary to DAVIS
- already mentioned by the CoTracker README for evaluation

### 5. Open custom videos

Use for the "our analysis" part of the project.

Recommended sources:

- Pexels Videos: https://www.pexels.com/videos/
- Pixabay Videos: https://pixabay.com/videos/
- Internet Archive / Prelinger: https://archive.org/details/prelinger

Why use them:

- easy to find varied real-world motion
- useful for categories like camera shake, occlusion, blur, crowds, and low texture

Important note:

- always check the license and terms on the source page before reusing clips in a report or presentation
- in this Windows terminal environment, scripted bulk downloads from Pexels and Pixabay ran into Cloudflare interstitial pages, so the reproducible batch collection path was switched to Internet Archive public-domain items

## Current collected open-video set

The current reproducible open-video batch uses ten public-domain Internet Archive items:

- `Design for Dreaming`
- `[Willie Shoemaker at Golden Gate Fields]`
- `Tip-Tops in Peppyland, The`
- `Trading Centers of the Pacific Coast`
- `Street of Memory`
- `Aurora Stunt and Drag Race Set Commercial`
- `Park Conscious`
- `Case of Spring Fever, A`
- `122 Eyes`
- `Sleep for Health`

Their local inventory is written to:

- `project_data/manifests/data_inventory.csv`

## Recommendation

Use this order:

1. repo assets for smoke tests
2. TAP-Vid DAVIS for quantitative baselines
3. custom videos for qualitative failure analysis
4. CoTracker3_Kubric or Dynamic Replica only if we want more depth

## Suggested custom-video categories

- strong camera motion
- object occlusion
- motion blur
- low texture
- scale change
- crowded scenes
- articulated motion

## Local organization

- put downloaded custom clips in `project_data/raw_videos/`
- record metadata in `project_data/manifests/custom_video_catalog_template.csv`
- record observations in `project_results/tables/failure_analysis_template.csv`
