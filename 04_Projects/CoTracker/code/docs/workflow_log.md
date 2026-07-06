# Workflow Log

## Purpose

This document keeps a continuous chronology of the reproduction workflow, environment checks, execution results, and code changes made while adapting and validating the CoTracker3 repository for this course project.

## Chronology

### 2026-03-18 18:42:09 -04:00

- Created this log to track work continuously rather than reconstructing it later.
- Current goal for this phase:
  - download the CoTracker3 checkpoints
  - run explicit online and offline demos with checkpoint arguments
  - set up the TAP-Vid DAVIS evaluation path
  - run the first evaluation command if the dataset becomes available
- Current repository context:
  - `origin` points to the course project GitHub repo
  - `upstream` points to the official Meta CoTracker repository
  - previous Windows compatibility patches for training signal handling have already been applied and pushed

### 2026-03-18 18:45:36 -04:00

- Created the local `checkpoints/` directory.
- Downloaded the explicit CoTracker3 checkpoint files referenced by the repository:
  - `checkpoints/scaled_online.pth`
  - `checkpoints/scaled_offline.pth`
- Downloaded file sizes:
  - `scaled_online.pth`: `101,695,610` bytes
  - `scaled_offline.pth`: `101,890,938` bytes
- This step makes the demos and evaluation commands reproducible without relying on implicit `torch.hub` checkpoint download behavior.

### 2026-03-18 18:46:28 -04:00

- Ran explicit checkpoint demos from the local repository instead of relying on `torch.hub` model selection:
  - `python demo.py --grid_size 10 --checkpoint .\checkpoints\scaled_online.pth`
  - `python demo.py --grid_size 10 --checkpoint .\checkpoints\scaled_offline.pth --offline`
- Both commands completed successfully and printed `computed`.
- Preserved the outputs as:
  - `saved_videos/demo_online_explicit.mp4`
  - `saved_videos/demo_offline_explicit.mp4`
- Observed one non-blocking compatibility note during checkpoint loading:
  - PyTorch emitted a `FutureWarning` about `torch.load(..., weights_only=False)` in `cotracker/models/build_cotracker.py`
  - This is not a current execution failure, but it is worth noting for future maintenance

### 2026-03-18 19:13:12 -04:00

- Began TAP-Vid DAVIS setup for the first benchmark run.
- Verified from the CoTracker evaluation code that the expected DAVIS path is:
  - `dataset_root/tapvid_davis/tapvid_davis.pkl`
- First dataset download attempt used `Invoke-WebRequest` and produced a corrupted partial archive:
  - file name: `datasets/tapvid_davis.zip`
  - size observed before retry: `203,578,094` bytes
  - extraction and decompression checks failed
- Retried the dataset download with `curl.exe`, which completed successfully.
- Verified the extracted DAVIS payload:
  - `datasets/tapvid_davis/tapvid_davis.pkl`
  - size: `2,481,403,560` bytes
- Kept the auxiliary files that shipped with the archive:
  - `datasets/tapvid_davis/README.md`
  - `datasets/tapvid_davis/SOURCES.md`
- Ran the first full DAVIS evaluation with the explicit online checkpoint:
  - `python .\cotracker\evaluation\evaluate.py --config-name eval_tapvid_davis_first exp_dir=.\eval_outputs\tapvid_davis_first_online dataset_root=.\datasets checkpoint=.\checkpoints\scaled_online.pth`
- Evaluation completed successfully.
- Runtime recorded in the result JSON:
  - `62.29804277420044` seconds
- Result artifact written to:
  - `eval_outputs/tapvid_davis_first_online/result_eval_.json`
- Baseline metrics from that first run:
  - `occlusion_accuracy`: `0.9088604772803294`
  - `average_jaccard`: `0.6444301091643813`
  - `average_pts_within_thresh`: `0.771162818724843`
- Interpretation note:
  - this is a valid first benchmark run and a useful project baseline
  - it should not yet be treated as a final paper-comparison claim without checking whether every evaluation setting matches the paper protocol exactly

### 2026-03-18 19:13:56 -04:00

- Updated `.gitignore` to keep the repository clean after the new execution artifacts were created.
- Added ignore rules for:
  - `datasets/`
  - `*.log`
- Reason for this change:
  - the downloaded DAVIS benchmark data are large and should remain local
  - generated console logs are workflow artifacts, not source files

### 2026-03-18 20:00:31 -04:00

- Ran the matching offline DAVIS baseline to complement the earlier online result:
  - `python .\cotracker\evaluation\evaluate.py --config-name eval_tapvid_davis_first exp_dir=.\eval_outputs\tapvid_davis_first_offline dataset_root=.\datasets offline_model=True window_len=60 checkpoint=.\checkpoints\scaled_offline.pth`
- Evaluation completed successfully and wrote:
  - `eval_outputs/tapvid_davis_first_offline/result_eval_.json`
- Runtime recorded in the result JSON:
  - `505.74844765663147` seconds
- Offline baseline metrics:
  - `occlusion_accuracy`: `0.9173205871355311`
  - `average_jaccard`: `0.6480599112594341`
  - `average_pts_within_thresh`: `0.7688051147336501`
- Online baseline metrics from the earlier run for direct comparison:
  - `occlusion_accuracy`: `0.9088604772803294`
  - `average_jaccard`: `0.6444301091643813`
  - `average_pts_within_thresh`: `0.771162818724843`
  - `time`: `62.29804277420044` seconds
- Immediate comparison takeaway:
  - offline performed slightly better on `average_jaccard`
  - online performed slightly better on `average_pts_within_thresh`
  - offline was dramatically slower on this machine and configuration
- Project implication:
  - this gives us a clean first online-vs-offline comparison table for the proposal and final report
  - the next step should shift from setup to analysis on custom videos and failure modes

### 2026-03-18 22:50:55 -04:00

- Created project-facing structure for the next phase of work:
  - `project_data/`
  - `project_results/`
  - `project_scripts/`
- Added tracked templates and summaries so the project can move from setup into analysis:
  - `project_data/manifests/custom_video_catalog_template.csv`
  - `project_results/tables/benchmark_summary.csv`
  - `project_results/tables/failure_analysis_template.csv`
- Pre-populated `project_results/tables/benchmark_summary.csv` with the completed online and offline DAVIS baselines.
- Added documentation for video sourcing and project direction:
  - `docs/data_sources.md`
  - `docs/paper_summary.md`
- Added ignore rules to prevent accidental commits of large custom video artifacts:
  - `project_data/raw_videos/`
  - `project_results/videos/`
  - `project_results/runs/`
- Outcome of this step:
  - the repo now has a clear place for open-source videos, tracked experiment summaries, and plain-language paper notes
  - this should make the next custom-video analysis phase much easier to execute and write up

### 2026-03-18 23:24:28 -04:00

- Resumed project data collection in the requested priority order:
  - official repository assets
  - TAP-Vid DAVIS
  - CoTracker3 broader data
  - open-source custom videos
- Added a reproducible collection script:
  - `project_scripts/collect_project_data.py`
- Expanded project-side data documentation so the collection process is easier to rerun:
  - `project_data/README.md`
  - `project_scripts/README.md`
  - `docs/data_sources.md`
- Completed the official repository asset copy into `project_data/raw_videos/official_assets/`:
  - `apple.mp4`
  - `apple_mask.png`
  - `bmx-bumps.gif`
- Re-verified that TAP-Vid DAVIS is already available locally and ready for evaluation:
  - `datasets/tapvid_davis/tapvid_davis.pkl`
  - size confirmed in the inventory: `2,481,403,560` bytes
- Kept the broader CoTracker-side data path reproducible by caching:
  - `project_data/kubric_cache/CoTracker3_Kubric_README.md`
  - `project_data/kubric_cache/0000.tar.gz`
- Reconfirmed the extracted official CoTracker3_Kubric sample:
  - `project_data/raw_videos/kubric_sample/0000/`
- Important implementation note about the Kubric shard:
  - the extracted sample is not a ready-made MP4-style video folder
  - the shard contains arrays and annotations such as trajectories, visibility, depth, and metadata
  - the official dataset loader expects a dataset layout with rendered `frames/` plus sequence annotations, so this shard should be treated as a broader-data sample rather than a drop-in demo video
- For the open custom-video collection, reproducible scripted downloads from Pexels and Pixabay were not used because they returned Cloudflare interstitial pages in this terminal environment.
- Switched the reproducible batch collection path to public-domain Internet Archive items and downloaded ten videos into:
  - `project_data/raw_videos/internet_archive/`
- Downloaded Internet Archive set:
  - `Designfo1956_512kb.mp4`
  - `WillieSh1950_512kb.mp4`
  - `TipTopsi1934_512kb.mp4`
  - `TradingC1947_512kb.mp4`
  - `Streetof1937_512kb.mp4`
  - `aurora_drag_race_set_512kb.mp4`
  - `ParkCons1938_512kb.mp4`
  - `CaseofSp1940_512kb.mp4`
  - `122Eyes1950_512kb.mp4`
  - `Sleepfor1950_512kb.mp4`
- Aggregate size of the downloaded Internet Archive video set:
  - `327,351,932` bytes
- Wrote the tracked inventory manifest:
  - `project_data/manifests/data_inventory.csv`
- Outcome of this step:
  - the repo now has a reproducible project-data collection script
  - the requested data tiers are all represented locally
  - we have a clean, licensed, and tracked custom-video set ready for qualitative analysis

### 2026-03-18 23:25:25 -04:00

- Tightened repository hygiene after the data collection pass.
- Updated `.gitignore` to ignore:
  - `project_data/kubric_cache/`
- Reason for this change:
  - the local CoTracker3_Kubric shard cache is large and should remain a local download artifact
  - the tracked manifest and workflow log already preserve the reproducible record of what was downloaded

### 2026-03-20 02:00:30 -04:00

- Started the first batch custom-video inference pass on the collected project videos.
- Added a project-side batch runner:
  - `project_scripts/batch_run_collected_videos.py`
- Updated project readmes so the new run path is documented:
  - `project_scripts/README.md`
  - `project_results/README.md`
- Batch runner scope for this pass:
  - scanned `project_data/raw_videos/official_assets/`
  - scanned `project_data/raw_videos/internet_archive/`
  - skipped non-video files automatically by extension
  - ran both `online` and `offline` CoTracker3 modes
  - used `grid_size=10`
- Deterministic clip rule used for this run:
  - short repository demo clips were processed from frame `0`
  - long Internet Archive videos were limited to `180` frames
  - long-video clips started at `10%` into the source video to reduce title-card bias
- Output layout for this run:
  - rendered videos: `project_results/videos/batch_collected_20260320/`
  - tracked run summary: `project_results/tables/batch_collected_20260320_summary.csv`
  - tracked failure-analysis seed table: `project_results/tables/batch_collected_20260320_failure_analysis.csv`
- Execution outcome:
  - completed jobs: `24 / 24`
  - device used: `cuda`
  - no inference jobs failed
- Runtime summary from the tracked CSV:
  - online runs: `12`, total `2325.727` seconds, average `193.811` seconds
  - offline runs: `12`, total `847.335` seconds, average `70.611` seconds
- Interpretation note:
  - for these short fixed-length qualitative clips, the offline path was faster on average than the online path on this machine
  - this differs from the earlier full-DAVIS benchmark behavior and is worth mentioning later as an implementation/configuration observation rather than a universal claim
- Non-blocking warnings observed during the batch run:
  - the same `torch.load(..., weights_only=False)` future warning appeared during checkpoint loading
  - `imageio` resized the rendered apple output height slightly for codec compatibility because the padded frame size was not divisible by `16`
- Outcome of this step:
  - we now have a full first-pass custom-video output set for qualitative inspection
  - the failure-analysis table is seeded with direct links to every rendered output
  - the project has moved beyond setup into repeatable qualitative experimentation
