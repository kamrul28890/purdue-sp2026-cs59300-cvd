# Project Scripts

This folder is reserved for project-specific wrappers and analysis scripts.

Recommended additions later:

- a batch runner for custom videos
- a result parser that turns JSON outputs into CSV rows
- a small plotting script for online-vs-offline comparisons

For now, the core commands are:

```powershell
python .\cotracker\evaluation\evaluate.py --config-name eval_tapvid_davis_first exp_dir=.\eval_outputs\tapvid_davis_first_online dataset_root=.\datasets checkpoint=.\checkpoints\scaled_online.pth
python .\cotracker\evaluation\evaluate.py --config-name eval_tapvid_davis_first exp_dir=.\eval_outputs\tapvid_davis_first_offline dataset_root=.\datasets offline_model=True window_len=60 checkpoint=.\checkpoints\scaled_offline.pth
```

For project data collection, use:

```powershell
python .\project_scripts\collect_project_data.py
```

That script:

- copies the official CoTracker repository assets into `project_data/raw_videos/official_assets/`
- verifies the local TAP-Vid DAVIS benchmark payload
- downloads the CoTracker3_Kubric dataset card and one official shard sample
- downloads a curated public-domain Internet Archive video set
- writes `project_data/manifests/data_inventory.csv`

For batch inference on the collected videos, use:

```powershell
python .\project_scripts\batch_run_collected_videos.py
```

That script:

- scans `project_data/raw_videos/official_assets/` and `project_data/raw_videos/internet_archive/`
- runs CoTracker in online and offline modes on a deterministic clip from each video
- saves rendered outputs to `project_results/videos/<run_name>/`
- writes tracked CSV summaries to `project_results/tables/`
