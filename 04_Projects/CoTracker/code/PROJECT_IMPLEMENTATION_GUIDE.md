# CoTracker3 Project Implementation Guide

## Project framing

**Course:** CS 59300 CVD  
**Paper:** CoTracker3: Simpler and Better Point Tracking by Pseudo-Labelling Real Videos  
**Primary goal:** Reproduce the main CoTracker3 evaluation and then present a clear, well-structured version of the project in our own style.  
**Secondary goal:** Add one meaningful extension after the baseline reproduction is stable.

This project should read as:

1. We understand the point tracking problem.
2. We understand how CoTracker3 is organized and evaluated.
3. We can reproduce baseline behavior with released checkpoints.
4. We can analyze where it works, where it fails, and why.
5. We can extend it in a technically sensible way.

## Recommended project scope

Use a two-stage scope.

**Stage 1: Strong reproduction**
- Run local demos successfully.
- Reproduce evaluation on at least one TAP-Vid split.
- Inspect the model, predictor, dataset, and evaluation code.
- Build our own figures, tables, and qualitative examples.

**Stage 2: Small but real extension**
- Add one extension that makes the project feel like ours.
- Keep the extension lightweight enough to finish and evaluate.

Best extension choices, in order:

1. **Camera-motion-aware analysis**
   Compare tracking behavior on videos with strong camera motion versus mild camera motion.
2. **Custom real-video benchmark**
   Collect a small set of our own videos and evaluate qualitative robustness under occlusion, fast motion, and textureless regions.
3. **Geometry-aware extension**
   Use an off-the-shelf monocular depth or motion estimation tool to stabilize or analyze tracks in a pseudo-3D way.

For the course, a strong reproduction plus a clean custom analysis is better than an over-ambitious unfinished method.

## Repo map

These are the most important files to understand:

- [README.md](/d:/Purdue/Courses/02.%20Spring%202026%20CS%2059300-CVD/Project/co-tracker/README.md)
- [demo.py](/d:/Purdue/Courses/02.%20Spring%202026%20CS%2059300-CVD/Project/co-tracker/demo.py)
- [online_demo.py](/d:/Purdue/Courses/02.%20Spring%202026%20CS%2059300-CVD/Project/co-tracker/online_demo.py)
- [hubconf.py](/d:/Purdue/Courses/02.%20Spring%202026%20CS%2059300-CVD/Project/co-tracker/hubconf.py)
- [cotracker/models/build_cotracker.py](/d:/Purdue/Courses/02.%20Spring%202026%20CS%2059300-CVD/Project/co-tracker/cotracker/models/build_cotracker.py)
- [cotracker/models/evaluation_predictor.py](/d:/Purdue/Courses/02.%20Spring%202026%20CS%2059300-CVD/Project/co-tracker/cotracker/models/evaluation_predictor.py)
- [cotracker/utils/visualizer.py](/d:/Purdue/Courses/02.%20Spring%202026%20CS%2059300-CVD/Project/co-tracker/cotracker/utils/visualizer.py)
- [cotracker/datasets/tap_vid_datasets.py](/d:/Purdue/Courses/02.%20Spring%202026%20CS%2059300-CVD/Project/co-tracker/cotracker/datasets/tap_vid_datasets.py)
- [cotracker/datasets/real_dataset.py](/d:/Purdue/Courses/02.%20Spring%202026%20CS%2059300-CVD/Project/co-tracker/cotracker/datasets/real_dataset.py)
- [cotracker/evaluation/evaluate.py](/d:/Purdue/Courses/02.%20Spring%202026%20CS%2059300-CVD/Project/co-tracker/cotracker/evaluation/evaluate.py)
- [cotracker/evaluation/configs/eval_tapvid_davis_first.yaml](/d:/Purdue/Courses/02.%20Spring%202026%20CS%2059300-CVD/Project/co-tracker/cotracker/evaluation/configs/eval_tapvid_davis_first.yaml)

## Working style for this repo

Use this repo layout convention:

- `main`: our project branch
- `upstream`: official Meta CoTracker repository
- do not commit `.venv/`, `checkpoints/`, datasets, or generated videos

Suggested folders we will add later if needed:

- `project_docs/` for proposal notes, report outlines, and presentation material
- `project_scripts/` for our custom analysis and evaluation helpers
- `project_results/` for saved CSV summaries and selected figures only

## Phase 0: Environment and sanity checks

Complete this before any experiments:

```powershell
.\.venv\Scripts\Activate.ps1
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python demo.py --help
python online_demo.py --help
```

Success criteria:

- PyTorch loads correctly
- CUDA is available
- demo scripts parse correctly

Important note:

- if CUDA is `False`, fix PyTorch before doing anything expensive

## Phase 1: Baseline demo reproduction

### 1. Download checkpoints

Use PowerShell-friendly commands:

```powershell
New-Item -ItemType Directory -Force checkpoints | Out-Null
Invoke-WebRequest https://huggingface.co/facebook/cotracker3/resolve/main/scaled_online.pth -OutFile .\checkpoints\scaled_online.pth
Invoke-WebRequest https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth -OutFile .\checkpoints\scaled_offline.pth
```

Optional baseline-only checkpoints:

```powershell
Invoke-WebRequest https://huggingface.co/facebook/cotracker3/resolve/main/baseline_online.pth -OutFile .\checkpoints\baseline_online.pth
Invoke-WebRequest https://huggingface.co/facebook/cotracker3/resolve/main/baseline_offline.pth -OutFile .\checkpoints\baseline_offline.pth
```

### 2. Run demo videos

Online:

```powershell
python demo.py --grid_size 10 --checkpoint .\checkpoints\scaled_online.pth
```

Offline:

```powershell
python demo.py --grid_size 10 --checkpoint .\checkpoints\scaled_offline.pth --offline
```

### 3. Save qualitative outputs

Collect a small curated set of examples:

- smooth motion
- fast motion
- occlusion
- large camera motion
- low texture

Save only representative outputs for the final report.

## Phase 2: Read the code with purpose

Do not read the whole repo line by line. Read it in this order:

1. `demo.py`
2. `hubconf.py`
3. `cotracker/models/evaluation_predictor.py`
4. `cotracker/models/build_cotracker.py`
5. `cotracker/evaluation/evaluate.py`
6. `cotracker/datasets/tap_vid_datasets.py`

Questions to answer while reading:

- How are query points initialized?
- What is the difference between online and offline inference?
- What inputs and outputs define the predictor API?
- Where is visibility predicted?
- How does the evaluation pipeline load TAP-Vid data?
- What metrics are reported?

We should turn these answers into 4 to 6 clean slides for the final presentation.

## Phase 3: Reproduce paper-style evaluation

### 1. Install evaluation extras

```powershell
pip install hydra-core==1.1.0 mediapy
```

### 2. Download evaluation data

Start with at least one TAP-Vid subset:

- DAVIS
- Kinetics
- RoboTAP
- RGB-Stacking

Recommended first target:

- **TAP-Vid DAVIS**, because it is manageable and visually intuitive

### 3. Run evaluation

Online model:

```powershell
python .\cotracker\evaluation\evaluate.py --config-name eval_tapvid_davis_first exp_dir=.\eval_outputs dataset_root=YOUR_TAPVID_PATH checkpoint=.\checkpoints\scaled_online.pth
```

Offline model:

```powershell
python .\cotracker\evaluation\evaluate.py --config-name eval_tapvid_davis_first exp_dir=.\eval_outputs dataset_root=YOUR_TAPVID_PATH offline_model=True window_len=60 checkpoint=.\checkpoints\scaled_offline.pth
```

### 4. Record reproducibility notes

For every run, log:

- checkpoint used
- config name
- dataset split
- GPU used
- runtime
- any code modification
- output metrics

This will save a huge amount of time when writing the report.

## Phase 4: Build the project in our own way

This is where reproduction becomes a course project rather than a repo rerun.

### Option A: Camera-motion-aware analysis

Main question:

- Does CoTracker3 degrade more under camera motion or object motion?

Implementation idea:

- collect or select videos with strong camera shake and weak camera motion
- compare qualitative track stability and visibility confidence
- group failure cases by motion type

Deliverables:

- curated video set
- side-by-side track visualizations
- short analysis table

### Option B: Custom real-video benchmark

Main question:

- How well does CoTracker3 generalize to our own videos without fine-tuning?

Implementation idea:

- collect 10 to 20 short videos
- label categories instead of dense ground truth:
  - occlusion
  - scale change
  - blur
  - low texture
  - camera motion
- report qualitative robustness and common failure modes

Deliverables:

- dataset sheet
- figure grid of successful and failed examples
- category-based analysis

### Option C: Geometry-aware extension

Main question:

- Can pseudo-3D reasoning improve analysis or stabilization of tracks?

Implementation idea:

- run a monocular depth estimator or camera-motion estimator on the same videos
- compare image-space tracks versus geometry-aware normalized tracks
- focus on analysis even if the method is simple

Deliverables:

- one pipeline diagram
- one comparison table
- one honest discussion of limitations

## Recommended final project story

This is the story I recommend using:

**Title draft:**  
Reproducing and Analyzing CoTracker3 for Robust Point Tracking in Real Videos

**Narrative:**

- Reproduce CoTracker3 inference and evaluation
- Benchmark on TAP-Vid DAVIS
- Analyze real-world behavior on custom videos
- Study failure modes under occlusion and camera motion
- Optionally add a small geometry-aware or motion-aware analysis layer

This is strong because it is:

- technically grounded
- visibly impressive in the presentation
- realistic to finish
- easy to write up in ICCV style

## Minimal deliverables checklist

For the proposal:

- problem statement
- 3 to 5 related papers
- reproducibility plan
- extension hypothesis

For the final report:

- method overview of CoTracker3
- reproducibility setup
- benchmark results
- qualitative examples
- failure analysis
- extension results

For the final presentation:

- one-slide motivation
- one-slide method summary
- one-slide repo/pipeline overview
- one-slide reproduction setup
- one-slide results
- one-slide failure analysis
- one-slide conclusion

## Concrete milestones

### Milestone 1

- environment works
- checkpoints downloaded
- demo video produced

### Milestone 2

- evaluation runs on one TAP-Vid split
- baseline metrics saved

### Milestone 3

- custom analysis pipeline created
- qualitative examples collected

### Milestone 4

- extension completed
- tables and figures finalized

### Milestone 5

- proposal, report, and presentation aligned to the same story

## Suggested task breakdown

### Reproduction tasks

- verify environment
- inspect predictor and model code
- download checkpoints
- run demo and evaluation
- store metrics and outputs

### Analysis tasks

- identify success and failure cases
- collect custom videos
- define categories
- prepare visual comparisons

### Writing tasks

- maintain running notes while experimenting
- convert notes into report-ready figures and tables
- write method and experimental setup from code-backed evidence

## Risks and how to manage them

### Risk 1: Evaluation dataset setup takes too long

Mitigation:

- start with one TAP-Vid subset only
- do not wait for every benchmark before writing

### Risk 2: Training is too expensive

Mitigation:

- do not plan full CoTracker3 training as the main contribution
- use released checkpoints for the main reproduction

### Risk 3: Extension scope explodes

Mitigation:

- keep the extension analysis-first
- do not promise a heavy re-training pipeline unless time clearly allows it

### Risk 4: Fine-tuning paths need extra dependencies

Mitigation:

- `train_on_real_data.py` depends on external TAPIR/TAPNet code and checkpoints that are not installed by default
- treat pseudo-label fine-tuning as an optional stretch goal, not the baseline plan

## What not to do

- do not make full training the core project
- do not promise all benchmarks unless we already have them running
- do not spend most of the time polishing UI or demos
- do not turn the project into a vague VLM survey after choosing CoTracker3

## Immediate next steps

1. Verify GPU-enabled PyTorch is active.
2. Download the `scaled_online` and `scaled_offline` checkpoints.
3. Run one online and one offline demo.
4. Set up TAP-Vid DAVIS evaluation.
5. Decide the extension track after the first benchmark is stable.
