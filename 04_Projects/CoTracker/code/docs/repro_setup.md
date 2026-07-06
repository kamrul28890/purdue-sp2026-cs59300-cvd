## Reproducibility Notes

This repo started from the upstream `facebookresearch/co-tracker` codebase and now includes a small set of local changes to make setup and verification easier:

- `scripts/setup_venv.ps1`: Windows PowerShell environment bootstrapper
- `scripts/setup_venv.sh`: Unix shell bootstrapper
- `requirements-repro.txt`: portable dependency snapshot for a fresh local install
- `requirements-lock.txt`: exact local environment snapshot captured from one working machine
- `Dockerfile`: containerized setup path
- `tests/test_bilinear_sample.py`: guarded `unittest.main()` so it works both with `pytest` and direct execution

## Recommended Reproduction Path

Use Python 3.10.

### Windows

```powershell
winget install --id Python.Python.3.10 -e --accept-source-agreements --accept-package-agreements
.\scripts\setup_venv.ps1
. .\.venv\Scripts\Activate.ps1
python -m pytest tests/test_bilinear_sample.py -q
```

### Linux or macOS

```bash
./scripts/setup_venv.sh
source .venv/bin/activate
python -m pytest tests/test_bilinear_sample.py -q
```

## Dependency Files

- `requirements-repro.txt` is the recommended default. It installs a portable dependency set and then uses `pip install -e .` so the local checkout is the package being tested.
- `requirements-lock.txt` is an exact freeze from one existing environment. It is useful as a reference snapshot, but it may be platform-specific because it includes transitive dependencies and CUDA-specific wheels.

If you want to use the exact snapshot instead of the portable dependency file:

- PowerShell: `.\scripts\setup_venv.ps1 -UseLockfile`
- Bash: `./scripts/setup_venv.sh --use-lockfile`

## Docker

Build the portable image:

```bash
docker build --build-arg MODE=portable -t cotracker:portable .
```

Build from the exact local lock snapshot:

```bash
docker build --build-arg MODE=lockfile -t cotracker:lockfile .
```

## What To Run Next

After the environment is up:

1. Run the smoke test:

```bash
python -m pytest tests/test_bilinear_sample.py -q
```

2. Try the offline demo:

```bash
python demo.py --grid_size 10
```

3. For model evaluation or training, follow the dataset and checkpoint instructions already documented in `README.md`.
