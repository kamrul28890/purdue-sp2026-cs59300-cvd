# CS59300CVD HW4

This repository contains a complete implementation for Assignment 4:

- Part 1: Fundamental matrix estimation, camera calibration, triangulation
- Part 2: Affine factorization (Tomasi-Kanade)

## Environment

- Python: 3.10.11
- OS: Windows
- Core packages:
  - torch
  - torchvision
  - numpy
  - matplotlib

## Install

From the workspace root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch torchvision numpy matplotlib
```

## Run Part 1

```powershell
cd starter_code\part1
python main.py -i library
python main.py -i lab
```

Generated outputs are saved to `starter_code/part1/results`.

- Match visualizations: `*-matches.png`
- Epipolar line visualizations: `*-F_unnorm.png`, `*-F_norm.png`
- Console output includes:
  - Fundamental matrices (unnormalized, normalized)
  - Fundamental residual MSE
  - Camera projection matrices
  - Reprojection MSE
  - Triangulation 3D MSE (lab)

## Run Part 2

```powershell
cd starter_code\part2
python main.py
```

Generated outputs are saved to `starter_code/part2/results`.

- 3D structure before metric upgrade: `S1-*.png`
- 3D structure after metric upgrade: `S2-*.png`
- Frame-by-frame keypoint comparisons: `k*.png`
- Framewise MSE plot: `mse.png`

## Reproducibility Notes

- All result figures in the report should be generated directly from script outputs.
- If results differ across runs, confirm package versions and Python version.
- For grading, include:
  - PDF report built from `report_template`
  - Source code zip including this README

## Report Build

```powershell
cd report_template
pdflatex main.tex
pdflatex main.tex
```

Use the provided template and edit `report_template/student_response/report.tex`.
