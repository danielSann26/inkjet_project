# Inkjet Scaffold Analyzer

Desktop application for analyzing fluorescence/brightfield microscope images of
microfiber scaffolds. Detects fiber intersections via a double-circle density
algorithm, supports human-in-the-loop labeling with active learning, and
trains a small CNN for binary intersection classification.

## Install

Requires Python 3.11

## Activate Virtual Environment

```bash
python -m venv .venv # create virtual environment
.venv/Scripts/activate # use this in powershell to activate virtual environment
```

```bash
# CPU-only PyTorch (kept separate so installs work on machines without CUDA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Everything else
pip install -r requirements.txt
```

## Run

```bash
python main.py # for entire app
python preprocess.py ./images/filename # use this to preprocess image which will save under data/preprocessed
python train.py ./data/preprocessed/filename # use this to train model, click intersections that you see to reaffirm

```

On first launch the app creates `data/annotations.db` and the `data/`,
`models/`, and `logs/` subfolders if they don't already exist.

## Logs

- `logs/app.log` — UI actions, file loads, errors, processing times
- `logs/training.log` — per-epoch metrics, dataset stats, checkpoint saves
