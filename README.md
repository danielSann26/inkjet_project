# Inkjet Scaffold Analyzer

Desktop application for analyzing fluorescence/brightfield microscope images of
microfiber scaffolds. Detects fiber intersections via a double-circle density
algorithm, supports human-in-the-loop labeling with active learning, and
trains a small CNN for binary intersection classification.

## Install

Requires Python 3.10 or 3.11.

```bash
# CPU-only PyTorch (kept separate so installs work on machines without CUDA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Everything else
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

On first launch the app creates `data/annotations.db` and the `data/`,
`models/`, and `logs/` subfolders if they don't already exist.

## Layout

See `Inkjet_Scaffold_Final_Spec.md` (Section 4) for the authoritative folder
structure. All paths stored in the database are relative to the project root.

## Logs

- `logs/app.log` — UI actions, file loads, errors, processing times
- `logs/training.log` — per-epoch metrics, dataset stats, checkpoint saves
