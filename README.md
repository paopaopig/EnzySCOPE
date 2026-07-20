<div align="center">

# EnzySCOPE

**A structure-aware enzyme reaction prediction model for enzyme-substrate-product screening**

Reproducible test-case inference with checkpoints from **seed 1231 to seed 1240**.

</div>

---

## Overview

**EnzySCOPE** is a structure-aware enzyme reaction prediction model for enzyme-substrate-product screening. This repository provides a reproducible test case with checkpoints from seed 1231 to seed 1240.

## Project Structure

```text
configs/test_case.json        Test prediction config
configs/train_config.json     Training workflow config
src/predict.py                Run test prediction
src/train.py                  Training entry point
src/model.py                  EnzySCOPE model
data/test_case/               Test data and graph files
checkpoints/                  test_seed_<seed>.pt
outputs/                      Prediction outputs
```

## Installation

Check the existing PyTorch/CUDA environment first. EnzySCOPE requires **PyTorch >= 2.0.0**.

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

Install prediction dependencies:

```bash
pip install -r requirements.txt
```

## Test Prediction

Run from the project root:

```bash
python src/predict.py
```

Default behavior:

```text
config: configs/test_case.json
seeds: 1231-1240
output: outputs/
```

## Full Pipeline

```bash
python src/embed_molecules.py --molecule-csv data/train/train_molecules.csv --output-pkl data/train/train_molecule_embeddings.pkl
python src/embed_enzymes.py --enzyme-csv data/train/train_enzymes.csv --output-pt data/train/train_enzyme_embeddings.pt
python src/build_graphs.py --enzyme-csv data/train/train_enzymes.csv --structure-dir raw_structures --output-dir data/train/train_protein_graphs
python src/train.py --config configs/train_config.json --epochs 100 --seed 1234
python src/predict.py
```
