# ECS 170 Project SUDB

Super Ultra Dragon Battlers

This repository contains the ECS 170 Spring 2026 quarter-long AI course project. It is organized into five stages:

1. Stage 1: starter environment, toy data, and baseline methods.
2. Stage 2: tabular/image-like multiclass classification with a PyTorch MLP.
3. Stage 3: image classification with CNN models.
4. Stage 4: text classification and text generation with RNN/LSTM/GRU models.
5. Stage 5: graph node classification with GCN models.

Every stage is runnable through `script/stage_N_script/main.py`. Helper scripts may exist, but `main.py` is the canonical entry point.

## Repository Layout

- `data/stage_N_data/`: stage-specific input data. Large dataset payloads are ignored by git, but the directory structure is preserved.
- `local_code/base_class/`: shared starter abstract classes.
- `local_code/stage_N_code/`: stage-specific datasets, models, settings, evaluators, and utilities.
- `script/stage_N_script/`: runnable stage scripts. Use `main.py` for normal reproduction.
- `result/stage_N_result/`: generated predictions, metrics, plots, summaries, and report artifacts.
- `requirements.txt`: platform-neutral Python packages. PyTorch is installed separately because the correct build depends on hardware.

The stage code folders intentionally migrate forward. Later stages may keep earlier-stage files even when a specific file is not used by that stage.

## Environment Setup

Use Anaconda or Miniconda. The team environment name is `battlers`.

```powershell
conda create -n battlers python=3.12 -y
conda activate battlers
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install PyTorch separately for your machine using the official selector:

https://pytorch.org/get-started/locally/

Common examples:

```powershell
# NVIDIA CUDA example. Confirm the exact CUDA command on the PyTorch website.
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# CPU-only example.
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# macOS example.
python -m pip install torch torchvision torchaudio
```

Verify the environment:

```powershell
python --version
python -c "import numpy, scipy, sklearn, torch, matplotlib, pandas, PIL, networkx, seaborn, tqdm; print('torch', torch.__version__); print('cuda', torch.cuda.is_available())"
```

## Data Placement

All reproducible scripts search under `data/stage_N_data/`. Keep the following local structure:

```text
data/
  stage_1_data/
    toy_data_file.txt
  stage_2_data/
    train.csv
    test.csv
  stage_3_data/
    MNIST/
    ORL/
    CIFAR/
  stage_4_data/
    text_classification/
    text_generation/
  stage_5_data/
    cora/
      node
      link
    pubmed/
      node
      link
    citeseer/
      node
      link
```

Large dataset files are intentionally ignored by git. If a fresh clone cannot run a later stage, download/extract the instructor-provided dataset into the matching folder above.

## General Reproduction

From the repository root:

```powershell
conda activate battlers
python script\stage_N_script\main.py
```

The scripts create result directories automatically under `result/stage_N_result/`.

## Stage 1: Starter Methods

Stage 1 runs the starter MLP path through its `main.py`. The older helper scripts are still present for decision tree, SVM, MLP, and result loading.

Run:

```powershell
python script\stage_1_script\main.py
```

Optional helper scripts:

```powershell
cd script\stage_1_script
python script_decision_tree.py
python script_svm.py
python script_mlp.py
python script_load_result.py
cd ..\..
```

Inputs:

- `data/stage_1_data/toy_data_file.txt`

Typical outputs:

- `result/stage_1_result/DT_prediction_result_*`
- `result/stage_1_result/SVM_prediction_result_*`
- `result/stage_1_result/MLP_prediction_result_*`

## Stage 2: MLP Classification

Stage 2 trains a PyTorch MLP on the pre-split train/test CSV files.

Run:

```powershell
python script\stage_2_script\main.py
```

Inputs:

- `data/stage_2_data/train.csv`
- `data/stage_2_data/test.csv`

Outputs:

- `result/stage_2_result/stage_2_loss_curve.png`
- `result/stage_2_result/MLP_prediction_result_None`
- Console metrics: accuracy plus macro precision, recall, and F1.

Optional result-loader check:

```powershell
python script\stage_2_script\script_load_result.py
```

## Stage 3: CNN Image Classification

Stage 3 trains CNN image classifiers on MNIST, ORL, and CIFAR. It compares a baseline CNN against a shallow ablation model.

Full run:

```powershell
python script\stage_3_script\main.py
```

Quick smoke test:

```powershell
python script\stage_3_script\main.py --quick
```

Run selected datasets or experiments:

```powershell
python script\stage_3_script\main.py --datasets MNIST ORL CIFAR
python script\stage_3_script\main.py --experiments baseline shallow
```

Inputs:

- `data/stage_3_data/MNIST/`
- `data/stage_3_data/ORL/`
- `data/stage_3_data/CIFAR/`

Outputs:

- `result/stage_3_result/stage_3_summary.json`
- `result/stage_3_result/mnist_baseline_loss_curve.png`
- `result/stage_3_result/mnist_shallow_loss_curve.png`
- `result/stage_3_result/orl_baseline_loss_curve.png`
- `result/stage_3_result/orl_shallow_loss_curve.png`
- `result/stage_3_result/cifar_baseline_loss_curve.png`
- `result/stage_3_result/cifar_shallow_loss_curve.png`
- Prediction files named like `MNIST_baseline_CNN_prediction_result_None`.

## Stage 4: RNN Text Classification And Generation

Stage 4 has two tasks:

- Text classification with RNN, LSTM, and GRU.
- Text generation from three seed words with RNN, LSTM, and GRU.

Full run for both tasks:

```powershell
python script\stage_4_script\main.py
```

Quick smoke test:

```powershell
python script\stage_4_script\main.py --quick
```

Run one task:

```powershell
python script\stage_4_script\main.py --task classification
python script\stage_4_script\main.py --task generation
```

Run selected recurrent units:

```powershell
python script\stage_4_script\main.py --models RNN LSTM GRU
python script\stage_4_script\main.py --task classification --models LSTM
python script\stage_4_script\main.py --task generation --models GRU
```

Useful classification options:

```powershell
python script\stage_4_script\main.py --task classification --classification-epochs 10 --classification-batch-size 64 --classification-hidden-dim 128
```

Useful generation options:

```powershell
python script\stage_4_script\main.py --task generation --generation-epochs 30 --generation-seed-words what did the --generation-num-generate 20
```

Inputs:

- `data/stage_4_data/text_classification/`
- `data/stage_4_data/text_generation/data` or `data/stage_4_data/text_generation/data.csv`
- Optional GloVe file at the project root or provided with `--glove-path`.

Outputs:

- `result/stage_4_result/stage_4_text_classification_metrics.csv`
- `result/stage_4_result/stage_4_text_classification_results.json`
- `result/stage_4_result/stage_4_text_classification_curves.png`
- `result/stage_4_result/stage_4_text_classification_confusion_matrices.png`
- `result/stage_4_result/stage_4_text_generation_metrics.csv`
- `result/stage_4_result/stage_4_text_generation_results.json`
- `result/stage_4_result/stage_4_text_generation_curves.png`
- `result/stage_4_result/stage_4_text_generation_samples.txt`

## Stage 5: GCN Node Classification

Stage 5 trains GCN models on Cora, Pubmed, and Citeseer. The default run uses a paper-style split based on Kipf and Welling (2017): 20 labeled nodes per class, 500 validation nodes, and 1000 test nodes.

Full paper-style run:

```powershell
python script\stage_5_script\main.py
```

Quick smoke test:

```powershell
python script\stage_5_script\main.py --quick
```

Run selected datasets:

```powershell
python script\stage_5_script\main.py --datasets cora pubmed citeseer
```

Run selected experiments:

```powershell
python script\stage_5_script\main.py --experiments paper_gcn baseline wide deep low_dropout
```

Run the stronger stratified project split:

```powershell
python script\stage_5_script\main.py --split-mode stratified --train-ratio 0.60 --val-ratio 0.20
```

Run the instructor-provided example split:

```powershell
python script\stage_5_script\main.py --split-mode instructor
```

Inputs:

- `data/stage_5_data/cora/node`
- `data/stage_5_data/cora/link`
- `data/stage_5_data/pubmed/node`
- `data/stage_5_data/pubmed/link`
- `data/stage_5_data/citeseer/node`
- `data/stage_5_data/citeseer/link`

Default experiments:

- `paper_gcn`: 16 hidden units, dropout `0.50`, validation-loss selection, 10 seeds.
- `baseline`: 64 hidden units.
- `wide`: 128 hidden units.
- `deep`: hidden dimensions `128, 64`.
- `low_dropout`: 64 hidden units, dropout `0.35`.
- dataset-tuned variants for Cora, Pubmed, and Citeseer.

Outputs:

- `result/stage_5_result/stage_5_all_metrics.csv`
- `result/stage_5_result/stage_5_best_metrics.csv`
- `result/stage_5_result/stage_5_aggregate_metrics.csv`
- `result/stage_5_result/stage_5_reference_comparison.csv`
- `result/stage_5_result/stage_5_all_results.json`
- `result/stage_5_result/stage_5_cora_training_curves.png`
- `result/stage_5_result/stage_5_pubmed_training_curves.png`
- `result/stage_5_result/stage_5_citeseer_training_curves.png`
- `result/stage_5_result/stage_5_best_performance.png`
- `result/stage_5_result/stage_5_ablation_accuracy.png`
- `result/stage_5_result/stage_5_best_confusion_matrices.png`
- Test prediction CSV files named like `stage_5_cora_baseline_test_predictions.csv`.

Preserved comparison artifacts:

- `result/stage_5_result/stage_5_stratified_best_metrics.csv`
- `result/stage_5_result/stage_5_stratified_all_metrics.csv`
- `result/stage_5_result/stage_5_stratified_all_results.json`

These preserve the stronger stratified-split results and are useful for the Stage 5 report, but the default script now reproduces the paper-style setup.

## Report Artifacts

For reports, use the generated plots and CSVs under the corresponding `result/stage_N_result/` folder.

Stage 5 report priority:

1. Training curves: `stage_5_cora_training_curves.png`, `stage_5_pubmed_training_curves.png`, `stage_5_citeseer_training_curves.png`.
2. Model performance: `stage_5_best_metrics.csv`, `stage_5_best_performance.png`, `stage_5_best_confusion_matrices.png`.
3. Ablation: `stage_5_ablation_accuracy.png`, `stage_5_aggregate_metrics.csv`.
4. Paper comparison: `stage_5_reference_comparison.csv`.

## Git And Cleanup Notes

Commit:

- Source files under `local_code/`.
- Runnable scripts under `script/`.
- README and requirements updates.
- Small report artifacts and result files needed for grading.
- Data folder placeholders and small starter data.

Do not commit:

- Python environments.
- `__pycache__/`.
- `.DS_Store`.
- Large downloaded datasets.
- Duplicate zip archives or nested unzip junk.
- Model checkpoints unless the team explicitly needs them.

If Git auto-cleanup gets stuck in OneDrive, stop retrying, then run:

```powershell
git config gc.auto 0
git reflog expire --expire=now --expire-unreachable=now --all
git gc --prune=now
git count-objects -vH
```

## Reproducibility Checklist

Before submitting or pushing:

- `conda activate battlers`
- Dataset folders exist under `data/stage_N_data/`.
- The stage runs through `script/stage_N_script/main.py`.
- Required metrics are printed or saved.
- Required plots are generated.
- `git status` is clean except for intentional changes.
- Reports stay within the 5-page limit.
