# ecs170-Project-SUDB

Super Ultra Dragon Battlers

This repository is for the ECS 170 Spring 2026 quarter-long AI course project. The project is worth 40% of the course grade and is split into five stages:

1. Stage 1: group formation and programming environment setup.
2. Stage 2: data classification with a PyTorch MLP.
3. Stage 3: image classification with CNN models.
4. Stage 4: text classification and text generation with RNN models.
5. Stage 5: graph embedding and node classification with GNN/GCN models.

Only the stage 1 starter code is implemented right now. The `stage_2_code`, `stage_3_code`, `stage_4_code`, and `stage_5_code` folders currently contain only package placeholders, so the main work is to copy/adapt the stage 1 template for each later stage.

## Repository Layout

- `.gitignore`: ignores IDE files, `.DS_Store`, and Python cache folders.
- `README.md`: project guide and team workflow.
- `data/stage_1_data/`: tracked toy data for stage 1.
- `data/stage_2_data/` through `data/stage_5_data/`: tracked folder structure for later stages. The folders stay in git with `.gitkeep`, but the large downloaded dataset files inside them are ignored.
- `local_code/base_class/`: abstract base classes for datasets, methods, results, settings, and evaluators.
- `local_code/stage_1_code/`: working examples for dataset loading, decision tree, SVM, MLP, result saving/loading, train-test split, k-fold CV, and accuracy evaluation.
- `local_code/stage_2_code/` through `local_code/stage_5_code/`: empty stage folders where future implementations should go.
- `script/stage_1_script/`: runnable stage 1 scripts.
- `result/stage_1_result/`: tracked sample prediction results and the place to keep later stage outputs.

## Environment Setup

Use Anaconda or Miniconda and create a separate conda environment for this project. Do not install packages into `base`. A shared environment name keeps the team organized, while the PyTorch install can differ by each teammate's OS/GPU.

Recommended environment:

- Environment name: `battlers`
- Python: `3.14.4`, or another PyTorch-supported Python `3.10` through `3.14` if needed
- Shared package list: `requirements.txt`
- PyTorch packages: install separately for your machine

Create and activate the project environment:

```powershell
conda create -n battlers python=3.14.4 -y
conda activate battlers
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If Python `3.14.4` is not available on a teammate's system, use the newest Python version supported by the current PyTorch install selector:

```powershell
conda create -n battlers python=3.12 -y
conda activate battlers
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you already have a `battlers` environment, activate it and update the shared packages:

```powershell
conda activate battlers
python -m pip install -r requirements.txt
```

## PyTorch Install By Machine

PyTorch is intentionally not pinned in `requirements.txt`. The correct `torch`, `torchvision`, and `torchaudio` wheels depend on the computer:

- Windows/Linux with a supported NVIDIA GPU: install a CUDA build from the official PyTorch selector. Use the newest CUDA option that supports the GPU and driver.
- MacBook Pro with Apple silicon: install the regular macOS PyTorch build and use the `mps` device if available.
- Windows/Linux without a supported GPU: install the CPU build. It will be slower, but it should still run the course models.
- AMD GPU: do not install CUDA packages. Check current ROCm support for that OS/GPU; if ROCm is not supported, use CPU mode.
- Older GPUs: use the newest PyTorch build supported by the installed driver and hardware. If the newest CUDA build does not work, try an older supported CUDA option from the PyTorch selector.

Official selector: https://pytorch.org/get-started/locally/

Common install examples:

```powershell
# Windows/Linux NVIDIA CUDA example. Check the selector before installing.
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Windows/Linux CPU-only example.
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# macOS example.
python -m pip install torch torchvision torchaudio
```

AMD GPU users should follow the current AMD ROCm/PyTorch compatibility instructions for their exact OS and GPU. ROCm support changes by hardware generation and operating system, so do not copy CUDA commands onto AMD systems.

Check your important versions:

```powershell
python --version
python -c "import numpy, sklearn, torch, torchvision, matplotlib, scipy, PIL, networkx, pandas, seaborn, tqdm; print('numpy', numpy.__version__); print('scikit-learn', sklearn.__version__); print('torch', torch.__version__); print('torchvision', torchvision.__version__); print('matplotlib', matplotlib.__version__); print('scipy', scipy.__version__); print('pillow', PIL.__version__); print('networkx', networkx.__version__); print('pandas', pandas.__version__); print('seaborn', seaborn.__version__); print('tqdm', tqdm.__version__); print('cuda available', torch.cuda.is_available()); print('mps available', hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())"
```

In PyTorch code, prefer automatic device selection so the same script runs on CUDA, MPS, or CPU:

```python
import torch

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    else "cpu"
)
```

If a teammate uses a different PyTorch build because of their machine, mention it in the pull request. Optional packages for Stage 3, Stage 4, and Stage 5 are listed as comments at the bottom of `requirements.txt`; uncomment them only after the team agrees to use that approach.

## Dependency Coverage

The starter project currently imports only Python standard library modules plus `numpy`, `scikit-learn`, and `torch`. The assignment stages suggest these extra needs:

- Stage 1: `numpy`, `scikit-learn`, `torch`.
- Stage 2: `torch`, `numpy`, `scikit-learn`, `matplotlib`, and likely `pandas` for train/test data tables.
- Stage 3: `torch`, `torchvision`, `pillow`, `numpy`, `scikit-learn`, `matplotlib`; `opencv-python` is optional only if image preprocessing needs it.
- Stage 4: `torch`, `numpy`, `scikit-learn`, `matplotlib`; use simple Python tokenization first, and add `nltk` only if the team needs stronger text utilities.
- Stage 5: `torch`, `numpy`, `scipy`, `networkx`, `scikit-learn`, `matplotlib`; `torch-geometric` is optional if the team chooses a PyG implementation instead of writing GCN operations with torch/scipy/networkx.
- Reports and plots: `matplotlib`, `seaborn`, `pandas`, and `tqdm` are included to make experiment tracking, confusion matrices, tables, and progress bars easier.

When a teammate adds code that imports a new external package, update `requirements.txt` in the same pull request.

## Running Stage 1

The stage 1 scripts use relative paths like `../../data/stage_1_data/`, so run them from inside `script/stage_1_script`.

```powershell
New-Item -ItemType Directory -Force result\stage_1_result
$env:PYTHONPATH = (Get-Location).Path
cd script\stage_1_script
python script_decision_tree.py
python script_svm.py
python script_mlp.py
python script_load_result.py
```

On macOS/Linux, use `export PYTHONPATH="$(pwd)"` from the project root before changing into `script/stage_1_script`.

Expected behavior:

- The decision tree, SVM, and MLP scripts load `data/stage_1_data/toy_data_file.txt`.
- The setting object runs k-fold cross validation by default.
- Prediction results are saved as pickle files under `result/stage_1_result/`.
- `script_load_result.py` loads saved SVM prediction results. It currently tries to load folds `1`, `2`, `3`, and `None`; if `SVM_prediction_result_None` does not exist, comment out `None` in that script or run the train-test-split version of the SVM script first.

If a script cannot find the data or result folder, pull the latest branch changes first or create the expected folder if your branch has not added it yet. The team now tracks `result/` and the `data/` folder structure in git, while large downloaded datasets for later stages stay local and are ignored.

## How To Finish The Project

Use the stage 1 code as the pattern for every future stage:

1. Download the instructor-provided dataset and report template for the stage.
2. Put raw datasets under `data/stage_N_data/` and keep the folder names clean so every teammate uses the same paths. For the large later-stage datasets, commit the folder structure only and keep the downloaded dataset files local.
3. Copy the useful stage 1 template files into `local_code/stage_N_code/`.
4. Update imports from `local_code.stage_1_code...` to `local_code.stage_N_code...`.
5. When moving from one stage to the next, migrate the previous stage folder structure forward first. Keep the same code/script filenames unless a new file is truly necessary for the new stage.
6. Write a dataset loader that matches the new dataset format.
7. Write or adapt the model class in a `Method_*.py` file.
8. Write the setting class that loads data, trains the model, tests it, saves results, and evaluates metrics.
9. Add evaluation classes for every metric required by the report.
10. Add a runnable script under `script/stage_N_script/`.
11. Save results, plots, and notes needed for the report.
12. Run the script from its script folder and confirm paths work from a clean checkout.
13. Write the stage report using the instructor template. Keep each report at or under 5 pages.

Keep each stage self-contained. Do not edit stage 1 code for later-stage experiments unless the change is a shared bug fix.

## Stage 2 Plan: MLP Classification

Goal: implement a PyTorch MLP for the provided multiclass dataset.

Main tasks:

- Download the stage 2 train/test dataset and inspect its columns, labels, and file format.
- Copy the stage 1 structure into `local_code/stage_2_code/`.
- Create a stage 2 dataset loader that loads the pre-partitioned train and test sets. The project description says no train-test split or cross validation is needed for stage 2.
- Adapt `Method_MLP.py` so the input dimension, hidden layers, output dimension, loss function, optimizer, and learning rate match the stage 2 dataset.
- Add multiclass metrics such as accuracy, macro F1, weighted F1, macro precision, macro recall, weighted precision, and weighted recall. Do not use binary-only F1/precision/recall.
- Generate a learning convergence curve, usually loss and/or accuracy over epochs.
- Try multiple architectures, loss functions, optimizers, learning rates, epoch counts, and hidden layer sizes.
- Save the final predictions and experiment results.
- Write the stage 2 report with process, settings, curves, results, and comparison of experiments.

Suggested files:

- `local_code/stage_2_code/Dataset_Loader.py`
- `local_code/stage_2_code/Method_MLP.py`
- `local_code/stage_2_code/Evaluate_Accuracy.py`
- `local_code/stage_2_code/Result_Saver.py`
- `local_code/stage_2_code/Result_Loader.py`
- `local_code/stage_2_code/Setting_Train_Test.py`
- `script/stage_2_script/script_mlp.py`
- `script/stage_2_script/script_load_result.py`

Running stage 2:

```powershell
python script\stage_2_script\script_mlp.py
```

How to see stage 2 results:

- Watch the console output for accuracy, macro precision, macro recall, and macro F1.
- Open `result/stage_2_result/stage_2_loss_curve.png` to see the convergence curve.
- Run the saved-result loader:

```powershell
python script\stage_2_script\script_load_result.py
```

- Inspect the saved prediction file at `result/stage_2_result/MLP_prediction_result_None`.

## Stage 3 Plan: CNN Image Classification

Goal: train CNN models for three image datasets: handwritten digits, human faces, and colored objects.

Main tasks:

- Download all three image datasets and inspect image shape, channel count, labels, and train/test layout.
- Build an image dataset loader. For the face dataset, one grayscale channel should be enough even if the files contain equal RGB channels.
- Implement a CNN model with configurable convolution layers, activation functions, pooling, fully connected layers, loss function, and optimizer.
- Train and test one CNN for handwritten digits.
- Train and test separate CNN runs for face images and colored object images.
- Generate learning curves for every dataset.
- Compare multiple CNN configurations: model depth, kernel size, padding, stride, pooling, hidden dimensions, loss function, optimizer, batch size, learning rate, and epoch count.
- Report evaluation results and configuration impacts.

Suggested files:

- `local_code/stage_3_code/Dataset_Loader.py`
- `local_code/stage_3_code/Method_CNN.py`
- `local_code/stage_3_code/Evaluate_Classification.py`
- `local_code/stage_3_code/Result_Saver.py`
- `local_code/stage_3_code/Setting_Image_Classification.py`
- `script/stage_3_script/script_cnn_digits.py`
- `script/stage_3_script/script_cnn_faces.py`
- `script/stage_3_script/script_cnn_objects.py`

## Stage 4 Plan: RNN Text Classification And Generation

Goal: use recurrent models for one text classification task and one text generation task.

Main tasks:

- Download both text datasets and inspect text format, labels, sequence lengths, and train/test layout.
- Create text loaders that tokenize text, build vocabularies, convert tokens to ids, pad or truncate sequences, and create tensors.
- Implement an RNN text classifier.
- Train and evaluate the classifier on the provided test set.
- Implement a text generation model that can generate a story from three starting words.
- Compare generated text with the training data and report correctness/quality observations.
- Repeat experiments with LSTM and GRU units.
- Try architecture and training changes such as embedding size, hidden size, number of layers, dropout, learning rate, batch size, and epoch count.
- Generate learning curves and report results.

Suggested files:

- `local_code/stage_4_code/Dataset_Loader.py`
- `local_code/stage_4_code/Method_RNN_Classifier.py`
- `local_code/stage_4_code/Method_RNN_Generator.py`
- `local_code/stage_4_code/Method_LSTM.py`
- `local_code/stage_4_code/Method_GRU.py`
- `local_code/stage_4_code/Evaluate_Text.py`
- `local_code/stage_4_code/Setting_Text_Classification.py`
- `local_code/stage_4_code/Setting_Text_Generation.py`
- `script/stage_4_script/script_text_classification.py`
- `script/stage_4_script/script_text_generation.py`

## Stage 5 Plan: GNN/GCN Node Classification

Goal: implement graph embedding and node classification with a GCN model on Cora, Pubmed, and Citeseer.

Main tasks:

- Download Cora, Pubmed, and Citeseer datasets and inspect node features, labels, edges, and train/test masks or splits.
- Create a graph dataset loader that returns features, labels, edges/adjacency, and masks/splits.
- Implement a GCN model for node classification.
- Train and evaluate on Cora.
- Repeat the same process for Pubmed and Citeseer.
- Generate learning curves and final evaluation results for all three datasets.
- Save predictions and metrics.
- Write the stage 5 report with dataset descriptions, model setup, curves, results, and comparison across datasets.

Suggested files:

- `local_code/stage_5_code/Dataset_Loader.py`
- `local_code/stage_5_code/Method_GCN.py`
- `local_code/stage_5_code/Evaluate_Node_Classification.py`
- `local_code/stage_5_code/Result_Saver.py`
- `local_code/stage_5_code/Setting_Node_Classification.py`
- `script/stage_5_script/script_gcn_cora.py`
- `script/stage_5_script/script_gcn_pubmed.py`
- `script/stage_5_script/script_gcn_citeseer.py`

## Team Workflow

There are five people on the team including the four collaborators, so split ownership clearly but review each other often.

Recommended ownership:

- Repo/integration lead: keeps `main` clean, reviews pull requests, checks that scripts run, and resolves merge conflicts.
- Stage 2 lead: MLP classification implementation and report draft.
- Stage 3 lead: CNN image classification implementation and report draft.
- Stage 4 lead: RNN/LSTM/GRU text implementation and report draft.
- Stage 5 lead: GCN/GNN implementation and report draft.

Even with owners, everyone should help test and review because each stage is graded separately.

## Git Branch Workflow

Never work directly on `main`. Each teammate should create a branch for their own task, commit only the files they changed, push the branch to GitHub, and open a pull request.

Start new work:

```bash
git checkout main
git pull origin main
git checkout -b yourname/stage-2-mlp
```

Check what changed before committing:

```bash
git status
git diff
```

Stage and commit only your files:

```bash
git add local_code/stage_2_code script/stage_2_script result/stage_2_result README.md .gitignore data/stage_2_data/.gitkeep
git commit -m "Implement stage 2 MLP classifier"
```

Push your branch:

```bash
git push -u origin yourname/stage-2-mlp
```

Then open a pull request on GitHub from `yourname/stage-2-mlp` into `main`.

Before continuing work on an existing branch, update it:

```bash
git checkout main
git pull origin main
git checkout yourname/stage-2-mlp
git merge main
```

If there are conflicts, open the conflicted files, keep the correct parts, remove conflict markers, then run:

```bash
git add path/to/resolved_file.py
git commit
git push
```

Good branch names:

- `yourname/stage-2-loader`
- `yourname/stage-2-mlp`
- `yourname/stage-3-cnn`
- `yourname/stage-4-rnn`
- `yourname/stage-5-gcn`
- `yourname/report-stage-3`

## What To Commit

Commit:

- Source code under `local_code/`.
- Runnable scripts under `script/`.
- Small shared data files that belong in the repo, such as the stage 1 toy dataset.
- Stage data folder placeholders such as `data/stage_2_data/.gitkeep`.
- Result outputs, plots, and evaluation artifacts under `result/` when they are part of the current stage work.
- README or documentation updates.
- Small report files, notes, or generated figures only if the team wants them in GitHub.
- Updates to `requirements.txt` whenever the team adds or changes a package.

Do not commit:

- `.venv/` or local Python environments.
- `__pycache__/` folders.
- `.DS_Store`.
- Large downloaded datasets under `data/stage_2_data/`, `data/stage_3_data/`, `data/stage_4_data/`, and `data/stage_5_data/`.
- Large result files, checkpoints, or model weights unless the team agrees they are required.
- Duplicate unzip junk such as `__MACOSX` folders.
- Redundant zip archives if the extracted dataset folder is already the shared team source of truth.

The current workflow tracks `result/` and the `data/` directory layout, but not the large later-stage dataset payloads. Pull before running so the expected folders exist, then download/extract the instructor datasets into those same local paths.

## Pull Request Checklist

Before asking teammates to review:

- Your branch is up to date with `main`.
- The relevant script runs from its script folder.
- New imports point to the correct `stage_N_code` package.
- Paths work from a clean checkout once local data is present.
- Metrics required by the assignment are printed or saved.
- Learning curves are generated when required.
- Report notes include the exact hyperparameters and final scores.
- Your shared packages satisfy `requirements.txt`, and your pull request explains which PyTorch build you used if it differs from another teammate's.
- If your branch changes `data/` or `result/`, the folder names stay clean and do not reintroduce nested unzip folders or `__MACOSX` junk.
- `git status` does not show accidental files like cache folders, ignored raw dataset payloads, or IDE metadata.

## Stage Definition Of Done

A stage is finished when:

- The correct dataset is loaded.
- The requested model trains without crashing.
- The model is evaluated on the required test data.
- Required metrics are computed.
- Learning curves are generated when requested.
- Results are saved or documented.
- The report is complete and under 5 pages.
- A teammate has reviewed the code or report.
- The branch has been merged into `main`.
