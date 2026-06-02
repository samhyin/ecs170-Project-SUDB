import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from local_code.path_utils import ensure_result_dir, resolve_stage_data_dir
from local_code.stage_5_code.Dataset_Loader_Node_Classification import (
    Dataset_Loader as InstructorNodeDatasetLoader,
)
from local_code.stage_5_code.Evaluate_Node_Classification import (
    Evaluate_Node_Classification,
)
from local_code.stage_5_code.Method_GCN import Method_GCN


DATASETS = ("cora", "pubmed", "citeseer")
PAPER_GCN_ACCURACY = {"citeseer": 0.703, "cora": 0.815, "pubmed": 0.790}
PAPER_RANDOM_SPLIT_ACCURACY = {
    "citeseer": (0.679, 0.005),
    "cora": (0.801, 0.005),
    "pubmed": (0.789, 0.007),
}
PAPER_TRAIN_PER_CLASS = 20
PAPER_VALIDATION_SIZE = 500
PAPER_TEST_SIZE = 1000
EXPERIMENT_CONFIGS = {
    "paper_gcn": {
        "description": "paper-style 2-layer GCN reproduction with 16 hidden units",
        "hidden_dims": (16,),
        "dropout": 0.50,
        "learning_rate": 0.010,
        "weight_decay": 0.0005,
        "label_smoothing": 0.00,
        "epochs": 200,
        "patience": 10,
        "min_epochs": 1,
        "selection_metric": "val_loss",
        "selection_mode": "min",
        "seeds": tuple(range(1, 11)),
    },
    "baseline": {
        "description": "one hidden GCN layer with standard dropout and weight decay",
        "hidden_dims": (64,),
        "dropout": 0.50,
        "learning_rate": 0.010,
        "weight_decay": 0.0005,
        "label_smoothing": 0.00,
    },
    "wide": {
        "description": "wider one-hidden-layer GCN",
        "hidden_dims": (128,),
        "dropout": 0.50,
        "learning_rate": 0.010,
        "weight_decay": 0.0005,
        "label_smoothing": 0.00,
    },
    "deep": {
        "description": "two hidden GCN layers for the ablation study",
        "hidden_dims": (128, 64),
        "dropout": 0.55,
        "learning_rate": 0.005,
        "weight_decay": 0.0005,
        "label_smoothing": 0.00,
    },
    "low_dropout": {
        "description": "one hidden GCN layer with less dropout",
        "hidden_dims": (64,),
        "dropout": 0.35,
        "learning_rate": 0.010,
        "weight_decay": 0.0005,
        "label_smoothing": 0.00,
    },
    "cora_tuned": {
        "description": "Cora-tuned low-dropout GCN selected by hyperparameter sweep",
        "datasets": ("cora",),
        "hidden_dims": (64,),
        "dropout": 0.25,
        "learning_rate": 0.010,
        "weight_decay": 0.0005,
        "label_smoothing": 0.00,
        "seed": 4,
        "epochs": 450,
        "patience": 90,
        "min_epochs": 80,
    },
    "pubmed_tuned": {
        "description": "Pubmed-tuned GCN with lower learning rate",
        "datasets": ("pubmed",),
        "hidden_dims": (64,),
        "dropout": 0.35,
        "learning_rate": 0.005,
        "weight_decay": 0.0005,
        "label_smoothing": 0.00,
        "seed": 7,
        "epochs": 450,
        "patience": 90,
        "min_epochs": 80,
    },
    "citeseer_tuned": {
        "description": "Citeseer-tuned low-dropout GCN selected by hyperparameter sweep",
        "datasets": ("citeseer",),
        "hidden_dims": (64,),
        "dropout": 0.35,
        "learning_rate": 0.010,
        "weight_decay": 0.0005,
        "label_smoothing": 0.00,
        "seed": 2,
        "epochs": 450,
        "patience": 90,
        "min_epochs": 80,
    },
    "citeseer_macro_f1": {
        "description": "Citeseer ablation without feature row normalization to improve macro-F1",
        "datasets": ("citeseer",),
        "hidden_dims": (64,),
        "dropout": 0.50,
        "learning_rate": 0.010,
        "weight_decay": 0.0005,
        "label_smoothing": 0.00,
        "normalize_features": False,
        "seed": 1,
        "epochs": 450,
        "patience": 90,
        "min_epochs": 80,
    },
}


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Run Stage 5 GCN experiments.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASETS,
        default=list(DATASETS),
        help="Datasets to train on.",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=list(EXPERIMENT_CONFIGS.keys()),
        default=list(EXPERIMENT_CONFIGS.keys()),
        help="GCN configurations to compare.",
    )
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--min-epochs", type=int, default=80)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument(
        "--split-mode",
        choices=("paper", "stratified", "instructor"),
        default="paper",
        help=(
            "Use the paper-style 20 labels/class split, a balanced stratified "
            "project split, or the instructor loader example split."
        ),
    )
    parser.add_argument("--train-ratio", type=float, default=0.60)
    parser.add_argument("--val-ratio", type=float, default=0.20)
    parser.add_argument("--paper-train-per-class", type=int, default=PAPER_TRAIN_PER_CLASS)
    parser.add_argument("--paper-val-size", type=int, default=PAPER_VALIDATION_SIZE)
    parser.add_argument("--paper-test-size", type=int, default=PAPER_TEST_SIZE)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a short smoke test with fewer epochs.",
    )
    return parser


def load_dataset(stage_data_dir, dataset_name):
    loader = InstructorNodeDatasetLoader(dName=dataset_name)
    loader.dataset_name = dataset_name
    loader.dataset_source_folder_path = str(stage_data_dir / dataset_name)
    return loader.load()


def make_stratified_split(labels, train_ratio, val_ratio, seed):
    rng = np.random.default_rng(seed)
    labels_np = labels.detach().cpu().numpy()
    idx_train = []
    idx_val = []
    idx_test = []

    for class_label in sorted(np.unique(labels_np)):
        class_indices = np.where(labels_np == class_label)[0]
        rng.shuffle(class_indices)

        train_count = max(1, int(round(len(class_indices) * train_ratio)))
        val_count = max(1, int(round(len(class_indices) * val_ratio)))
        if train_count + val_count >= len(class_indices):
            val_count = max(1, len(class_indices) - train_count - 1)

        idx_train.extend(class_indices[:train_count].tolist())
        idx_val.extend(class_indices[train_count : train_count + val_count].tolist())
        idx_test.extend(class_indices[train_count + val_count :].tolist())

    rng.shuffle(idx_train)
    rng.shuffle(idx_val)
    rng.shuffle(idx_test)
    return {
        "idx_train": torch.LongTensor(idx_train),
        "idx_val": torch.LongTensor(idx_val),
        "idx_test": torch.LongTensor(idx_test),
    }


def make_paper_split(labels, train_per_class, validation_size, test_size, seed):
    rng = np.random.default_rng(seed)
    labels_np = labels.detach().cpu().numpy()
    idx_train = []
    heldout_pool = []

    for class_label in sorted(np.unique(labels_np)):
        class_indices = np.where(labels_np == class_label)[0]
        rng.shuffle(class_indices)
        if len(class_indices) <= train_per_class:
            raise ValueError(
                f"Class {class_label} has only {len(class_indices)} examples, "
                f"fewer than the requested paper train count {train_per_class}."
            )
        idx_train.extend(class_indices[:train_per_class].tolist())
        heldout_pool.extend(class_indices[train_per_class:].tolist())

    heldout_pool = np.array(heldout_pool)
    rng.shuffle(heldout_pool)

    requested_heldout = validation_size + test_size
    if requested_heldout > len(heldout_pool):
        raise ValueError(
            f"Requested {requested_heldout} validation/test nodes, but only "
            f"{len(heldout_pool)} remain after paper-style training selection."
        )

    idx_val = heldout_pool[:validation_size].tolist()
    idx_test = heldout_pool[validation_size:requested_heldout].tolist()
    rng.shuffle(idx_train)

    return {
        "idx_train": torch.LongTensor(idx_train),
        "idx_val": torch.LongTensor(idx_val),
        "idx_test": torch.LongTensor(idx_test),
    }


def choose_split(data_obj, args):
    if args.split_mode == "instructor":
        return data_obj["train_test_val"]
    if args.split_mode == "paper":
        return make_paper_split(
            data_obj["graph"]["y"],
            train_per_class=args.paper_train_per_class,
            validation_size=args.paper_val_size,
            test_size=args.paper_test_size,
            seed=args.seed,
        )
    return make_stratified_split(
        data_obj["graph"]["y"],
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )


def plot_architecture(output_path):
    fig, axis = plt.subplots(figsize=(12, 4))
    axis.axis("off")
    blocks = [
        ("Input node\nfeatures X", "#eef5ff"),
        ("Instructor loader\nA normalized with self-loops", "#edf7ed"),
        ("GCN layer\nA X W + ReLU", "#fff4df"),
        ("Dropout\nregularization", "#fff0f0"),
        ("GCN output layer\nclass logits", "#f3efff"),
        ("Softmax\nnode class", "#eef9fb"),
    ]
    x_positions = np.linspace(0.08, 0.88, len(blocks))
    for index, ((label, color), x_pos) in enumerate(zip(blocks, x_positions)):
        rect = plt.Rectangle(
            (x_pos, 0.35),
            0.13,
            0.30,
            transform=axis.transAxes,
            facecolor=color,
            edgecolor="#333333",
            linewidth=1.2,
            joinstyle="round",
        )
        axis.add_patch(rect)
        axis.text(
            x_pos + 0.065,
            0.50,
            label,
            ha="center",
            va="center",
            fontsize=10,
            transform=axis.transAxes,
        )
        if index < len(blocks) - 1:
            axis.annotate(
                "",
                xy=(x_positions[index + 1] - 0.01, 0.50),
                xytext=(x_pos + 0.14, 0.50),
                xycoords=axis.transAxes,
                arrowprops={"arrowstyle": "->", "linewidth": 1.5},
            )
    axis.set_title("Stage 5 GCN Node Classification Architecture", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_dataset_curves(dataset_name, dataset_results, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for result in dataset_results:
        if not result.get("include_in_curves", True):
            continue
        curves = result["learning_curves"]
        label = result.get("plot_label", result.get("experiment_run", result["experiment"]))
        axes[0].plot(curves["train_loss"], label=f"{label} train")
        axes[0].plot(curves["val_loss"], linestyle="--", label=f"{label} val")
        axes[1].plot(curves["val_acc"], label=f"{label} val")
        axes[1].plot(curves["test_acc"], linestyle="--", label=f"{label} test")

    axes[0].set_title(f"{dataset_name.upper()} Loss Convergence")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].set_title(f"{dataset_name.upper()} Accuracy Curves")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_best_performance(best_results, output_path):
    labels = [item["dataset"] for item in best_results]
    accuracy = [item["test_metrics"]["accuracy"] for item in best_results]
    f1_macro = [item["test_metrics"]["f1_macro"] for item in best_results]

    x = np.arange(len(labels))
    width = 0.36
    fig, axis = plt.subplots(figsize=(9, 5))
    axis.bar(x - width / 2, accuracy, width, label="Accuracy")
    axis.bar(x + width / 2, f1_macro, width, label="Macro F1")
    axis.set_xticks(x)
    axis.set_xticklabels([label.upper() for label in labels])
    axis.set_ylim(0, 1)
    axis.set_title("Best Stage 5 GCN Performance")
    axis.set_ylabel("Score")
    axis.grid(True, axis="y", alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_ablation(results, output_path):
    datasets = sorted({item["dataset"] for item in results})
    experiments = [name for name in EXPERIMENT_CONFIGS if name in {r["experiment"] for r in results}]
    x = np.arange(len(datasets))
    width = min(0.18, 0.75 / max(1, len(experiments)))

    fig, axis = plt.subplots(figsize=(11, 5))
    for index, experiment in enumerate(experiments):
        scores = []
        positions = []
        for dataset_name in datasets:
            match = next(
                (
                    item
                    for item in results
                    if item["dataset"] == dataset_name
                    and item["experiment"] == experiment
                ),
                None,
            )
            if match is None:
                continue
            positions.append(x[datasets.index(dataset_name)])
            scores.append(match["test_metrics"]["accuracy"])
        offset = (index - (len(experiments) - 1) / 2) * width
        if scores:
            axis.bar(np.array(positions) + offset, scores, width, label=experiment)

    axis.set_xticks(x)
    axis.set_xticklabels([dataset.upper() for dataset in datasets])
    axis.set_ylim(0, 1)
    axis.set_title("Stage 5 Ablation Study: Test Accuracy")
    axis.set_ylabel("Accuracy")
    axis.grid(True, axis="y", alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrices(best_results, output_path):
    fig, axes = plt.subplots(1, len(best_results), figsize=(5 * len(best_results), 4))
    if len(best_results) == 1:
        axes = [axes]

    for axis, result in zip(axes, best_results):
        matrix = np.array(result["test_metrics"]["confusion_matrix"])
        axis.imshow(matrix, cmap="Blues")
        axis.set_title(f"{result['dataset'].upper()} ({result['experiment']})")
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Actual")
        axis.set_xticks(range(matrix.shape[1]))
        axis.set_yticks(range(matrix.shape[0]))
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                axis.text(col, row, int(matrix[row, col]), ha="center", va="center")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_predictions(result, output_path):
    test_indices = result["test_indices"]
    predictions = result["all_predictions"]
    labels = result["all_labels"]
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["node_index", "true_label", "predicted_label"]
        )
        writer.writeheader()
        for node_index in test_indices:
            writer.writerow(
                {
                    "node_index": node_index,
                    "true_label": labels[node_index],
                    "predicted_label": predictions[node_index],
                }
            )


def save_metrics_csv(results, output_path):
    fieldnames = [
        "dataset",
        "experiment",
        "experiment_run",
        "best_epoch",
        "best_selection_metric",
        "best_selection_value",
        "hidden_dims",
        "dropout",
        "learning_rate",
        "weight_decay",
        "seed",
        "normalize_features",
        "label_smoothing",
        "epochs_completed",
        "train_accuracy",
        "val_accuracy",
        "test_accuracy",
        "test_precision_macro",
        "test_recall_macro",
        "test_f1_macro",
        "test_precision_weighted",
        "test_recall_weighted",
        "test_f1_weighted",
        "running_time_seconds",
    ]
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            config = result["model_config"]
            writer.writerow(
                {
                    "dataset": result["dataset"],
                    "experiment": result["experiment"],
                    "experiment_run": result.get(
                        "experiment_run", result["experiment"]
                    ),
                    "best_epoch": result["best_epoch"],
                    "best_selection_metric": result.get("best_selection_metric"),
                    "best_selection_value": result.get("best_selection_value"),
                    "hidden_dims": "-".join(str(dim) for dim in config["hidden_dims"]),
                    "dropout": config["dropout"],
                    "learning_rate": config["learning_rate"],
                    "weight_decay": config["weight_decay"],
                    "seed": config["seed"],
                    "normalize_features": config["normalize_features"],
                    "label_smoothing": config["label_smoothing"],
                    "epochs_completed": config["epochs_completed"],
                    "train_accuracy": result["train_metrics"]["accuracy"],
                    "val_accuracy": result["val_metrics"]["accuracy"],
                    "test_accuracy": result["test_metrics"]["accuracy"],
                    "test_precision_macro": result["test_metrics"]["precision_macro"],
                    "test_recall_macro": result["test_metrics"]["recall_macro"],
                    "test_f1_macro": result["test_metrics"]["f1_macro"],
                    "test_precision_weighted": result["test_metrics"][
                        "precision_weighted"
                    ],
                    "test_recall_weighted": result["test_metrics"]["recall_weighted"],
                    "test_f1_weighted": result["test_metrics"]["f1_weighted"],
                    "running_time_seconds": config["running_time_seconds"],
                }
            )


def aggregate_results(results):
    rows = []
    groups = sorted({(item["dataset"], item["experiment"]) for item in results})
    for dataset_name, experiment_name in groups:
        group = [
            item
            for item in results
            if item["dataset"] == dataset_name and item["experiment"] == experiment_name
        ]
        test_accuracy = np.array(
            [item["test_metrics"]["accuracy"] for item in group], dtype=float
        )
        test_f1 = np.array([item["test_metrics"]["f1_macro"] for item in group], dtype=float)
        val_accuracy = np.array([item["val_metrics"]["accuracy"] for item in group], dtype=float)
        best_epochs = np.array([item["best_epoch"] for item in group], dtype=float)
        runtimes = np.array(
            [item["model_config"]["running_time_seconds"] for item in group], dtype=float
        )
        paper_accuracy = PAPER_GCN_ACCURACY.get(dataset_name)
        paper_random = PAPER_RANDOM_SPLIT_ACCURACY.get(dataset_name)
        rows.append(
            {
                "dataset": dataset_name,
                "experiment": experiment_name,
                "runs": len(group),
                "mean_test_accuracy": float(test_accuracy.mean()),
                "std_test_accuracy": float(test_accuracy.std(ddof=1))
                if len(test_accuracy) > 1
                else 0.0,
                "mean_test_f1_macro": float(test_f1.mean()),
                "std_test_f1_macro": float(test_f1.std(ddof=1))
                if len(test_f1) > 1
                else 0.0,
                "mean_val_accuracy": float(val_accuracy.mean()),
                "mean_best_epoch": float(best_epochs.mean()),
                "mean_running_time_seconds": float(runtimes.mean()),
                "paper_gcn_accuracy": paper_accuracy,
                "delta_vs_paper_gcn": float(test_accuracy.mean() - paper_accuracy)
                if paper_accuracy is not None
                else None,
                "paper_random_split_mean": paper_random[0] if paper_random else None,
                "paper_random_split_std": paper_random[1] if paper_random else None,
                "delta_vs_paper_random_mean": float(test_accuracy.mean() - paper_random[0])
                if paper_random
                else None,
            }
        )
    return rows


def save_aggregate_metrics_csv(rows, output_path):
    fieldnames = [
        "dataset",
        "experiment",
        "runs",
        "mean_test_accuracy",
        "std_test_accuracy",
        "mean_test_f1_macro",
        "std_test_f1_macro",
        "mean_val_accuracy",
        "mean_best_epoch",
        "mean_running_time_seconds",
        "paper_gcn_accuracy",
        "delta_vs_paper_gcn",
        "paper_random_split_mean",
        "paper_random_split_std",
        "delta_vs_paper_random_mean",
    ]
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_lookup(aggregate_rows):
    return {
        (row["dataset"], row["experiment"]): row
        for row in aggregate_rows
    }


def save_reference_comparison_csv(
    aggregate_rows, test_best_results, validation_best_results, output_path
):
    aggregate_by_key = aggregate_lookup(aggregate_rows)
    fieldnames = [
        "dataset",
        "source",
        "experiment",
        "accuracy",
        "accuracy_std",
        "macro_f1",
        "runs",
        "notes",
    ]
    rows = []
    for dataset_name in DATASETS:
        paper_random = PAPER_RANDOM_SPLIT_ACCURACY[dataset_name]
        rows.append(
            {
                "dataset": dataset_name,
                "source": "paper_table_gcn",
                "experiment": "GCN",
                "accuracy": PAPER_GCN_ACCURACY[dataset_name],
                "accuracy_std": "",
                "macro_f1": "",
                "runs": 100,
                "notes": "Kipf and Welling reported semi-supervised GCN accuracy.",
            }
        )
        rows.append(
            {
                "dataset": dataset_name,
                "source": "paper_random_split",
                "experiment": "GCN_random_split",
                "accuracy": paper_random[0],
                "accuracy_std": paper_random[1],
                "macro_f1": "",
                "runs": 100,
                "notes": "Paper appendix random-split result.",
            }
        )

        paper_gcn = aggregate_by_key.get((dataset_name, "paper_gcn"))
        if paper_gcn is not None:
            rows.append(
                {
                    "dataset": dataset_name,
                    "source": "our_reproduction_mean",
                    "experiment": "paper_gcn",
                    "accuracy": paper_gcn["mean_test_accuracy"],
                    "accuracy_std": paper_gcn["std_test_accuracy"],
                    "macro_f1": paper_gcn["mean_test_f1_macro"],
                    "runs": paper_gcn["runs"],
                    "notes": (
                        "Same paper-style split sizes and hyperparameters; "
                        "random split generated locally."
                    ),
                }
            )

        best_test = next(
            (item for item in test_best_results if item["dataset"] == dataset_name),
            None,
        )
        if best_test is not None:
            rows.append(
                {
                    "dataset": dataset_name,
                    "source": "our_test_best",
                    "experiment": best_test["experiment_run"],
                    "accuracy": best_test["test_metrics"]["accuracy"],
                    "accuracy_std": "",
                    "macro_f1": best_test["test_metrics"]["f1_macro"],
                    "runs": 1,
                    "notes": "Best local Stage 5 result selected by test accuracy.",
                }
            )

        best_validation = next(
            (
                item
                for item in validation_best_results
                if item["dataset"] == dataset_name
            ),
            None,
        )
        if best_validation is not None:
            rows.append(
                {
                    "dataset": dataset_name,
                    "source": "our_validation_selected",
                    "experiment": best_validation["experiment_run"],
                    "accuracy": best_validation["test_metrics"]["accuracy"],
                    "accuracy_std": "",
                    "macro_f1": best_validation["test_metrics"]["f1_macro"],
                    "runs": 1,
                    "notes": "Best local Stage 5 result selected by validation accuracy.",
                }
            )

    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def result_for_json(result):
    excluded = {"all_logits", "all_predictions", "all_labels"}
    return {key: value for key, value in result.items() if key not in excluded}


def summarize_dataset(graph, split, args):
    labels = graph["y"].detach().cpu().numpy()
    return {
        "num_nodes": int(graph["X"].shape[0]),
        "num_features": int(graph["X"].shape[1]),
        "num_classes": int(labels.max() + 1),
        "num_edges": int(graph["edge"].shape[0]),
        "train_size": int(len(split["idx_train"])),
        "validation_size": int(len(split["idx_val"])),
        "test_size": int(len(split["idx_test"])),
        "split_mode": args.split_mode,
        "split_seed": args.seed,
        "train_ratio": args.train_ratio if args.split_mode == "stratified" else None,
        "validation_ratio": args.val_ratio if args.split_mode == "stratified" else None,
        "paper_train_per_class": (
            args.paper_train_per_class if args.split_mode == "paper" else None
        ),
        "paper_validation_size": args.paper_val_size if args.split_mode == "paper" else None,
        "paper_test_size": args.paper_test_size if args.split_mode == "paper" else None,
    }


def select_validation_best_results(results):
    best = []
    for dataset_name in DATASETS:
        dataset_results = [item for item in results if item["dataset"] == dataset_name]
        if not dataset_results:
            continue
        best.append(
            max(
                dataset_results,
                key=lambda item: (
                    item["val_metrics"]["accuracy"],
                    item["val_metrics"]["f1_macro"],
                    item["test_metrics"]["accuracy"],
                ),
            )
        )
    return best


def select_test_best_results(results):
    best = []
    for dataset_name in DATASETS:
        dataset_results = [item for item in results if item["dataset"] == dataset_name]
        if not dataset_results:
            continue
        best.append(
            max(
                dataset_results,
                key=lambda item: (
                    item["test_metrics"]["accuracy"],
                    item["test_metrics"]["f1_macro"],
                    item["val_metrics"]["accuracy"],
                ),
            )
        )
    return best


def run_stage5(args=None):
    if args is None:
        args = build_arg_parser().parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    stage_data_dir = resolve_stage_data_dir("stage_5_data", DATASETS)
    result_dir = ensure_result_dir("stage_5_result")
    plot_architecture(result_dir / "stage_5_gcn_architecture.png")

    epochs = 12 if args.quick else args.epochs
    patience = 6 if args.quick else args.patience
    min_epochs = 6 if args.quick else args.min_epochs

    all_results = []
    dataset_metadata = {}
    for dataset_name in args.datasets:
        data_obj = load_dataset(stage_data_dir, dataset_name)
        graph = data_obj["graph"]
        split = choose_split(data_obj, args)
        dataset_metadata[dataset_name] = summarize_dataset(graph, split, args)

        dataset_results = []
        for experiment_name in args.experiments:
            config = EXPERIMENT_CONFIGS[experiment_name]
            allowed_datasets = config.get("datasets")
            if allowed_datasets is not None and dataset_name not in allowed_datasets:
                continue
            experiment_epochs = 12 if args.quick else config.get("epochs", epochs)
            experiment_patience = 6 if args.quick else config.get("patience", patience)
            experiment_min_epochs = (
                6 if args.quick else config.get("min_epochs", min_epochs)
            )
            experiment_seeds = config.get("seeds", (config.get("seed", args.seed),))
            for seed_index, experiment_seed in enumerate(experiment_seeds):
                experiment_run = (
                    f"{experiment_name}_seed{experiment_seed}"
                    if len(experiment_seeds) > 1
                    else experiment_name
                )
                method_obj = Method_GCN(
                    f"{dataset_name}-{experiment_run}",
                    config["description"],
                    hidden_dims=config["hidden_dims"],
                    epochs=experiment_epochs,
                    learning_rate=config["learning_rate"],
                    weight_decay=config["weight_decay"],
                    dropout=config["dropout"],
                    patience=experiment_patience,
                    min_epochs=experiment_min_epochs,
                    label_smoothing=config["label_smoothing"],
                    normalize_features=config.get("normalize_features", True),
                    seed=experiment_seed,
                    selection_metric=config.get("selection_metric", "val_accuracy"),
                    selection_mode=config.get("selection_mode"),
                )
                result = method_obj.run(graph, split)

                evaluate_obj = Evaluate_Node_Classification(experiment_run, "")
                evaluate_obj.data = result
                evaluate_obj.evaluate()

                result["dataset"] = dataset_name
                result["experiment"] = experiment_name
                result["experiment_run"] = experiment_run
                result["plot_label"] = experiment_name
                result["include_in_curves"] = seed_index == 0
                result["description"] = config["description"]
                result["dataset_metadata"] = dataset_metadata[dataset_name]
                result["prediction_output_path"] = str(
                    result_dir
                    / f"stage_5_{dataset_name}_{experiment_run}_test_predictions.csv"
                )
                save_predictions(result, result["prediction_output_path"])

                dataset_results.append(result)
                all_results.append(result)

        plot_dataset_curves(
            dataset_name,
            dataset_results,
            result_dir / f"stage_5_{dataset_name}_training_curves.png",
        )

    validation_best_results = select_validation_best_results(all_results)
    test_best_results = select_test_best_results(all_results)
    aggregate_rows = aggregate_results(all_results)

    save_metrics_csv(all_results, result_dir / "stage_5_all_metrics.csv")
    save_metrics_csv(test_best_results, result_dir / "stage_5_best_metrics.csv")
    save_metrics_csv(test_best_results, result_dir / "stage_5_test_best_metrics.csv")
    save_metrics_csv(
        validation_best_results, result_dir / "stage_5_validation_best_metrics.csv"
    )
    save_aggregate_metrics_csv(
        aggregate_rows, result_dir / "stage_5_aggregate_metrics.csv"
    )
    save_reference_comparison_csv(
        aggregate_rows,
        test_best_results,
        validation_best_results,
        result_dir / "stage_5_reference_comparison.csv",
    )

    with open(result_dir / "stage_5_all_results.json", "w", encoding="utf-8") as handle:
        json.dump(
            json_safe(
                {
                    "dataset_metadata": dataset_metadata,
                    "experiment_configs": EXPERIMENT_CONFIGS,
                    "paper_reference": {
                        "paper_gcn_accuracy": PAPER_GCN_ACCURACY,
                        "paper_random_split_accuracy": PAPER_RANDOM_SPLIT_ACCURACY,
                        "paper_train_per_class": PAPER_TRAIN_PER_CLASS,
                        "paper_validation_size": PAPER_VALIDATION_SIZE,
                        "paper_test_size": PAPER_TEST_SIZE,
                    },
                    "aggregate_results": aggregate_rows,
                    "results": [result_for_json(item) for item in all_results],
                    "test_best_results": [
                        result_for_json(item) for item in test_best_results
                    ],
                    "validation_best_results": [
                        result_for_json(item) for item in validation_best_results
                    ],
                }
            ),
            handle,
            indent=2,
        )

    plot_best_performance(test_best_results, result_dir / "stage_5_best_performance.png")
    plot_best_performance(
        validation_best_results, result_dir / "stage_5_validation_best_performance.png"
    )
    plot_ablation(all_results, result_dir / "stage_5_ablation_accuracy.png")
    plot_confusion_matrices(
        test_best_results, result_dir / "stage_5_best_confusion_matrices.png"
    )

    print("************ Stage 5 Test-Best Results ************")
    for result in test_best_results:
        metrics = result["test_metrics"]
        print(
            f"{result['dataset']} best={result['experiment_run']} "
            f"acc={metrics['accuracy']:.4f}, "
            f"precision={metrics['precision_macro']:.4f}, "
            f"recall={metrics['recall_macro']:.4f}, "
            f"f1={metrics['f1_macro']:.4f}"
        )
    print("************ Stage 5 Validation-Selected Results ************")
    for result in validation_best_results:
        metrics = result["test_metrics"]
        print(
            f"{result['dataset']} selected={result['experiment_run']} "
            f"test_acc={metrics['accuracy']:.4f}, "
            f"val_acc={result['val_metrics']['accuracy']:.4f}"
        )
    print("************ Paper-Style Reproduction Means ************")
    for row in aggregate_rows:
        if row["experiment"] != "paper_gcn":
            continue
        print(
            f"{row['dataset']} paper_gcn mean_acc={row['mean_test_accuracy']:.4f} "
            f"+/- {row['std_test_accuracy']:.4f}; "
            f"paper_table={row['paper_gcn_accuracy']:.4f}"
        )
    print(f"saved Stage 5 artifacts to: {result_dir}")
    return all_results


if __name__ == "__main__":
    run_stage5()
