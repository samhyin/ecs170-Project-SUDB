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
from local_code.stage_4_code.Dataset_Loader import Dataset_Loader_TextClassification
from local_code.stage_4_code.Evaluate_Text import Evaluate_TextClassification
from local_code.stage_4_code.Method_RNN_TextClassification import (
    Method_RNN_TextClassification,
)


MODEL_CHOICES = ("RNN", "LSTM", "GRU")
DEFAULT_GLOVE_FILE = (
    "wiki_giga_2024_50_MFT20_vectors_seed_123_alpha_0.75_eta_0.075_combined.txt"
)


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


def resolve_optional_path(path_text):
    if not path_text:
        default_path = PROJECT_ROOT / DEFAULT_GLOVE_FILE
        return default_path if default_path.exists() else None

    path = Path(path_text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path if path.exists() else None


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Run Stage 4 text classification.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_CHOICES),
        choices=MODEL_CHOICES,
        help="Recurrent unit types to train.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--max-seq-length", type=int, default=128)
    parser.add_argument("--max-vocab-size", type=int, default=25000)
    parser.add_argument("--embedding-dim", type=int, default=50)
    parser.add_argument("--max-files-per-class", type=int, default=None)
    parser.add_argument("--glove-path", default=None)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument(
        "--no-bidirectional",
        action="store_true",
        help="Use a one-direction recurrent classifier.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a short smoke test with fewer files and one epoch.",
    )
    return parser


def plot_learning_curves(results, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    curve_specs = (
        ("train_loss", "Training Loss", "Loss"),
        ("train_acc", "Training Accuracy", "Accuracy"),
        ("test_acc", "Testing Accuracy", "Accuracy"),
    )

    for axis, (curve_key, title, ylabel) in zip(axes, curve_specs):
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(ylabel)
        for model_name, curves in results.items():
            axis.plot(curves[curve_key], label=model_name, marker="o", markersize=3)
        axis.legend()
        axis.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrices(results, output_path):
    if not results:
        return

    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4))
    if len(results) == 1:
        axes = [axes]

    for axis, (model_name, matrix) in zip(axes, results.items()):
        axis.imshow(matrix, cmap="Blues")
        axis.set_title(f"{model_name} Confusion Matrix")
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Actual")
        axis.set_xticks([0, 1])
        axis.set_yticks([0, 1])
        axis.set_xticklabels(["neg", "pos"])
        axis.set_yticklabels(["neg", "pos"])

        for row in range(2):
            for col in range(2):
                axis.text(
                    col,
                    row,
                    str(matrix[row][col]),
                    ha="center",
                    va="center",
                    color="black",
                )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_metrics_csv(summaries, output_path):
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "training_time_seconds",
                "testing_time_seconds",
                "running_time_seconds",
            ],
        )
        writer.writeheader()
        for summary in summaries:
            metrics = summary["metrics"]
            config = summary["model_config"]
            writer.writerow(
                {
                    "model": summary["model"],
                    "accuracy": metrics["accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "training_time_seconds": config["training_time_seconds"],
                    "testing_time_seconds": config["testing_time_seconds"],
                    "running_time_seconds": config["running_time_seconds"],
                }
            )


def run_classification(args=None):
    if args is None:
        args = build_arg_parser().parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_root = resolve_stage_data_dir(
        "stage_4_data", ("text_classification", "text_generation")
    )
    data_dir = data_root / "text_classification"
    result_dir = ensure_result_dir("stage_4_result")

    epochs = 1 if args.quick else args.epochs
    max_files_per_class = args.max_files_per_class
    if args.quick and max_files_per_class is None:
        max_files_per_class = 100

    glove_path = resolve_optional_path(args.glove_path)
    if glove_path is None:
        print("No usable GloVe file found; classification will train embeddings.")
    else:
        print(f"Using GloVe vectors: {glove_path}")

    dataset_loader = Dataset_Loader_TextClassification(
        "IMDb Text Classification",
        "binary positive/negative review classification",
        max_seq_length=args.max_seq_length,
        max_vocab_size=args.max_vocab_size,
        embedding_dim=args.embedding_dim,
        max_files_per_class=max_files_per_class,
    )
    dataset_loader.dataset_source_folder_path = str(data_dir)
    data_obj = dataset_loader.load(str(glove_path) if glove_path else None)

    curve_results = {}
    confusion_matrices = {}
    summaries = []
    for model_name in args.models:
        method_obj = Method_RNN_TextClassification(
            model_name,
            "Stage 4 recurrent text classifier",
            model_type=model_name,
            hidden_dim=args.hidden_dim,
            epochs=epochs,
            batch_size=args.batch_size,
            lr=args.learning_rate,
            dropout=args.dropout,
            bidirectional=not args.no_bidirectional,
            num_layers=args.num_layers,
            grad_clip=args.grad_clip,
            seed=args.seed,
        )
        result = method_obj.run(
            data_obj["train"],
            data_obj["test"],
            data_obj["vocab_size"],
            data_obj["embeddings"],
        )

        evaluate_obj = Evaluate_TextClassification(model_name, "")
        evaluate_obj.data = result
        metrics = evaluate_obj.evaluate()
        curve_results[model_name] = result["learning_curves"]
        confusion_matrices[model_name] = result["test_confusion_matrix"]
        summaries.append(
            {
                "model": model_name,
                "metrics": metrics,
                "confusion_matrix": result["test_confusion_matrix"],
                "learning_curves": result["learning_curves"],
                "model_config": result["model_config"],
                "data_metadata": data_obj["metadata"],
            }
        )

    curve_path = result_dir / "stage_4_text_classification_curves.png"
    plot_learning_curves(curve_results, curve_path)

    confusion_path = result_dir / "stage_4_text_classification_confusion_matrices.png"
    plot_confusion_matrices(confusion_matrices, confusion_path)

    summary_path = result_dir / "stage_4_text_classification_results.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(json_safe(summaries), handle, indent=2)

    metrics_path = result_dir / "stage_4_text_classification_metrics.csv"
    save_metrics_csv(summaries, metrics_path)

    print("************ Stage 4 Text Classification Summary ************")
    for summary in summaries:
        metrics = summary["metrics"]
        print(
            f"{summary['model']}: acc={metrics['accuracy']:.4f}, "
            f"precision={metrics['precision']:.4f}, "
            f"recall={metrics['recall']:.4f}, f1={metrics['f1']:.4f}"
        )
    print(f"saved learning curves to: {curve_path}")
    print(f"saved confusion matrices to: {confusion_path}")
    print(f"saved metrics to: {summary_path}")
    print(f"saved metrics table to: {metrics_path}")
    return summaries


if __name__ == "__main__":
    run_classification()
