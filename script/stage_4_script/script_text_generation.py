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
from local_code.stage_4_code.Dataset_Loader import Dataset_Loader_TextGeneration
from local_code.stage_4_code.Evaluate_Text import Evaluate_TextGeneration
from local_code.stage_4_code.Method_RNN_TextGeneration import Method_RNN_TextGeneration


MODEL_CHOICES = ("RNN", "LSTM", "GRU")


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
    parser = argparse.ArgumentParser(description="Run Stage 4 text generation.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_CHOICES),
        choices=MODEL_CHOICES,
        help="Recurrent unit types to train.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--context-length", type=int, default=3)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--num-generate", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument(
        "--seed-words",
        nargs=3,
        default=["what", "did", "the"],
        help="Three starting words for generated text.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a short smoke test with fewer rows and one epoch.",
    )
    return parser


def plot_learning_curves(results, output_path):
    fig, axis = plt.subplots(figsize=(7, 5))
    for model_name, curves in results.items():
        axis.plot(curves["train_loss"], label=model_name, marker="o", markersize=3)
    axis.set_title("Text Generation Training Loss")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.legend()
    axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_generation_csv(summaries, output_path):
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model",
                "final_train_loss",
                "training_time_seconds",
                "testing_time_seconds",
                "running_time_seconds",
                "generated_text",
            ],
        )
        writer.writeheader()
        for summary in summaries:
            curves = summary["learning_curves"]
            config = summary["model_config"]
            writer.writerow(
                {
                    "model": summary["model"],
                    "final_train_loss": curves["train_loss"][-1],
                    "training_time_seconds": config["training_time_seconds"],
                    "testing_time_seconds": config["testing_time_seconds"],
                    "running_time_seconds": config["running_time_seconds"],
                    "generated_text": summary["generated_text"],
                }
            )


def save_generated_samples(summaries, output_path):
    with open(output_path, "w", encoding="utf-8") as handle:
        for summary in summaries:
            handle.write(f"{summary['model']}:\n")
            handle.write(summary["generated_text"])
            handle.write("\n\n")


def run_generation(args=None):
    if args is None:
        args = build_arg_parser().parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_root = resolve_stage_data_dir(
        "stage_4_data", ("text_classification", "text_generation")
    )
    data_dir = data_root / "text_generation"
    result_dir = ensure_result_dir("stage_4_result")

    data_file = data_dir / "data"
    if not data_file.exists():
        data_file = data_dir / "data.csv"
    if not data_file.exists():
        raise FileNotFoundError(f"Missing text generation CSV file: {data_dir / 'data'}")

    epochs = 1 if args.quick else args.epochs
    max_rows = args.max_rows
    if args.quick and max_rows is None:
        max_rows = 200

    dataset_loader = Dataset_Loader_TextGeneration(
        "Jokes Text Generation",
        "next-token generation from three starting words",
        context_length=args.context_length,
        max_rows=max_rows,
    )
    dataset_loader.dataset_source_folder_path = str(data_dir)
    dataset_loader.dataset_source_file_name = data_file.name
    data_obj = dataset_loader.load()

    curve_results = {}
    summaries = []
    for model_name in args.models:
        method_obj = Method_RNN_TextGeneration(
            model_name,
            "Stage 4 recurrent text generator",
            model_type=model_name,
            hidden_dim=args.hidden_dim,
            epochs=epochs,
            batch_size=args.batch_size,
            lr=args.learning_rate,
            embedding_dim=args.embedding_dim,
            seed=args.seed,
        )
        result = method_obj.run(
            data_obj["dataset"],
            data_obj["vocab_size"],
            data_obj["word2idx"],
            data_obj["idx2word"],
            seed_words=args.seed_words,
            num_generate=args.num_generate,
        )

        evaluate_obj = Evaluate_TextGeneration(model_name, "")
        evaluate_obj.data = result
        generated_text = evaluate_obj.evaluate()
        curve_results[model_name] = result["learning_curves"]
        summaries.append(
            {
                "model": model_name,
                "generated_text": generated_text,
                "learning_curves": result["learning_curves"],
                "model_config": result["model_config"],
                "data_metadata": data_obj["metadata"],
            }
        )

    curve_path = result_dir / "stage_4_text_generation_curves.png"
    plot_learning_curves(curve_results, curve_path)

    summary_path = result_dir / "stage_4_text_generation_results.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(json_safe(summaries), handle, indent=2)

    metrics_path = result_dir / "stage_4_text_generation_metrics.csv"
    save_generation_csv(summaries, metrics_path)

    samples_path = result_dir / "stage_4_text_generation_samples.txt"
    save_generated_samples(summaries, samples_path)

    print("************ Stage 4 Text Generation Summary ************")
    for summary in summaries:
        print(f"{summary['model']}: {summary['generated_text']}")
    print(f"saved learning curves to: {curve_path}")
    print(f"saved generated text to: {summary_path}")
    print(f"saved metrics table to: {metrics_path}")
    print(f"saved generated samples to: {samples_path}")
    return summaries


if __name__ == "__main__":
    run_generation()
