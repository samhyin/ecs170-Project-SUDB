import argparse
import sys
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from script_text_classification import MODEL_CHOICES, run_classification
from script_text_generation import run_generation


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Stage 4 RNN text classification and text generation."
    )
    parser.add_argument(
        "--task",
        choices=("classification", "generation", "all"),
        default="all",
        help="Stage 4 task to run.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_CHOICES),
        choices=MODEL_CHOICES,
        help="Recurrent unit types to train.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run short smoke-test settings for the selected task.",
    )
    parser.add_argument("--seed", type=int, default=2)

    parser.add_argument("--classification-epochs", type=int, default=10)
    parser.add_argument("--classification-batch-size", type=int, default=64)
    parser.add_argument("--classification-hidden-dim", type=int, default=128)
    parser.add_argument("--classification-learning-rate", type=float, default=0.001)
    parser.add_argument("--classification-dropout", type=float, default=0.5)
    parser.add_argument("--classification-num-layers", type=int, default=1)
    parser.add_argument("--classification-grad-clip", type=float, default=5.0)
    parser.add_argument("--classification-max-seq-length", type=int, default=128)
    parser.add_argument("--classification-max-vocab-size", type=int, default=25000)
    parser.add_argument("--classification-embedding-dim", type=int, default=50)
    parser.add_argument("--classification-max-files-per-class", type=int, default=None)
    parser.add_argument("--glove-path", default=None)
    parser.add_argument(
        "--classification-no-bidirectional",
        action="store_true",
        help="Use a one-direction recurrent classifier.",
    )

    parser.add_argument("--generation-epochs", type=int, default=30)
    parser.add_argument("--generation-batch-size", type=int, default=128)
    parser.add_argument("--generation-hidden-dim", type=int, default=128)
    parser.add_argument("--generation-learning-rate", type=float, default=0.01)
    parser.add_argument("--generation-embedding-dim", type=int, default=64)
    parser.add_argument("--generation-context-length", type=int, default=3)
    parser.add_argument("--generation-max-rows", type=int, default=None)
    parser.add_argument("--generation-num-generate", type=int, default=20)
    parser.add_argument(
        "--generation-seed-words",
        nargs=3,
        default=["what", "did", "the"],
        help="Three starting words for generated text.",
    )
    return parser.parse_args()


def make_classification_args(args):
    return SimpleNamespace(
        models=args.models,
        epochs=args.classification_epochs,
        batch_size=args.classification_batch_size,
        hidden_dim=args.classification_hidden_dim,
        learning_rate=args.classification_learning_rate,
        dropout=args.classification_dropout,
        num_layers=args.classification_num_layers,
        grad_clip=args.classification_grad_clip,
        max_seq_length=args.classification_max_seq_length,
        max_vocab_size=args.classification_max_vocab_size,
        embedding_dim=args.classification_embedding_dim,
        max_files_per_class=args.classification_max_files_per_class,
        glove_path=args.glove_path,
        seed=args.seed,
        no_bidirectional=args.classification_no_bidirectional,
        quick=args.quick,
    )


def make_generation_args(args):
    return SimpleNamespace(
        models=args.models,
        epochs=args.generation_epochs,
        batch_size=args.generation_batch_size,
        hidden_dim=args.generation_hidden_dim,
        learning_rate=args.generation_learning_rate,
        embedding_dim=args.generation_embedding_dim,
        context_length=args.generation_context_length,
        max_rows=args.generation_max_rows,
        num_generate=args.generation_num_generate,
        seed=args.seed,
        seed_words=args.generation_seed_words,
        quick=args.quick,
    )


def main():
    args = parse_args()

    if args.task in ("classification", "all"):
        run_classification(make_classification_args(args))

    if args.task in ("generation", "all"):
        run_generation(make_generation_args(args))


if __name__ == "__main__":
    main()
