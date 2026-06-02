import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from local_code.base_class.method import method


def format_seconds(seconds):
    seconds = int(round(seconds))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def get_torch_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def row_normalize_features(features):
    rowsum = features.sum(dim=1, keepdim=True)
    rowsum = rowsum.clamp(min=1.0)
    return features / rowsum


class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(input_dim, output_dim))
        if bias:
            self.bias = nn.Parameter(torch.empty(output_dim))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, features, adjacency):
        support = torch.mm(features, self.weight)
        output = torch.sparse.mm(adjacency, support)
        if self.bias is not None:
            output = output + self.bias
        return output


class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.5):
        super().__init__()
        dims = [input_dim] + list(hidden_dims) + [output_dim]
        self.layers = nn.ModuleList(
            GraphConvolution(dims[index], dims[index + 1])
            for index in range(len(dims) - 1)
        )
        self.dropout = dropout

    def forward(self, features, adjacency):
        x = features
        for layer in self.layers[:-1]:
            x = layer(x, adjacency)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.layers[-1](x, adjacency)


class Method_GCN(method):
    data = None

    def __init__(
        self,
        mName=None,
        mDescription=None,
        hidden_dims=(64,),
        epochs=400,
        learning_rate=0.01,
        weight_decay=0.0005,
        dropout=0.5,
        patience=80,
        min_epochs=80,
        label_smoothing=0.0,
        normalize_features=True,
        seed=2,
        verbose=True,
        print_every=10,
        selection_metric="val_accuracy",
        selection_mode=None,
        selection_tolerance=0.0,
    ):
        super().__init__(mName, mDescription)
        self.hidden_dims = tuple(hidden_dims)
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.patience = patience
        self.min_epochs = min_epochs
        self.label_smoothing = label_smoothing
        self.normalize_features = normalize_features
        self.seed = seed
        self.verbose = verbose
        self.print_every = print_every
        self.selection_metric = selection_metric
        self.selection_mode = selection_mode or (
            "min" if selection_metric.endswith("loss") else "max"
        )
        self.selection_tolerance = selection_tolerance
        self.device = get_torch_device()

    def _metrics(self, logits, labels, indices, criterion=None):
        subset_logits = logits[indices]
        subset_labels = labels[indices]
        loss = None
        if criterion is not None:
            loss = criterion(subset_logits, subset_labels).item()

        predictions = subset_logits.argmax(dim=1).detach().cpu().numpy()
        truth = subset_labels.detach().cpu().numpy()
        accuracy = accuracy_score(truth, predictions)
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            truth, predictions, average="macro", zero_division=0
        )
        precision_weighted, recall_weighted, f1_weighted, _ = (
            precision_recall_fscore_support(
                truth, predictions, average="weighted", zero_division=0
            )
        )
        labels_sorted = sorted(np.unique(labels.detach().cpu().numpy()).tolist())
        matrix = confusion_matrix(truth, predictions, labels=labels_sorted).tolist()
        return {
            "loss": loss,
            "accuracy": accuracy,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro,
            "precision_weighted": precision_weighted,
            "recall_weighted": recall_weighted,
            "f1_weighted": f1_weighted,
            "confusion_matrix": matrix,
            "predictions": predictions.tolist(),
            "truth": truth.tolist(),
        }

    def _selection_value(self, train_metrics, val_metrics, test_metrics):
        values = {
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_f1_macro": val_metrics["f1_macro"],
            "test_accuracy": test_metrics["accuracy"],
            "test_f1_macro": test_metrics["f1_macro"],
        }
        if self.selection_metric not in values:
            allowed = ", ".join(sorted(values))
            raise ValueError(
                f"Unknown selection_metric={self.selection_metric!r}; "
                f"allowed metrics are: {allowed}"
            )
        return values[self.selection_metric]

    def _is_improved(self, current_value, best_value):
        if self.selection_mode == "min":
            return current_value < best_value - self.selection_tolerance
        if self.selection_mode == "max":
            return current_value > best_value + self.selection_tolerance
        raise ValueError("selection_mode must be either 'min' or 'max'")

    def run(self, graph, train_test_val):
        if self.verbose:
            print(f"Running {self.method_name} on {self.device}...")
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        self.method_start_time = time.time()

        features = graph["X"].float()
        if self.normalize_features:
            features = row_normalize_features(features)
        features = features.to(self.device)

        labels = graph["y"].long().to(self.device)
        adjacency = graph["utility"]["A"].coalesce().to(self.device)
        idx_train = train_test_val["idx_train"].long().to(self.device)
        idx_val = train_test_val["idx_val"].long().to(self.device)
        idx_test = train_test_val["idx_test"].long().to(self.device)

        input_dim = features.shape[1]
        output_dim = int(labels.max().item()) + 1
        model = GCN(input_dim, self.hidden_dims, output_dim, dropout=self.dropout).to(
            self.device
        )
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)

        learning_curves = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "val_f1_macro": [],
            "test_acc": [],
        }

        best_selection_start = (
            float("inf") if self.selection_mode == "min" else -float("inf")
        )
        best = {
            "epoch": 0,
            "selection_value": best_selection_start,
            "val_loss": float("inf"),
            "val_accuracy": -1.0,
            "val_f1_macro": -1.0,
            "state": None,
        }
        wait = 0
        self.method_training_time = 0
        self.method_testing_time = 0

        table_width = 112
        if self.verbose:
            print("\n" + "=" * table_width)
            print(f"{self.method_name} GCN TRAINING / VALIDATION".center(table_width))
            print("=" * table_width)
            print(
                f"{'Epoch':>9} | {'Train Loss':>10} | {'Train Acc':>9} | "
                f"{'Val Loss':>8} | {'Val Acc':>8} | {'Val F1':>8} | "
                f"{'Test Acc':>8} | {'Time':>8}"
            )
            print("-" * table_width)

        for epoch in range(1, self.epochs + 1):
            epoch_start = time.time()
            model.train()
            optimizer.zero_grad()
            logits = model(features, adjacency)
            train_loss = criterion(logits[idx_train], labels[idx_train])
            train_loss.backward()
            optimizer.step()
            self.method_training_time += time.time() - epoch_start

            eval_start = time.time()
            model.eval()
            with torch.no_grad():
                logits = model(features, adjacency)
                train_metrics = self._metrics(logits, labels, idx_train, criterion)
                val_metrics = self._metrics(logits, labels, idx_val, criterion)
                test_metrics = self._metrics(logits, labels, idx_test, criterion)
            self.method_testing_time += time.time() - eval_start

            learning_curves["train_loss"].append(train_metrics["loss"])
            learning_curves["val_loss"].append(val_metrics["loss"])
            learning_curves["train_acc"].append(train_metrics["accuracy"])
            learning_curves["val_acc"].append(val_metrics["accuracy"])
            learning_curves["val_f1_macro"].append(val_metrics["f1_macro"])
            learning_curves["test_acc"].append(test_metrics["accuracy"])

            selection_value = self._selection_value(
                train_metrics, val_metrics, test_metrics
            )
            improved = self._is_improved(selection_value, best["selection_value"])
            if improved:
                best = {
                    "epoch": epoch,
                    "selection_value": selection_value,
                    "val_loss": val_metrics["loss"],
                    "val_accuracy": val_metrics["accuracy"],
                    "val_f1_macro": val_metrics["f1_macro"],
                    "state": copy.deepcopy(model.state_dict()),
                }
                wait = 0
            else:
                wait += 1

            should_print = (
                epoch == 1
                or epoch == self.epochs
                or epoch % self.print_every == 0
                or improved
                or wait == self.patience
            )
            if self.verbose and should_print:
                print(
                    f"{epoch:>3}/{self.epochs:<5} | "
                    f"{train_metrics['loss']:>10.4f} | "
                    f"{train_metrics['accuracy']:>9.4f} | "
                    f"{val_metrics['loss']:>8.4f} | "
                    f"{val_metrics['accuracy']:>8.4f} | "
                    f"{val_metrics['f1_macro']:>8.4f} | "
                    f"{test_metrics['accuracy']:>8.4f} | "
                    f"{format_seconds(time.time() - epoch_start):>8}",
                    flush=True,
                )

            if epoch >= self.min_epochs and wait >= self.patience:
                if self.verbose:
                    print(
                        f"Early stopping at epoch {epoch}; "
                        f"best epoch was {best['epoch']}."
                    )
                break

        if best["state"] is not None:
            model.load_state_dict(best["state"])

        model.eval()
        with torch.no_grad():
            logits = model(features, adjacency)
            train_metrics = self._metrics(logits, labels, idx_train, criterion)
            val_metrics = self._metrics(logits, labels, idx_val, criterion)
            test_metrics = self._metrics(logits, labels, idx_test, criterion)
            all_predictions = logits.argmax(dim=1).detach().cpu().numpy().tolist()
            all_logits = logits.detach().cpu().numpy().tolist()

        self.method_stop_time = time.time()
        self.method_running_time = self.method_stop_time - self.method_start_time

        if self.verbose:
            print("-" * table_width)
            print(
                f"Best Epoch: {best['epoch']} | "
                f"{self.selection_metric}: {best['selection_value']:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.4f} | "
                f"Test Acc: {test_metrics['accuracy']:.4f} | "
                f"Test Macro F1: {test_metrics['f1_macro']:.4f} | "
                f"Runtime: {format_seconds(self.method_running_time)}"
            )
            print("=" * table_width)

        return {
            "best_epoch": best["epoch"],
            "best_selection_metric": self.selection_metric,
            "best_selection_mode": self.selection_mode,
            "best_selection_value": best["selection_value"],
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "learning_curves": learning_curves,
            "all_predictions": all_predictions,
            "all_logits": all_logits,
            "all_labels": labels.detach().cpu().numpy().tolist(),
            "test_indices": idx_test.detach().cpu().numpy().tolist(),
            "model_config": {
                "hidden_dims": list(self.hidden_dims),
                "epochs_requested": self.epochs,
                "epochs_completed": len(learning_curves["train_loss"]),
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "dropout": self.dropout,
                "patience": self.patience,
                "min_epochs": self.min_epochs,
                "label_smoothing": self.label_smoothing,
                "normalize_features": self.normalize_features,
                "selection_metric": self.selection_metric,
                "selection_mode": self.selection_mode,
                "selection_tolerance": self.selection_tolerance,
                "seed": self.seed,
                "device": str(self.device),
                "training_time_seconds": self.method_training_time,
                "testing_time_seconds": self.method_testing_time,
                "running_time_seconds": self.method_running_time,
            },
        }
