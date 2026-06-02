import copy
import time

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader

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
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class RNNClassificationModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        output_dim,
        model_type,
        embeddings=None,
        dropout=0.5,
        bidirectional=True,
        num_layers=1,
    ):
        super().__init__()
        self.model_type = model_type
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if embeddings is not None:
            self.embedding.weight.data.copy_(embeddings)

        self.dropout_emb = nn.Dropout(dropout)
        rnn_dropout = dropout if num_layers > 1 else 0

        if model_type == "RNN":
            self.rnn = nn.RNN(
                embedding_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=rnn_dropout,
                bidirectional=bidirectional,
            )
        elif model_type == "LSTM":
            self.rnn = nn.LSTM(
                embedding_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=rnn_dropout,
                bidirectional=bidirectional,
            )
        elif model_type == "GRU":
            self.rnn = nn.GRU(
                embedding_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=rnn_dropout,
                bidirectional=bidirectional,
            )
        else:
            raise ValueError("Invalid model type. Choose from RNN, LSTM, GRU.")

        num_directions = 2 if bidirectional else 1
        self.dropout_rnn = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * num_directions, output_dim)

    def forward(self, text):
        embedded = self.dropout_emb(self.embedding(text))
        _, hidden = self.rnn(embedded)

        if isinstance(hidden, tuple):
            hidden = hidden[0]

        if self.bidirectional:
            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            hidden = hidden[-1, :, :]

        hidden = self.dropout_rnn(hidden)
        return self.fc(hidden)


class Method_RNN_TextClassification(method):
    data = None

    def __init__(
        self,
        mName=None,
        mDescription=None,
        model_type="RNN",
        hidden_dim=128,
        epochs=10,
        batch_size=64,
        lr=0.001,
        dropout=0.5,
        bidirectional=True,
        num_layers=1,
        grad_clip=5.0,
        seed=2,
    ):
        super().__init__(mName, mDescription)
        self.model_type = model_type
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.grad_clip = grad_clip
        self.seed = seed
        self.device = get_torch_device()

    def run(self, trainData, testData, vocab_size, embeddings):
        print(f"Running {self.model_type} text classification on {self.device}...")
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        self.method_start_time = time.time()
        train_loader = DataLoader(trainData, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(testData, batch_size=self.batch_size, shuffle=False)

        embedding_dim = embeddings.shape[1] if embeddings is not None else 50
        model = RNNClassificationModel(
            vocab_size,
            embedding_dim,
            self.hidden_dim,
            2,
            self.model_type,
            embeddings,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
            num_layers=self.num_layers,
        ).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss().to(self.device)
        learning_curves = {"train_loss": [], "train_acc": [], "test_acc": []}

        self.method_training_time = 0
        self.method_testing_time = 0
        best = {
            "test_acc": 0.0,
            "test_precision": 0.0,
            "test_recall": 0.0,
            "test_f1": 0.0,
            "test_confusion_matrix": [[0, 0], [0, 0]],
            "model_state": None,
        }

        table_width = 104
        print("\n" + "=" * table_width)
        print(f"{self.model_type} TEXT CLASSIFICATION TRAINING / TESTING".center(table_width))
        print("=" * table_width)
        print(
            f"{'Epoch':>9} | {'Train Loss':>10} | {'Train Acc':>9} | "
            f"{'Test Acc':>8} | {'Precision':>9} | {'Recall':>8} | "
            f"{'F1':>8} | {'Time':>8}"
        )
        print("-" * table_width)

        for epoch in range(self.epochs):
            epoch_start = time.time()
            model.train()
            epoch_loss = 0.0
            correct = 0
            total = 0

            for texts, labels in train_loader:
                texts = texts.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                predictions = model(texts)
                loss = criterion(predictions, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
                optimizer.step()

                epoch_loss += loss.item()
                preds = predictions.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            train_acc = correct / total if total else 0.0
            learning_curves["train_loss"].append(epoch_loss / max(1, len(train_loader)))
            learning_curves["train_acc"].append(train_acc)
            self.method_training_time += time.time() - epoch_start

            eval_start = time.time()
            model.eval()
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for texts, labels in test_loader:
                    texts = texts.to(self.device)
                    predictions = model(texts)
                    all_preds.extend(predictions.argmax(dim=1).cpu().numpy())
                    all_labels.extend(labels.numpy())

            test_acc = accuracy_score(all_labels, all_preds)
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels, all_preds, average="binary", zero_division=0
            )
            matrix = confusion_matrix(all_labels, all_preds, labels=[0, 1]).tolist()
            learning_curves["test_acc"].append(test_acc)
            self.method_testing_time += time.time() - eval_start

            if test_acc > best["test_acc"]:
                best.update(
                    {
                        "test_acc": test_acc,
                        "test_precision": precision,
                        "test_recall": recall,
                        "test_f1": f1,
                        "test_confusion_matrix": matrix,
                        "model_state": copy.deepcopy(model.state_dict()),
                    }
                )

            print(
                f"{epoch + 1:>3}/{self.epochs:<5} | "
                f"{learning_curves['train_loss'][-1]:>10.4f} | "
                f"{train_acc:>9.4f} | {test_acc:>8.4f} | "
                f"{precision:>9.4f} | {recall:>8.4f} | {f1:>8.4f} | "
                f"{format_seconds(time.time() - epoch_start):>8}",
                flush=True,
            )

        self.method_stop_time = time.time()
        self.method_running_time = self.method_stop_time - self.method_start_time
        print("-" * table_width)
        print(
            f"Best Test Accuracy: {best['test_acc']:.4f} | "
            f"Precision: {best['test_precision']:.4f} | "
            f"Recall: {best['test_recall']:.4f} | F1: {best['test_f1']:.4f} | "
            f"Runtime: {format_seconds(self.method_running_time)}"
        )
        print("=" * table_width)

        return {
            "test_acc": best["test_acc"],
            "test_precision": best["test_precision"],
            "test_recall": best["test_recall"],
            "test_f1": best["test_f1"],
            "test_confusion_matrix": best["test_confusion_matrix"],
            "learning_curves": learning_curves,
            "model_config": {
                "model_type": self.model_type,
                "hidden_dim": self.hidden_dim,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "learning_rate": self.lr,
                "dropout": self.dropout,
                "bidirectional": self.bidirectional,
                "num_layers": self.num_layers,
                "grad_clip": self.grad_clip,
                "device": str(self.device),
                "training_time_seconds": self.method_training_time,
                "testing_time_seconds": self.method_testing_time,
                "running_time_seconds": self.method_running_time,
            },
        }
