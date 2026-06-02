import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from local_code.base_class.method import method
from local_code.stage_5_code.Method_RNN_TextClassification import (
    format_seconds,
    get_torch_device,
)


class RNNGenerationModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, model_type):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        if model_type == "RNN":
            self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        elif model_type == "LSTM":
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        elif model_type == "GRU":
            self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        else:
            raise ValueError("Invalid model type. Choose from RNN, LSTM, GRU.")

        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, text):
        embedded = self.embedding(text)
        output, hidden = self.rnn(embedded)

        if isinstance(hidden, tuple):
            hidden = hidden[0]

        hidden = hidden.squeeze(0)
        if hidden.ndim == 3:
            hidden = hidden[-1]
        return self.fc(hidden)


class Method_RNN_TextGeneration(method):
    data = None

    def __init__(
        self,
        mName=None,
        mDescription=None,
        model_type="RNN",
        hidden_dim=128,
        epochs=30,
        batch_size=128,
        lr=0.01,
        embedding_dim=64,
        seed=2,
    ):
        super().__init__(mName, mDescription)
        self.model_type = model_type
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.embedding_dim = embedding_dim
        self.seed = seed
        self.device = get_torch_device()

    def run(self, dataset, vocab_size, word2idx, idx2word, seed_words=None, num_generate=20):
        print(f"Running {self.model_type} text generation on {self.device}...")
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        self.method_start_time = time.time()
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        model = RNNGenerationModel(
            vocab_size, self.embedding_dim, self.hidden_dim, self.model_type
        ).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss().to(self.device)
        learning_curves = {"train_loss": []}

        self.method_training_time = 0
        self.method_testing_time = 0

        table_width = 58
        print("\n" + "=" * table_width)
        print(f"{self.model_type} TEXT GENERATION TRAINING".center(table_width))
        print("=" * table_width)
        print(f"{'Epoch':>9} | {'Train Loss':>12} | {'Time':>8}")
        print("-" * table_width)

        for epoch in range(self.epochs):
            epoch_start = time.time()
            model.train()
            epoch_loss = 0.0

            for texts, labels in train_loader:
                texts = texts.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                predictions = model(texts)
                loss = criterion(predictions, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / max(1, len(train_loader))
            learning_curves["train_loss"].append(avg_loss)
            self.method_training_time += time.time() - epoch_start
            print(
                f"{epoch + 1:>3}/{self.epochs:<5} | "
                f"{avg_loss:>12.4f} | {format_seconds(time.time() - epoch_start):>8}",
                flush=True,
            )

        eval_start = time.time()
        model.eval()
        if seed_words is None or len(seed_words) != 3:
            seed_words = ["what", "did", "the"]

        current_words = [word.lower() for word in seed_words]
        generated_story = current_words.copy()

        with torch.no_grad():
            for _ in range(num_generate):
                index_sequence = [word2idx.get(word, 0) for word in current_words]
                index_tensor = torch.tensor([index_sequence], dtype=torch.long).to(
                    self.device
                )
                predictions = model(index_tensor)
                predicted_index = predictions.argmax(dim=1).item()
                predicted_word = idx2word.get(predicted_index, "<UNK>")
                generated_story.append(predicted_word)
                current_words = current_words[1:] + [predicted_word]

        generated_text = " ".join(generated_story)
        print("-" * table_width)
        print(f"Generated text: {generated_text}")

        self.method_testing_time += time.time() - eval_start
        self.method_stop_time = time.time()
        self.method_running_time = self.method_stop_time - self.method_start_time
        print(f"Runtime: {format_seconds(self.method_running_time)}")
        print("=" * table_width)

        return {
            "generated_text": generated_text,
            "learning_curves": learning_curves,
            "model_config": {
                "model_type": self.model_type,
                "hidden_dim": self.hidden_dim,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "learning_rate": self.lr,
                "embedding_dim": self.embedding_dim,
                "device": str(self.device),
                "training_time_seconds": self.method_training_time,
                "testing_time_seconds": self.method_testing_time,
                "running_time_seconds": self.method_running_time,
            },
        }
