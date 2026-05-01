"""
Stage 3 configurable CNN method.
"""

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from local_code.base_class.method import method


class ImageListDataset(Dataset):
    def __init__(self, instances, image_mode):
        self.instances = instances
        self.image_mode = image_mode

    def __len__(self):
        return len(self.instances)

    def _prepare_image(self, image):
        image = np.asarray(image, dtype=np.float32)
        if image.max() > 1.0:
            image = image / 255.0

        if self.image_mode == "grayscale":
            if image.ndim == 3:
                image = image[:, :, 0]
            image = image[None, :, :]
        elif image.ndim == 2:
            image = image[None, :, :]
        else:
            image = np.transpose(image, (2, 0, 1))

        return torch.from_numpy(np.ascontiguousarray(image))

    def __getitem__(self, index):
        instance = self.instances[index]
        image = self._prepare_image(instance["image"])
        label = torch.tensor(instance["encoded_label"], dtype=torch.long)
        return image, label


class ConvNet(nn.Module):
    def __init__(
        self,
        input_shape,
        num_classes,
        conv_channels,
        hidden_dim,
        dropout,
        kernel_size,
    ):
        super().__init__()
        in_channels = input_shape[0]
        padding = kernel_size // 2
        layers = []

        for out_channels in conv_channels:
            layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                ]
            )
            in_channels = out_channels

        self.features = nn.Sequential(*layers)
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            flattened_dim = self.features(dummy).view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(flattened_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class Method_CNN(method):
    data = None

    def __init__(
        self,
        mName,
        mDescription,
        conv_channels=(32, 64, 128),
        hidden_dim=128,
        dropout=0.25,
        kernel_size=3,
        max_epoch=5,
        learning_rate=0.001,
        batch_size=128,
        weight_decay=0.0001,
        num_workers=0,
        seed=2,
    ):
        super().__init__(mName, mDescription)
        self.conv_channels = tuple(conv_channels)
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.kernel_size = kernel_size
        self.max_epoch = max_epoch
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.num_workers = num_workers
        self.seed = seed
        self.loss_history = []
        self.train_accuracy_history = []
        self.test_accuracy_history = []
        self.curve_output_path = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"current device: {self.device}")

    def _make_loader(self, split, image_mode, shuffle):
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        dataset = ImageListDataset(split, image_mode)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            generator=generator if shuffle else None,
        )

    def _accuracy(self, data_loader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x_batch, y_batch in data_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                pred = self.model(x_batch).argmax(dim=1)
                correct += (pred == y_batch).sum().item()
                total += y_batch.numel()
        return correct / total if total else 0.0

    def train(self, train_loader, test_loader, metadata):
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.benchmark = True

        self.model = ConvNet(
            metadata["input_shape"],
            metadata["num_classes"],
            self.conv_channels,
            self.hidden_dim,
            self.dropout,
            self.kernel_size,
        ).to(self.device)

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        loss_function = nn.CrossEntropyLoss()

        self.loss_history = []
        self.train_accuracy_history = []
        self.test_accuracy_history = []

        for epoch in range(1, self.max_epoch + 1):
            self.model.train()
            running_loss = 0.0
            total = 0

            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                logits = self.model(x_batch)
                train_loss = loss_function(logits, y_batch)
                train_loss.backward()
                optimizer.step()

                batch_size = y_batch.numel()
                running_loss += train_loss.item() * batch_size
                total += batch_size

            avg_loss = running_loss / total
            train_acc = self._accuracy(train_loader)
            test_acc = self._accuracy(test_loader)
            self.loss_history.append(avg_loss)
            self.train_accuracy_history.append(train_acc)
            self.test_accuracy_history.append(test_acc)

            print(
                f"Epoch {epoch:03d}/{self.max_epoch:03d} "
                f"loss={avg_loss:.4f} train_acc={train_acc:.4f} test_acc={test_acc:.4f}"
            )

        self.plot_loss(metadata["dataset_name"])

    def plot_loss(self, dataset_name):
        print("generating convergence curve...")
        plt.figure(figsize=(8, 4.5))
        plt.plot(range(1, len(self.loss_history) + 1), self.loss_history, marker="o")
        plt.title(f"{dataset_name} CNN Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if self.curve_output_path:
            output_path = Path(self.curve_output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=160)
            print(f"saved convergence curve to: {output_path}")
        plt.close()

    def test(self, test_loader, class_labels):
        self.model.eval()
        pred_encoded = []
        true_encoded = []
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.to(self.device)
                logits = self.model(x_batch)
                pred_encoded.extend(logits.argmax(dim=1).cpu().numpy().tolist())
                true_encoded.extend(y_batch.numpy().tolist())

        pred_y = [class_labels[index] for index in pred_encoded]
        true_y = [class_labels[index] for index in true_encoded]
        return pred_y, true_y

    def run(self):
        print("method running...")
        metadata = self.data["metadata"]
        train_loader = self._make_loader(
            self.data["train"], metadata["image_mode"], shuffle=True
        )
        test_loader = self._make_loader(
            self.data["test"], metadata["image_mode"], shuffle=False
        )

        print("--start training...")
        start_time = time.time()
        self.train(train_loader, test_loader, metadata)
        self.method_training_time = time.time() - start_time

        print("--start testing...")
        pred_y, true_y = self.test(test_loader, metadata["class_labels"])

        return {
            "pred_y": pred_y,
            "true_y": true_y,
            "metadata": metadata,
            "history": {
                "loss": self.loss_history,
                "train_accuracy": self.train_accuracy_history,
                "test_accuracy": self.test_accuracy_history,
            },
            "model_config": {
                "conv_channels": self.conv_channels,
                "hidden_dim": self.hidden_dim,
                "dropout": self.dropout,
                "kernel_size": self.kernel_size,
                "max_epoch": self.max_epoch,
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "weight_decay": self.weight_decay,
                "device": str(self.device),
                "training_time_seconds": self.method_training_time,
            },
        }
