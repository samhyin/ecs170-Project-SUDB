import csv
import os
import re
from collections import Counter

import torch
from torch.utils.data import Dataset

from local_code.base_class.dataset import dataset


class TextClassificationDataset(Dataset):
    def __init__(self, data_x, data_y):
        self.X = torch.tensor(data_x, dtype=torch.long)
        self.y = torch.tensor(data_y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


class Dataset_Loader_TextClassification(dataset):
    data = None
    dataset_source_folder_path = None

    def __init__(
        self,
        dName=None,
        dDescription=None,
        max_seq_length=128,
        max_vocab_size=25000,
        embedding_dim=50,
        max_files_per_class=None,
    ):
        super().__init__(dName, dDescription)
        self.max_seq_length = max_seq_length
        self.max_vocab_size = max_vocab_size
        self.embedding_dim = embedding_dim
        self.max_files_per_class = max_files_per_class
        self.word2idx = {}
        self.idx2word = {}
        self.embeddings = None

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r"<br\s*/?>", " ", text)
        text = re.sub(r"[^a-z0-9\s]", "", text)
        return text.split()

    def load_glove(self, glove_path=None):
        if not glove_path or not os.path.exists(glove_path):
            print("GloVe file not found; using random trainable embeddings.")
            return {}

        print(f"Loading GloVe vectors from {glove_path}...")
        glove_dict = {}
        with open(glove_path, "r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.split()
                if len(parts) < self.embedding_dim + 1:
                    continue
                word = " ".join(parts[:-self.embedding_dim])
                vector = [float(item) for item in parts[-self.embedding_dim:]]
                glove_dict[word] = vector
        return glove_dict

    def build_vocab_and_embeddings(self, train_texts, glove_path=None):
        glove_dict = self.load_glove(glove_path)

        word_counts = Counter()
        for text in train_texts:
            word_counts.update(text)

        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}

        sorted_words = sorted(
            word_counts.keys(),
            key=lambda word: (word in glove_dict, word_counts[word]),
            reverse=True,
        )
        for word in sorted_words[: max(0, self.max_vocab_size - 2)]:
            index = len(self.word2idx)
            self.word2idx[word] = index
            self.idx2word[index] = word

        vocab_size = len(self.word2idx)
        print(f"Vocabulary size: {vocab_size}")

        self.embeddings = torch.zeros(vocab_size, self.embedding_dim)
        self.embeddings[1] = torch.randn(self.embedding_dim) * 0.1
        glove_hits = 0
        for word, index in self.word2idx.items():
            if word in glove_dict:
                self.embeddings[index] = torch.tensor(glove_dict[word])
                glove_hits += 1
            elif index > 1:
                self.embeddings[index] = torch.randn(self.embedding_dim) * 0.1

        if glove_dict:
            coverage = 100 * glove_hits / vocab_size
            print(f"GloVe coverage: {glove_hits}/{vocab_size} ({coverage:.1f}%)")

    def process_texts(self, texts):
        sequences = []
        for text in texts:
            sequence = [self.word2idx.get(word, self.word2idx["<UNK>"]) for word in text]
            if len(sequence) > self.max_seq_length:
                sequence = sequence[: self.max_seq_length]
            else:
                sequence += [self.word2idx["<PAD>"]] * (
                    self.max_seq_length - len(sequence)
                )
            sequences.append(sequence)
        return sequences

    def load_folder(self, path):
        texts = []
        labels = []
        for label_dir, label_value in (("pos", 1), ("neg", 0)):
            dir_path = os.path.join(path, label_dir)
            if not os.path.exists(dir_path):
                continue

            filenames = sorted(
                filename for filename in os.listdir(dir_path) if filename.endswith(".txt")
            )
            if self.max_files_per_class is not None:
                filenames = filenames[: self.max_files_per_class]

            for filename in filenames:
                with open(
                    os.path.join(dir_path, filename), "r", encoding="utf-8"
                ) as handle:
                    texts.append(self.clean_text(handle.read()))
                    labels.append(label_value)
        return texts, labels

    def load(self, glove_path=None):
        print("Loading text classification data...")

        train_path = os.path.join(self.dataset_source_folder_path, "train")
        test_path = os.path.join(self.dataset_source_folder_path, "test")

        train_texts, train_labels = self.load_folder(train_path)
        test_texts, test_labels = self.load_folder(test_path)
        self.build_vocab_and_embeddings(train_texts, glove_path)

        train_dataset = TextClassificationDataset(
            self.process_texts(train_texts), train_labels
        )
        test_dataset = TextClassificationDataset(
            self.process_texts(test_texts), test_labels
        )

        return {
            "train": train_dataset,
            "test": test_dataset,
            "embeddings": self.embeddings,
            "vocab_size": len(self.word2idx),
            "word2idx": self.word2idx,
            "idx2word": self.idx2word,
            "metadata": {
                "train_size": len(train_dataset),
                "test_size": len(test_dataset),
                "max_seq_length": self.max_seq_length,
                "max_vocab_size": self.max_vocab_size,
                "embedding_dim": self.embedding_dim,
                "max_files_per_class": self.max_files_per_class,
            },
        }


class TextGenerationDataset(Dataset):
    def __init__(self, data_x, data_y):
        self.X = torch.tensor(data_x, dtype=torch.long)
        self.y = torch.tensor(data_y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


class Dataset_Loader_TextGeneration(dataset):
    dataset_source_folder_path = None
    dataset_source_file_name = None

    def __init__(
        self,
        dName=None,
        dDescription=None,
        context_length=3,
        max_rows=None,
    ):
        super().__init__(dName, dDescription)
        self.context_length = context_length
        self.max_rows = max_rows
        self.word2idx = {}
        self.idx2word = {}

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", "", text)
        return text.split()

    def load(self):
        print("Loading text generation data...")

        texts = []
        all_words = []
        data_path = os.path.join(
            self.dataset_source_folder_path, self.dataset_source_file_name
        )
        with open(data_path, "r", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            next(reader, None)
            for row_index, row in enumerate(reader):
                if self.max_rows is not None and row_index >= self.max_rows:
                    break
                if len(row) <= 1:
                    continue
                text = self.clean_text(row[1])
                if len(text) > self.context_length:
                    texts.append(text)
                    all_words.extend(text)

        self.word2idx = {"<PAD>": 0}
        self.idx2word = {0: "<PAD>"}
        for word in all_words:
            if word not in self.word2idx:
                index = len(self.word2idx)
                self.word2idx[word] = index
                self.idx2word[index] = word

        x_data = []
        y_data = []
        for text in texts:
            sequence = [self.word2idx[word] for word in text]
            for index in range(len(sequence) - self.context_length):
                x_data.append(sequence[index : index + self.context_length])
                y_data.append(sequence[index + self.context_length])

        dataset_obj = TextGenerationDataset(x_data, y_data)
        vocab_size = len(self.word2idx)
        print(f"Vocabulary size: {vocab_size}")
        print(f"Training windows: {len(dataset_obj)}")

        return {
            "dataset": dataset_obj,
            "vocab_size": vocab_size,
            "word2idx": self.word2idx,
            "idx2word": self.idx2word,
            "metadata": {
                "context_length": self.context_length,
                "max_rows": self.max_rows,
                "source_rows": len(texts),
                "training_windows": len(dataset_obj),
            },
        }
