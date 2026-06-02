from local_code.base_class.evaluate import evaluate


class Evaluate_TextClassification(evaluate):
    data = None

    def evaluate(self):
        print("Evaluating text classification performance...")
        metrics = {
            "accuracy": self.data.get("test_acc", 0.0),
            "precision": self.data.get("test_precision", 0.0),
            "recall": self.data.get("test_recall", 0.0),
            "f1": self.data.get("test_f1", 0.0),
        }
        print(
            "Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | "
            "Recall: {recall:.4f} | F1: {f1:.4f}".format(**metrics)
        )
        return metrics


class Evaluate_TextGeneration(evaluate):
    data = None

    def evaluate(self):
        print("Evaluating text generation output...")
        generated_text = self.data["generated_text"]
        print(generated_text)
        return generated_text
