from local_code.base_class.evaluate import evaluate


class Evaluate_Node_Classification(evaluate):
    data = None

    def evaluate(self):
        print("Evaluating node classification performance...")
        metrics = self.data["test_metrics"]
        summary = {
            "accuracy": metrics["accuracy"],
            "precision_macro": metrics["precision_macro"],
            "recall_macro": metrics["recall_macro"],
            "f1_macro": metrics["f1_macro"],
            "precision_weighted": metrics["precision_weighted"],
            "recall_weighted": metrics["recall_weighted"],
            "f1_weighted": metrics["f1_weighted"],
        }
        print(
            "Accuracy: {accuracy:.4f} | Precision macro: {precision_macro:.4f} | "
            "Recall macro: {recall_macro:.4f} | F1 macro: {f1_macro:.4f}".format(
                **summary
            )
        )
        return summary
