'''
Concrete Evaluate class for a specific evaluation metrics
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class Evaluate_Accuracy(evaluate):
    data = None
    
    def evaluate(self):
        print('evaluating performance...')
        # return accuracy_score(self.data['true_y'], self.data['pred_y'])
        y_true = self.data['true_y']
        y_pred = self.data['pred_y']

        acc = accuracy_score(y_true, y_pred)
        pre = precision_score(y_true, y_pred, average='macro')
        rec = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')

        print(f"Accuracy: {acc:.4f}")
        print(f"Precision (Macro): {pre:.4f}")
        print(f"Recall (Macro): {rec:.4f}")
        print(f"F1 Score (Macro): {f1:.4f}")

        return acc
        