from code.base_class.method import method
from code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

class Method_MLP(method, nn.Module):
    data = None
    max_epoch = 200
    learning_rate = 1e-3

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.fc_layer_1 = nn.Linear(784, 256)
        self.activation_func_1 = nn.ReLU()

        self.fc_layer_2 = nn.Linear(256, 128)
        self.activation_func_2 = nn.ReLU()

        self.fc_layer_3 = nn.Linear(128, 10)

        self.epoch_history = []
        self.loss_history = []

    def forward(self, x):
        h1 = self.activation_func_1(self.fc_layer_1(x))
        h2 = self.activation_func_2(self.fc_layer_2(h1))
        y_pred = self.fc_layer_3(h2)
        return y_pred

    def train(self, X, y):
        self.epoch_history = []
        self.loss_history = []

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        X_tensor = torch.FloatTensor(np.array(X)) / 255.0
        y_true = torch.LongTensor(np.array(y))

        for epoch in range(self.max_epoch):
            y_pred = self.forward(X_tensor)
            train_loss = loss_function(y_pred, y_true)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            self.epoch_history.append(epoch + 1)
            self.loss_history.append(train_loss.item())

            if epoch % 10 == 0:
                accuracy_evaluator.data = {
                    'true_y': y_true,
                    'pred_y': y_pred.max(1)[1]
                }
                print('Epoch:', epoch,
                      'Accuracy:', accuracy_evaluator.evaluate(),
                      'Loss:', train_loss.item())

        plt.figure(figsize=(8, 5))
        plt.plot(self.epoch_history, self.loss_history)
        plt.xlabel('Epoch')
        plt.ylabel('Training Loss')
        plt.title('Training Convergence Plot')
        plt.grid(True)
        plt.savefig('training_convergence.png', dpi=300, bbox_inches='tight')
        plt.show()

    def test(self, X):
        with torch.no_grad():
            X_tensor = torch.FloatTensor(np.array(X)) / 255.0
            y_pred = self.forward(X_tensor)
            return y_pred.max(1)[1]

    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'])

        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])

        true_y = np.array(self.data['test']['y'])
        pred_y_np = pred_y.cpu().numpy()

        accuracy = np.mean(pred_y_np == true_y)

        precision_macro = precision_score(true_y, pred_y_np, average='macro', zero_division=0)
        recall_macro = recall_score(true_y, pred_y_np, average='macro', zero_division=0)
        f1_macro = f1_score(true_y, pred_y_np, average='macro', zero_division=0)

        precision_weighted = precision_score(true_y, pred_y_np, average='weighted', zero_division=0)
        recall_weighted = recall_score(true_y, pred_y_np, average='weighted', zero_division=0)
        f1_weighted = f1_score(true_y, pred_y_np, average='weighted', zero_division=0)

        print('************ Overall Performance ************')
        print('MLP Accuracy:', accuracy)
        print('Macro Precision:', precision_macro)
        print('Macro Recall:', recall_macro)
        print('Macro F1:', f1_macro)
        print('Weighted Precision:', precision_weighted)
        print('Weighted Recall:', recall_weighted)
        print('Weighted F1:', f1_weighted)
        print('************ Finish ************')

        return {
            'pred_y': pred_y,
            'true_y': self.data['test']['y'],
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted
        }