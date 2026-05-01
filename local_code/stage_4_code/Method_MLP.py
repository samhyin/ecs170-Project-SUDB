'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.method import method
from local_code.stage_4_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


class Method_MLP(method, nn.Module):
    data = None
    # it defines the max rounds to train the model
    max_epoch = 300
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-3
    loss_history = []
    curve_output_path = None
    show_plots = False

    # it defines the the MLP model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            else "cpu"
        )
        print(f"current device: {self.device}")

        # check here for nn.Linear doc: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        self.fc_layer_1 = nn.Linear(784, 512)
        # check here for nn.ReLU doc: https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
        self.activation_1 = nn.ReLU()
        self.fc_layer_2 = nn.Linear(512, 256)
        # check here for nn.Softmax doc: https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html
        self.activation_2 = nn.ReLU()
        self.fc_layer_3 = nn.Linear(256, 10)
        # self.activation_3 = nn.Softmax(dim=1)

        self.to(self.device)

    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer

    def forward(self, x):
        '''Forward propagation'''
        # hidden layer embeddings
        x = self.activation_1(self.fc_layer_1(x))
        x = self.activation_2(self.fc_layer_2(x))
        # outout layer result
        # self.fc_layer_2(h) will be a nx2 tensor
        # n (denotes the input instance number): 0th dimension; 2 (denotes the class number): 1st dimension
        # we do softmax along dim=1 to get the normalized classification probability distributions for each instance
        # y_pred = self.activation_func_2(self.fc_layer_2(h))
        y_pred = self.fc_layer_3(x)
        return y_pred

    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here

    def train(self, X, y):
        # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # check here for the nn.CrossEntropyLoss doc: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        loss_function = nn.CrossEntropyLoss()
        # for training accuracy investigation purpose
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        self.loss_history = []

        # it will be an iterative gradient updating process
        # we don't do mini-batch, we use the whole input as one batch
        # you can try to split X and y into smaller-sized batches by yourself
        for epoch in range(self.max_epoch): # you can do an early stop if self.max_epoch is too much...
            # get the output, we need to covert X into torch.tensor so pytorch algorithm can operate on it
            x_tensor = torch.FloatTensor(np.array(X)).to(self.device) / 255.0
            # convert y to torch.tensor as well
            y_true = torch.LongTensor(np.array(y)).to(self.device)

            y_pred = self.forward(x_tensor)
            # calculate the training loss
            train_loss = loss_function(y_pred, y_true)

            # check here for the gradient init doc: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
            optimizer.zero_grad()
            # check here for the loss.backward doc: https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html
            # do the error backpropagation to calculate the gradients
            train_loss.backward()
            # check here for the opti.step doc: https://pytorch.org/docs/stable/optim.html
            # update the variables according to the optimizer and the gradients calculated by the above loss.backward function
            optimizer.step()

            self.loss_history.append(train_loss.item())

            if epoch%100 == 0:
                predicted_labels = y_pred.max(1)[1].cpu().numpy()
                true_labels = y_true.cpu().numpy()
                accuracy_evaluator.data = {'true_y': true_labels, 'pred_y': predicted_labels}
                print('Epoch:', epoch, 'Accuracy:', accuracy_evaluator.evaluate(), 'Loss:', train_loss.item())

        self.plot_loss()

    def plot_loss(self):
        print('generating convergence curve...')
        plt.figure(figsize=(10, 5))
        plt.plot(self.loss_history)
        plt.title("Learning Convergence Curve (Stage 2)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.tight_layout()
        if self.curve_output_path:
            output_path = Path(self.curve_output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path)
            print(f"saved convergence curve to: {output_path}")
        if self.show_plots:
            plt.show()
        plt.close()

    def test(self, X):
        # do the testing, and result the result
        x_tensor = torch.FloatTensor(np.array(X)).to(self.device) / 255.0
        with torch.no_grad():
            y_pred = self.forward(x_tensor)
        # convert the probability distributions to the corresponding labels
        # instances will get the labels corresponding to the largest probability
        return y_pred.max(1)[1].cpu().numpy()
    
    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}
            
