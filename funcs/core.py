
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from funcs.model import MLP


class Evaluate():

    def __init__(self,
                 model: MLP,
                 train_loader: DataLoader,
                 eval_loader: DataLoader,
                 test_loader: DataLoader,
                 parameters: tuple):

        self.model = model
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.test_loader = test_loader
        self.n_total_steps_train = len(train_loader)
        self.n_total_steps_eval = len(eval_loader)

        self.num_epochs, self.criterion, self.optimizer, self.device = parameters

        # Pensar onde vão esses parêmtros depois
        self.history_depth = 10
        self.acc_eval_hist = np.zeros(self.history_depth, dtype=np.float32)
        self.loss_eval_hist = np.zeros(self.history_depth, dtype=np.float32)

    def train(self):
        for epoch in range(self.num_epochs):
            # enumerate mini batches
            for i, (inputs, targets) in enumerate(self.train_loader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device, dtype=torch.int64)
                # clear the gradients
                self.optimizer.zero_grad()
                # compute the model output
                yhat = self.model(inputs)
                # calculate loss
                loss = self.criterion(yhat, targets)
                # credit assignment
                loss.backward()
                # update model weights
                self.optimizer.step()
                print(
                    f"epoch {epoch + 1} / {self.num_epochs}, step {i + 1}/{self.n_total_steps_train}, loss = {loss.item():.4f}")
            # epoch end

            # eval
            with torch.no_grad():
                accumulated_loss = 0
                n_corrects = 0
                n_samples = 0
                for i, (inputs, targets) in enumerate(self.eval_loader):
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device, dtype=torch.int64)
                    yhat = self.model(inputs)
                    loss = self.criterion(yhat, targets)
                    accumulated_loss += loss
                    _, prediction = torch.max(yhat, 1)
                    n_samples += targets.shape[0]
                    n_corrects += (prediction == targets).sum().item()
                # entao, isso abaixo eh ruim pois a ultima batch talvez nao seja de 1500
                # nao eh a media mais contente que eu faria, "but it is fine for now"
                average_eval_loss = accumulated_loss / self.n_total_steps_eval
                acc = n_corrects / n_samples

                if (epoch >= self.history_depth):
                    # aqui faz sentido falar de
                    mean_loss = np.mean(self.loss_eval_hist)
                    mean_acc = np.mean(self.acc_eval_hist)
                    condition = False
                    # podemos trocar condition para ser alguma comparacao entre acc e mean_acc ou o outro par
                    if condition:
                        break  # leaves

                self.acc_eval_hist[epoch % self.history_depth] = acc
                self.loss_eval_hist[epoch % self.history_depth] = average_eval_loss

#    def _fazendo_evaluate()

    def report_final_result(self):
        # testagem final
        with torch.no_grad():
            n_corrects = 0
            n_samples = 0
            for i, (inputs, targets) in enumerate(self.test_loader):
                inputs = inputs.to(self.device)
                # print(inputs)
                targets = targets.to(self.device, dtype=torch.int64)
                outputs = self.model(inputs)
                # print(outputs)

                _, prediction = torch.max(outputs, 1)
                # print(prediction)
                n_samples += targets.shape[0]
                n_corrects += (prediction == targets).sum().item()

            acc = 100 * n_corrects / n_samples
            print(f'accuracy equals {acc}')
