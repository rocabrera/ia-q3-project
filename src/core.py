import torch
import torch.nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from src.model.model import MLP
from typing import Tuple


class Evaluate():

    def __init__(self,
                 model: MLP,
                 train_loader: DataLoader,
                 eval_loader: DataLoader,
                 test_loader: DataLoader,
                 parameters: tuple,
                 verbose=False):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_epochs, self.criterion, self.optimizer = parameters
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.test_loader = test_loader
        self.n_total_steps_train = len(train_loader)
        self.n_total_steps_eval = len(eval_loader)

        self.verbose = verbose
        # Pensar onde vão esses parêmtros depois
        self.eval_history_depth = 10

    def train(self):
        acc_eval_hist = np.zeros(self.eval_history_depth, dtype=np.float32)
        loss_eval_hist = np.zeros(self.eval_history_depth, dtype=np.float32)
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

                if self.verbose:
                    print(
                        f"epoch {epoch + 1} / {self.num_epochs},"
                        f" step {i + 1}/{self.n_total_steps_train},"
                        f" loss = {loss.item():.4f}")
            # epoch end

            # eval
            acc, average_eval_loss = self._fazer_eval()
            if (epoch >= self.eval_history_depth and epoch%self.eval_history_depth == 0):
                # TODO Isso aqui a gente não deveria usar em algum lugar?
                # mean_loss = np.mean(loss_eval_hist)
                mean_acc = np.mean(acc_eval_hist)
                condition = (mean_acc>1.01*acc)
                # podemos trocar condition para ser alguma comparacao entre acc e mean_acc ou o outro par
                if condition:
                    break  # stop training

            acc_eval_hist[epoch % self.eval_history_depth] = acc
            loss_eval_hist[epoch % self.eval_history_depth] = average_eval_loss

    def test_using(self, loader: DataLoader) -> Tuple[float, float]:
        with torch.no_grad():
            accumulated_loss = 0
            n_corrects = 0
            n_samples = 0
            for i, (inputs, targets) in enumerate(loader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device, dtype=torch.int64)
                yhat = self.model(inputs).to(self.device)
                loss = self.criterion(yhat, targets)
                accumulated_loss += loss.item()
                _, prediction = torch.max(yhat, 1)
                n_samples += targets.shape[0]
                n_corrects += (prediction == targets).sum().item()
            # entao, isso abaixo eh ruim pois a ultima batch talvez nao seja de 1500
            # nao eh a media mais contente que eu faria, "but it is fine for now"
            average_loss = accumulated_loss / len(loader)
            acc = n_corrects / n_samples
            return (acc, average_loss)

    def _fazer_eval(self) -> Tuple[float, float]:
        return self.test_using(self.eval_loader)

    def report(self):
        # testagem final
        acc, average_loss = self.test_using(self.test_loader)

        input_size, hidden_size_per_layer, _, activation_function = self.model.arch.get_architecture()

        report = {"model_name": "MLP",
                  "criterion": self.criterion,
                  "num_epochs": self.num_epochs,
                  "input_size": input_size,
                  "hidden_layers": hidden_size_per_layer,
                  "qtd_hidden_layers": len(hidden_size_per_layer),
                  "activation_function": activation_function,
                  "accuracy": acc,
                  
                  # TODO Se quisermos colocar esse parâmetro vamos precisar mandar da cpu para memoria e vice-versa
                  # ?? I made it work, I think
                  "average_loss": average_loss
                  }

        return pd.Series(report)
