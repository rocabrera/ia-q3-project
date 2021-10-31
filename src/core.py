# libs do python
import os
import pathlib
from typing import Tuple
# nossas libs
from src.model.model import MLP
# libs de terceiros
import torch
import torch.nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from src.logger.logging import logger
from pathlib import Path


class Classifier:

    def __init__(self,
                 model: MLP,
                 criterion,
                 optimizer):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer

    def attempt(self, X_input: torch.Tensor):
        X = X_input.to(self.device)
        y_hat = self.model(X)
        return y_hat

    def predict(self, X_input: torch.Tensor):
        with torch.no_grad():
            y_hat = self.attempt(X_input)
        return y_hat

    def learn(self, prediction: torch.Tensor, targets: torch.Tensor) -> None:
        targets_in_device = targets.to(self.device, dtype=torch.int64)
        loss = self.criterion(prediction, targets_in_device)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def test(self, X_input: torch.Tensor, y_input: torch.Tensor):
        with torch.no_grad():
            targets = y_input.to(self.device, dtype=torch.int64)
            y_hat = self.predict(X_input)
            loss = self.criterion(y_hat, targets)
            _, prediction = torch.max(y_hat, 1)
            n_samples = targets.shape[0]
            n_corrects = (prediction == targets).sum().item()
        return n_samples, n_corrects, loss.item()


class Trainer:

    def __init__(self,
                 train_loader: DataLoader,
                 eval_loader: DataLoader,
                 test_loader: DataLoader,
                 num_epochs: int):

        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.test_loader = test_loader
        self.num_epochs = num_epochs
        # Pensar onde vão esses parêmtros depois
        self.eval_history_depth = 10
        self.folder_results = "results"
        Path(self.folder_results).mkdir(parents=True, exist_ok=True)
        
        self.file_result_path = os.path.join(self.folder_results,
                                             "model_results.csv")

    def define_training(self, classifier):
        """
        Essa fução contém as informações necessárias de treinamento
        """
        (input_size,
         hidden_size_per_layer,
         _,
         activation_function) = classifier.model.arch.get_architecture()

        model_name = "MLP"
        criterion_name = type(classifier.criterion).__name__
        optimizer_name = type(classifier.optimizer).__name__
        learning_rate = classifier.optimizer.param_groups[0]['lr']
        num_epochs = self.num_epochs
        qtd_hidden_layers = len(hidden_size_per_layer)

        return {"model_name": model_name,
                "criterion": criterion_name,
                "optimizer": optimizer_name,
                "learning_rate": learning_rate,
                "num_epochs": num_epochs,
                "input_size": input_size,
                "hidden_layers": list(hidden_size_per_layer),
                "qtd_hidden_layers": len(hidden_size_per_layer),
                "activation_function": type(activation_function).__name__}

    def is_alredy_trained(self):

        if os.path.isfile(self.file_result_path):
            already_done = pd.read_csv(self.file_result_path)
            cond_model_name = already_done["model_name"] == self.parameters["model_name"]
            cond_criterion = already_done["criterion"] == self.parameters["criterion"]
            cond_optimizer = already_done["optimizer"] == self.parameters["optimizer"]
            cond_learning_rate = already_done["learning_rate"] == self.parameters["learning_rate"]
            cond_num_epochs = already_done["num_epochs"] == self.parameters["num_epochs"]
            cond_input_size = already_done["input_size"] == self.parameters["input_size"]
            cond_activation_function = already_done["activation_function"] == self.parameters["activation_function"]
            cond_hidden_layers = (already_done.hidden_layers
                                  .apply(lambda x: x == str(self.parameters["hidden_layers"])))

            result = already_done[cond_model_name &
                                  cond_criterion &
                                  cond_optimizer &
                                  cond_learning_rate &
                                  cond_num_epochs &
                                  cond_input_size &
                                  cond_activation_function &
                                  cond_hidden_layers]

            if len(result):
                logger.info(f'Already trained: lr = {self.parameters["learning_rate"]}, num_epochs = {self.parameters["num_epochs"]}, hidden_layers = {str(self.parameters["hidden_layers"])}')
                return True
            else:
                return False

    def train(self, classifier: Classifier):

        self.parameters = self.define_training(classifier)

        if self.is_alredy_trained():
            return False

        acc_eval_hist = np.zeros(self.eval_history_depth, dtype=np.float32)
        loss_eval_hist = np.zeros(self.eval_history_depth, dtype=np.float32)
        for epoch in range(self.num_epochs):
            # enumerate mini batches
            for i, (inputs, targets) in enumerate(self.train_loader):
                # compute the model output
                yhat = classifier.attempt(inputs)
                classifier.learn(yhat, targets)
            # epoch end

            # eval
            acc, average_eval_loss = self.evaluate(
                classifier, self.eval_loader)

            if (epoch >= self.eval_history_depth and epoch % self.eval_history_depth == 0):
                # TODO Isso aqui a gente não deveria usar em algum lugar?
                # mean_loss = np.mean(loss_eval_hist)
                mean_acc = np.mean(acc_eval_hist)
                condition = (mean_acc > 1.01 * acc)  # a mudar
                if condition:
                    break  # stop training

            acc_eval_hist[epoch % self.eval_history_depth] = acc
            loss_eval_hist[epoch % self.eval_history_depth] = average_eval_loss

        return True

    def evaluate(self, classifier: Classifier, loader: DataLoader):
        samples = 0
        corrects = 0
        accumulated_loss = 0
        for i, (inputs, targets) in enumerate(loader):
            new_samples, new_corrects, loss = classifier.test(inputs, targets)
            samples += new_samples
            corrects += new_corrects
            accumulated_loss += loss
        acc = corrects / samples
        average_eval_loss = accumulated_loss / len(loader)
        return acc, average_eval_loss

        # TODO Precisa de pd.DataFrame ... tentar trocar para pd.Series
    def save_df(self, parameters: dict):
        with open(self.file_result_path, 'a') as f:
            df = pd.DataFrame([parameters])
            df.to_csv(f, mode='a', header=f.tell() == 0, index=False)

    def save_report(self, classifier: Classifier):
        # testagem final
        acc, average_loss = self.evaluate(classifier, self.test_loader)

        (input_size,
         hidden_size_per_layer,
         _,
         activation_function) = classifier.model.arch.get_architecture()

        self.parameters["accuracy"] = acc
        self.parameters["average_loss"] = average_loss

        self.save_df(self.parameters)
