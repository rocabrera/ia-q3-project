# lib imports
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# in-project imports
from src.model import MLP
from src.processing.custom_Databases import BeansDataset
import src.processing.raw_data_processing as rdp
from src.model.architecture import ModelArchitecture
from src.core import Evaluate

# Spliting data in Train/Eval/Test
train_pack, eval_pack, test_pack, size_pack = rdp.split_data_from(
    os.path.join("data", "Dry_Bean_Dataset.xlsx"), 0.2, 0.1)
# unpacking
X_test, y_test = test_pack
X_eval, y_eval = eval_pack
X_train, y_train = train_pack

"""
# TODO Essa parte de dataset e dataloader talvez devesse estar em um lugar próprio
"""

# datasets and dataloaders
test_dataset = BeansDataset(X_test.to_numpy(), y_test.to_numpy())
eval_dataset = BeansDataset(X_eval.to_numpy(), y_eval.to_numpy())
train_dataset = BeansDataset(X_train.to_numpy(), y_train.to_numpy())

batch_size = 1500
test_loader = DataLoader(dataset=test_dataset,
                         shuffle=True,
                         batch_size=batch_size)

eval_loader = DataLoader(dataset=eval_dataset,
                         shuffle=True,
                         batch_size=batch_size)

train_loader = DataLoader(dataset=train_dataset,
                          shuffle=True,
                          batch_size=batch_size)

##############################################################################

# model network definition
n_features, n_classes = size_pack
arch1 = ModelArchitecture(n_features, [10], n_classes, nn.ReLU())
model = MLP(arch1)

# TODO Não sei onde essa parte vai, mas eu acho que aqui não está legal
learn_rate = 0.01
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

# training definition
num_epochs = 100
parameters = (num_epochs, criterion, optimizer)

"""
# TODO Esse evaluate está fazendo muita coisa. 
# Talvez no futuro separa em duas classes onde uma treina e outra prediz
"""

ev1 = Evaluate(model, train_loader, eval_loader, test_loader, parameters)

ev1.train()
ev1.report_final_result()