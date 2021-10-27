# lib imports
import os
from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# in-project imports
from src.model.model import MLP
from src.processing.Loader_Factory import Loader_Factory
import src.processing.raw_data_processing as rdp
from src.model.architecture import ModelArchitecture
from src.core import Evaluate

# Spliting data in Train/Eval/Test
df = rdp.get_dataframe_from(os.path.join("data", "Dry_Bean_Dataset.xlsx"))
train_pack, eval_pack, test_pack, size_pack, _ = rdp.df_tet_split(df, 0.2, 0.1)

# TODO Essa parte de dataset e dataloader talvez devesse estar em um lugar próprio
    # Era isso que tu queria?
# datasets and dataloaders
batch_size = 1500
factory = Loader_Factory(batch_size)
test_loader = factory.create_loader(test_pack)
eval_loader = factory.create_loader(eval_pack)
train_loader = factory.create_loader(train_pack)

##############################################################################

# model network definition
n_features, n_classes = size_pack
archs = [ModelArchitecture(n_features, [10**i], n_classes, nn.ReLU())
         for i in range(1, 6)]
models = [MLP(arch) for arch in archs]


# TODO Não sei onde essa parte vai, mas eu acho que aqui não está legal
learn_rate = 0.01

criterion = nn.CrossEntropyLoss()

optimizers = [torch.optim.Adam(
    model.parameters(), lr=learn_rate) for model in models]

# training definition
num_epochs = 100

# TODO Tem problema não instanciar um nn.CrossEntropy por modelo?
models_parameters = [(num_epochs, criterion, optimizer)
                     for optimizer in optimizers]

"""
# TODO Esse evaluate está fazendo muita coisa. 
# Talvez no futuro separa em duas classes onde uma treina e outra prediz
"""

results = []

for model, parameters in tqdm(zip(models, models_parameters), total=len(models)):
    ev = Evaluate(model, train_loader, eval_loader, test_loader, parameters)
    ev.train()
    results.append(ev.report())

print(pd.concat(results, axis=1).T)
