# lib imports
import os
from torch.optim import optimizer
from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# in-project imports
from src.model.model import MLP
from src.processing.loader_factory import LoaderFactory
import src.processing.data_processing as dp
from src.model.architecture import ModelArchitecture
from src.core import Trainer, Classificador

# Loading data
df = pd.read_excel(os.path.join("data", "Dry_Bean_Dataset.xlsx"))

# Splitting & processing data
train_pack, eval_pack, test_pack, size_pack, _ = dp.spliting_data(df, test_size = 0.2, eval_size = 0.1)

# Preparing loaders (iterators)
batch_size = 1500
factory = LoaderFactory(batch_size)
test_loader: DataLoader = factory.create_loader(test_pack)
eval_loader: DataLoader = factory.create_loader(eval_pack)
train_loader: DataLoader = factory.create_loader(train_pack)


# training definition
num_epochs = 100
trainer = Trainer(train_loader,
                  eval_loader,
                  test_loader,
                  num_epochs)

# model network definition
n_features, n_classes = size_pack

archs = [ModelArchitecture(n_features,
                           [10**i for j in range(1, 4)],
                           n_classes,
                           nn.ReLU()) for i in range(1, 4)]

print(archs)
# models = [MLP(arch) for arch in archs]

# learn_rate = 0.01

# criterion = nn.CrossEntropyLoss()

# optimizers = [torch.optim.Adam(
#     model.parameters(), lr=learn_rate) for model in models]

# classificadores = [Classificador(model, criterion, opti)
#                    for (model, opti) in zip(models, optimizers)]

# results = []
# for trainee in tqdm(classificadores, total=len(classificadores)):
#     trainer.train(trainee)

#     results.append(trainer.report(trainee))

# print(pd.concat(results, axis=1).T)
