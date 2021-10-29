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
from src.model.architecture import ArchitectureBuilder
from src.core import Trainer, Classifier


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

num_epochs = [50,100,200,300]

for num_epoch in tqdm(num_epochs, desc="num_epoch", position=0):
    
    trainer = Trainer(train_loader,
                      eval_loader,
                      test_loader,
                      num_epoch)
    # model network definition
    n_features, n_classes = size_pack

    activation = nn.ReLU()
    init = 10
    end = 100
    step = 5
    n_layers = 1
    
    archs = ArchitectureBuilder(n_features, 
                                 n_classes, 
                                 activation, 
                                 n_layers, 
                                 init, 
                                 end, 
                                 step).build()
    
    models = [MLP(arch) for arch in archs]
    
    learning_rates = [0.001, 0.01,  0.1, 0.15]
    
    for learning_rate in tqdm(learning_rates, desc="learning rate",position=1):
        criterion = nn.CrossEntropyLoss()
        optimizers = [torch.optim.Adam(model.parameters(), lr=learning_rate) for model in models]
        classificadores = [Classifier(model, criterion, optimizer) 
                           for (model, optimizer) in zip(models, optimizers)]

        for classificador in tqdm(classificadores, total=len(classificadores), desc="classifier", position=2, leave=False):
            if trainer.train(classificador):
                trainer.save_report(classificador)