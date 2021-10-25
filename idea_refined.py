# lib imports
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# in-project imports
from funcs.model import MLP
from funcs.custom_Databases import BeansDataset
from funcs.architecture_definition import ModelArchitecture
from funcs.core import Evaluate
import funcs.raw_data_processing as rdp

"""# getting data
df = pd.read_excel(os.path.join("data", "Dry_Bean_Dataset.xlsx"))
y = df["Class"]
y = y.astype("category").cat.codes
X = df.iloc[:, :-1]

# normalizando X
X = (X - X.min()) / (X.max() - X.min())

# spliting
X_forTraining, X_test, y_forTraining, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42)
X_train, X_eval, y_train, y_eval = train_test_split(
    X_forTraining, y_forTraining, stratify=y_forTraining, test_size=0.1, random_state=33)
"""

train_pack, eval_pack, test_pack, size_pack = rdp.split_data_from(os.path.join("data", "Dry_Bean_Dataset.xlsx")
                                                                   , 0.2, 0.1 )
X_test , y_test = test_pack
X_eval , y_eval = eval_pack
X_train , y_train = train_pack
# datasets and dataloaders
test_dataset = BeansDataset(X_test.to_numpy(), y_test.to_numpy())
eval_dataset = BeansDataset(X_eval.to_numpy(), y_eval.to_numpy())
train_dataset = BeansDataset(X_train.to_numpy(), y_train.to_numpy())

batch_size = 1500
test_loader = DataLoader(dataset=test_dataset, shuffle=True, batch_size=batch_size)
eval_loader = DataLoader(dataset=eval_dataset, shuffle=True, batch_size=batch_size)
train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=batch_size)


# model network definition

arch1 = ModelArchitecture(16, [10])
model = MLP(arch1)

learn_rate = 0.01
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

# training definition
num_epochs = 100
if (torch.cuda.is_available()):
    device = "cuda"
else:
    device = "cpu"
parameters = (num_epochs, criterion, optimizer, device)
ev1 = Evaluate(model, train_loader, eval_loader, test_loader, parameters)
ev1.train()
ev1.report_final_result()