#lib imports
import os
import numpy as np
from numpy.lib.function_base import average
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
#in-project imports
from funcs.model import MLP
from funcs.custom_Databases import BeansDataset


#getting data
df = pd.read_excel(os.path.join("data","Dry_Bean_Dataset.xlsx"))
y = df["Class"]
y = y.astype("category").cat.codes
X = df.iloc[:,:-1]
#normalizando X
X = (X-X.min())/(X.max()-X.min())

#spliting
X_forTraining, X_test, y_forTraining, y_test = train_test_split(X, y, stratify=y,test_size=0.2, random_state=42)
X_train, X_eval, y_train, y_eval = train_test_split(X_forTraining, y_forTraining, stratify=y_forTraining,test_size=0.1, random_state=33)

#datasets and dataloaders
test_dataset = BeansDataset(X_test.to_numpy(), y_test.to_numpy())
eval_dataset = BeansDataset(X_eval.to_numpy(), y_eval.to_numpy())
train_dataset = BeansDataset(X_train.to_numpy(), y_train.to_numpy())

batch_size = 1500

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, shuffle=True, batch_size=batch_size)
eval_loader = torch.utils.data.DataLoader(dataset=eval_dataset, shuffle=True, batch_size=batch_size)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, shuffle=True, batch_size=batch_size)

#model network definition
learn_rate = 0.01
input_size = 16
output_size = 7
hidden_size = 10

model = MLP(input_size,hidden_size,output_size)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr = learn_rate)

#training definition
num_epochs=100
device="cpu"
n_total_steps_train = len(train_loader)
n_total_steps_eval = len(eval_loader)

#eval history
history_depth = 10
acc_eval_hist = np.zeros(history_depth, dtype=np.float32)
loss_eval_hist = np.zeros(history_depth, dtype=np.float32)


for epoch in range(num_epochs):
    # enumerate mini batches
    for i, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device)
        targets = targets.to(device,dtype=torch.int64)
        # clear the gradients
        optimizer.zero_grad()
        # compute the model output
        yhat = model(inputs)
        # calculate loss
        loss = criterion(yhat, targets)
        # credit assignment
        loss.backward()
        # update model weights
        optimizer.step()
        print(f"epoch {epoch + 1} / {num_epochs}, step {i + 1}/{n_total_steps_train}, loss = {loss.item():.4f}")
    #epoch end

    #eval
    with torch.no_grad():
        accumulated_loss = 0
        n_corrects = 0
        n_samples = 0
        for i, (inputs, targets) in enumerate(eval_loader):
            inputs = inputs.to(device)
            targets = targets.to(device,dtype=torch.int64)
            yhat = model(inputs)
            loss = criterion(yhat, targets)
            accumulated_loss += loss
            _,prediction = torch.max(yhat,1)
            n_samples += targets.shape[0]
            n_corrects += (prediction == targets).sum().item()
        #entao, isso abaixo eh ruim pois a ultima batch talvez nao seja de 1500
        #nao eh a media mais contente que eu faria, "but it is fine for now"
        average_eval_loss = accumulated_loss/n_total_steps_eval
        acc = n_corrects/n_samples

        if (epoch>=history_depth):
            #aqui faz sentido falar de
            mean_loss = np.mean(loss_eval_hist)
            mean_acc = np.mean(acc_eval_hist)
            condition = False
            #podemos trocar condition para ser alguma comparacao entre acc e mean_acc ou o outro par
            if condition:
                break #leaves

        acc_eval_hist[epoch%history_depth] = acc
        loss_eval_hist[epoch%history_depth] = average_eval_loss


#testagem final
with torch.no_grad():
    n_corrects = 0
    n_samples = 0
    for i, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device)
        #print(inputs)
        targets = targets.to(device,dtype=torch.int64)
        outputs = model(inputs)
        #print(outputs)
        
        _,prediction = torch.max(outputs,1)
        #print(prediction)
        n_samples+=targets.shape[0]
        n_corrects += (prediction == targets).sum().item()
        
    acc = 100*n_corrects/n_samples
    print(f'accuracy equals {acc}')