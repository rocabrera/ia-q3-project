from src.processing.custom_Databases import BeansDataset
from torch.utils.data import DataLoader
import pandas as pd
from typing import Tuple

class Loader_Factory():

    def __init__(self, batch_size):
        self.batch_size = batch_size

    def create_loader(self,data: Tuple[pd.DataFrame,pd.DataFrame]):
        X,y = data
        dataset = BeansDataset(X.to_numpy(), y.to_numpy())
        loader = DataLoader(dataset=dataset,
                         shuffle=True,
                         batch_size=self.batch_size)
        return loader