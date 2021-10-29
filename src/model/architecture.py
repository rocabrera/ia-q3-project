from itertools import product
from dataclasses import dataclass, field
from typing import List, Tuple, Union
import torch.nn as nn


@dataclass
class ModelArchitecture:
    input_size: int  # Quantidade de features do dataset
    hidden_size_per_layer: Union[List, Tuple]  # Número de neurônios das camadas intermediárias
    output_size: int = 7  # Quantidade de classes do dataset
    activation_function: List = field(default_factory=[nn.ReLU()])
                            # list como atributo requer default_factory, vi solução em stackexchange

    def parse_architecture(self) -> List[Tuple[int, int]]:
        """
        achar nomes melhores para before e after
        """
        # TODO é possível juntar essas listas de forma mais simples?
        aux = [self.input_size] + list(self.hidden_size_per_layer) + [self.output_size]
        return [(before, after) for before, after in zip(aux, aux[1:])]

    def get_architecture(self):
        return self.input_size, self.hidden_size_per_layer, self.output_size, self.activation_function

    def __repr__(self) -> str:
        return f"{self.input_size}, ({len(self.hidden_size_per_layer)}, {repr(self.hidden_size_per_layer)}), {self.output_size}"
    
class ArchitectureBuilder:
    
    def __init__(self, n_features, n_classes, activation, n_layers, init, end, step):
        self.range = range(init, end+step, step)
        self.product = tuple(self.range for _ in range(n_layers))
        self.n_features = n_features
        self.n_classes = n_classes
        self.activation = activation
        
    def build(self):
        return [ModelArchitecture(self.n_features,
                                  hidden_layer,
                                  self.n_classes,
                                  nn.ReLU()) for hidden_layer in product(*self.product)]
