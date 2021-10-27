from dataclasses import dataclass, field
from typing import List, Tuple
import torch.nn as nn


@dataclass
class ModelArchitecture:
    input_size: int  # Quantidade de features do dataset
    hidden_size_per_layer: list  # Número de neurônios das camadas intermediárias
    output_size: int = 7  # Quantidade de classes do dataset
    activation_function: list = field(default_factory=[nn.ReLU()])
                            # list como atributo requer default_factory, vi solução em stackexchange

    def parse_architecture(self) -> List[Tuple[int, int]]:
        """
        achar nomes melhores para before e after
        """
        # TODO é possível juntar essas listas de forma mais simples?
        aux = [self.input_size] + self.hidden_size_per_layer + [self.output_size]
        return [(before, after) for before, after in zip(aux, aux[1:])]

    def get_architecture(self):
        return self.input_size, self.hidden_size_per_layer, self.output_size, self.activation_function

    def __repr__(self) -> str:
        return f"{self.input_size}, ({len(self.hidden_size_per_layer)}, \
                 {repr(self.hidden_size_per_layer)}), {self.output_size}"
