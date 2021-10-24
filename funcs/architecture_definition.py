from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class ModelArchitecture:
    input_size: int
    hidden_size: list
    output_size: int = 7

    def parse_architecture(self) -> List[Tuple[int, int]]:
        """
        achar nomes melhores para before e after
        """
        aux = [self.input_size] + self.hidden_size + [self.output_size]
        return [(before, after) for before, after in zip(aux, aux[1:])]
