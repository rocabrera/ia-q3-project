import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, model_arch):
        super(MLP, self).__init__()
        self.arch = model_arch
        self.camadas = nn.ModuleList([nn.Linear(before, after)
                                      for before, after in model_arch.parse_architecture()])  # achar nomes melhores

        self.relu = nn.ReLU()

    def forward(self, X):
        
        out = self.camadas[0](X)
        for camada in self.camadas[1:]:
            out = self.relu(out)
            out = camada(out)

        return out
    
    def __repr__(self) -> str:
        return ('Arch:' + repr(self.arch) + ', activate: ' + repr(self.relu))
