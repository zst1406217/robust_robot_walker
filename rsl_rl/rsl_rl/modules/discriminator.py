
import torch
import torch.nn as nn

class Discriminator(torch.nn.Module):

    def __init__(
            self,
            state_size,
            hidden_sizes,  # Can be empty list or None for none.
            ):
        super(Discriminator, self).__init__()
        
        layers=[]
        activation=nn.ELU()
        
        layers.append(nn.Linear(state_size*2, hidden_sizes[0]))
        layers.append(activation)
        for l in range(len(hidden_sizes)):
            if l == len(hidden_sizes) - 1:
                layers.append(nn.Linear(hidden_sizes[l], 1))
                # layers.append(nn.Tanh())
            else:
                layers.append(nn.Linear(hidden_sizes[l], hidden_sizes[l + 1]))
                layers.append(activation)
        self.backbone = nn.Sequential(*layers)

    def forward(self, input):
        """Compute the model on the input, assuming input shape [B, state_size*2]."""
        return self.backbone(input)
