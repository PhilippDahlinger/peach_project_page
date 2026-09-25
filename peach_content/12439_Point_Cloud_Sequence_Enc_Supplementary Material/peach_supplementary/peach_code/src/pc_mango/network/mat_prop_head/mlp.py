from torch import nn

from pc_mango.network.simulator.egno.basic import BaseMLP


class MLP(nn.Module):
    def __init__(self, config, input_dim, output_dim):
        super().__init__()
        if config.activation == "leakyrelu":
            activation = nn.LeakyReLU()
        elif config.activation == "silu":
            activation = nn.SiLU()
        elif config.activation == "relu":
            activation = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation function: {config.activation}")
        self._mlp = BaseMLP(
            input_dim=input_dim,
            hidden_dim=config.hidden_dim,
            output_dim=output_dim,
            activation=activation,
            residual=config.residual,
            last_act=config.last_act,
        )

    def forward(self, x):
        return self._mlp(x)

