import torch as ch
import torch.nn as nn

from trust_region_projections.utils.network_utils import get_activation, get_mlp, initialize_weights


class CosntraintNet(nn.Module):
    """
    An example value network, with support for arbitrarily many
    fully connected hidden layers (by default 2 * 64-neuron layers),
    maps a state of size (state_dim) -> a scalar value.
    """

    def __init__(self, input_dim, output_dim=3, init_bounds=[0.04, 0.001, 0.01], init="orthogonal", hidden_sizes=(64,),
                 activation: str = "celu"):
        """
        Initializes the value network.
        Args:
            input_dim: the input dimension of the network (i.e dimension of state)
            output_dim: number of output nodes
            init: initialization of layers
            hidden_sizes: an iterable of integers, each of which represents the size
                    of a hidden layer in the neural network.
            activation: activation of hidden layers
        Returns: Initialized Value network

        """
        """

        """
        super().__init__()

        self.init_bounds = ch.tensor(init_bounds)

        self.activation = get_activation(activation)
        self._affine_layers = get_mlp(input_dim, hidden_sizes, init, activation, False, True)

        self.final = self.get_final(hidden_sizes[-1], output_dim, init)

    def get_final(self, prev_size, output_dim, init):
        final = nn.Linear(prev_size, output_dim)
        initialize_weights(final, init, scale=0.001)
        return final

    def forward(self, x, train=True):
        """
        Performs inference using the value network.
        Inputs:
        - x, the state passed in from the agent
        Returns:
        - The scalar (float) value of that state, as estimated by the net
        """

        self.train(train)

        for affine in self.affine_layers:
            x = self.activation(affine(x))

        final = self.final(x).squeeze(-1)

        return nn.Softplus()(final + self.init_bounds)
