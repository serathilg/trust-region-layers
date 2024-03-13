import torch as ch
import torch.nn as nn

from trust_region_projections.models.value.vf_net import VFNet
from trust_region_projections.utils.network_utils import initialize_weights


class QFNet(VFNet):

    def __init__(self, input_dim, output_dim=1, init="fanin", hidden_sizes=(256, 256), activation: str = "relu",
                 layer_norm: bool = False):
        super().__init__(input_dim, output_dim, init, hidden_sizes, activation, layer_norm)

    def get_final(self, prev_size, output_dim, init, gain=1.0, scale=1/3):
        final = nn.Linear(prev_size, output_dim)
        # initialize_weights(final, "uniform", init_w=3e-3)
        return final

    def forward(self, x, train=True):
        flat_inputs = ch.cat(x, dim=-1)
        return super().forward(flat_inputs, train=train)
