""" GraphSAGE model. """

from torch import nn
from torch_geometric.nn import SAGEConv, global_mean_pool

class GraphSAGE(nn.Module):
    """ GraphSAGE. """

    def __init__(self, in_channels, out_channels):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, 16)
        self.conv2 = SAGEConv(16, out_channels)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        return x