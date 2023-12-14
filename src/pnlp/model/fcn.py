""" FCN model. """

from torch import nn

class FCN(nn.Module):
    """ Fully Connected Network """

    def __init__(self,
                 fcn_input_size,     # The number of input features
                 fcn_hidden_size,    # The number of features in hidden layer of FCN.
                 device):            # Device ('cpu' or 'cuda')
        super().__init__()
        self.device = device

        # FCN layers
        self.fcn = nn.Sequential(nn.Linear(fcn_input_size, fcn_hidden_size),
                                 nn.ReLU(),
                                 nn.Linear(fcn_hidden_size, 1))  # Adjust this line based on the required output size

    def forward(self, x):
        fcn_out = self.fcn(x)
        fcn_final_out = fcn_out[:, -1, :]
        prediction = fcn_final_out.to(self.device)

        return prediction