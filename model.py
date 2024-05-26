import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F
from lstm_srt_full import LSTMSRT
from gru_srt_full import GRUSRT

class ActorCriticModel(nn.Module):
    def __init__(self, config, observation_space, action_space_shape):
        """Model setup

        Arguments:
            config {dict} -- Configuration and hyperparameters of the environment, trainer and model.
            observation_space {box} -- Properties of the agent's observation space
            action_space_shape {tuple} -- Dimensions of the action space
        """
        super().__init__()
        self.hidden_size = config["hidden_layer_size"]
        self.recurrence = config["recurrence"]
        self.observation_space_shape = observation_space.shape
        self.image_size = (3, 12, 12)

        # Observation encoder
        # if len(self.observation_space_shape) > 1:
            # Case: visual observation is available
            # Visual encoder made of 3 convolutional layers
            # self.conv1 = nn.Conv2d(observation_space.shape[0], 32, 5, 2, 1)
            # self.conv2 = nn.Conv2d(32, 64, 3, 1, 0)
            # self.conv3 = nn.Conv2d(64, 64, 3, 1, 0)
            # nn.init.orthogonal_(self.conv1.weight)
            # nn.init.orthogonal_(self.conv2.weight)
            # nn.init.orthogonal_(self.conv3.weight)
        self.convs = nn.Sequential(
            nn.Conv2d(self.image_size[0], 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        # Compute output size of convolutional layers
        self.conv_out_size = self.get_conv_output(self.image_size)
        in_features_next_layer = self.conv_out_size
        
        # fuse layer
        img_size = self.image_size[0] * self.image_size[1] * self.image_size[2]
        in_features_next_layer += observation_space.shape[0] - img_size

        # Recurrent layer (GRU or LSTM)
        if self.recurrence["layer_type"] == "gru":
            self.recurrent_layer = nn.GRU(in_features_next_layer, self.recurrence["hidden_state_size"], batch_first=True)
        elif self.recurrence["layer_type"] == "lstm":
            self.recurrent_layer = nn.LSTM(in_features_next_layer, self.recurrence["hidden_state_size"], batch_first=True)
        elif self.recurrence["layer_type"] == "lstm_srt":
            self.recurrent_layer = LSTMSRT(in_features_next_layer, self.recurrence["hidden_state_size"], 3, batch_first=True)
        elif self.recurrence["layer_type"] == "gru_srt":
            self.recurrent_layer = GRUSRT(in_features_next_layer, self.recurrence["hidden_state_size"], 3, batch_first=True)
        # Init recurrent layer
        # for name, param in self.recurrent_layer.named_parameters():
        #     if "bias" in name:
        #         nn.init.constant_(param, 0)
        #     elif "weight" in name:
        #         nn.init.orthogonal_(param, np.sqrt(2))
        
        # Hidden layer
        self.lin_hidden = nn.Linear(self.recurrence["hidden_state_size"], self.hidden_size)
        # nn.init.orthogonal_(self.lin_hidden.weight, np.sqrt(2))

        # Decouple policy from value
        # Hidden layer of the policy
        self.lin_policy = nn.Linear(self.hidden_size, self.hidden_size)
        # nn.init.orthogonal_(self.lin_policy.weight, np.sqrt(2))

        # Hidden layer of the value function
        self.lin_value = nn.Linear(self.hidden_size, self.hidden_size)
        # nn.init.orthogonal_(self.lin_value.weight, np.sqrt(2))

        # Outputs / Model heads
        # Policy (Multi-discrete categorical distribution)
        self.policy_branches = nn.ModuleList()
        for num_actions in action_space_shape:
            actor_branch = nn.Linear(in_features=self.hidden_size, out_features=num_actions)
            # nn.init.orthogonal_(actor_branch.weight, np.sqrt(0.01))
            self.policy_branches.append(actor_branch)

        # Value function
        self.value = nn.Linear(self.hidden_size, 1)
        # nn.init.orthogonal_(self.value.weight, 1)

    def forward(self, obs:torch.tensor, recurrent_cell:torch.tensor, device:torch.device, sequence_length:int=1):
        """Forward pass of the model

        Arguments:
            obs {torch.tensor} -- Batch of observations
            recurrent_cell {torch.tensor} -- Memory cell of the recurrent layer
            device {torch.device} -- Current device
            sequence_length {int} -- Length of the fed sequences. Defaults to 1.

        Returns:
            {Categorical} -- Policy: Categorical distribution
            {torch.tensor} -- Value Function: Value
            {tuple} -- Recurrent cell
        """
        # Set observation as input to the model
        img_size = self.image_size[0] * self.image_size[1] * self.image_size[2]
        image = obs[:, :img_size].view(-1, 
                                  self.image_size[1], 
                                  self.image_size[2], 
                                  self.image_size[0]).permute(0, 3, 1, 2)
        # Forward observation encoder
        # if len(self.observation_space_shape) > 1:
        batch_size = obs.size()[0]
        # Propagate input through the visual encoder
        image = self.convs(image)
        # Flatten the output of the convolutional layers
        image = image.reshape((batch_size, -1))
        dpose = obs[:, img_size:]
        
        # concatenate image and dpose
        h = torch.cat((image, dpose), dim=-1)

        # Forward reccurent layer (GRU or LSTM)
        if sequence_length == 1:
            # Case: sampling training data or model optimization using sequence length == 1
            if "srt" in self.recurrence["layer_type"]:
                # Case: Spatial Recurrent Transformer
                h, recurrent_cell = self.recurrent_layer(h.unsqueeze(1), dpose.unsqueeze(1), recurrent_cell)
            else:
                # Case: GRU or LSTM
                h, recurrent_cell = self.recurrent_layer(h.unsqueeze(1), recurrent_cell)
            h = h.squeeze(1) # Remove sequence length dimension
        else:
            # Case: Model optimization given a sequence length > 1
            # Reshape the to be fed data to batch_size, sequence_length, data
            h_shape = tuple(h.size())
            h = h.reshape(h_shape[0] // sequence_length, sequence_length, h_shape[1])
            if "srt" in self.recurrence["layer_type"]:
                # Case: Spatial Recurrent Transformer
                dpose = dpose.reshape((h_shape[0] // sequence_length), sequence_length, dpose.size()[1])
                # Forward recurrent layer
                h, recurrent_cell = self.recurrent_layer(h, dpose, recurrent_cell)
            else:
                # Case: GRU or LSTM
                # Forward recurrent layer
                h = h.reshape(h_shape[0] // sequence_length, sequence_length, h_shape[1])
                h, recurrent_cell = self.recurrent_layer(h, recurrent_cell)

            # Reshape to the original tensor size
            h_shape = tuple(h.size())
            h = h.reshape(h_shape[0] * h_shape[1], h_shape[2])

        # The output of the recurrent layer is not activated as it already utilizes its own activations.

        # Feed hidden layer
        h = F.relu(self.lin_hidden(h))

        # Decouple policy from value
        # Feed hidden layer (policy)
        h_policy = F.relu(self.lin_policy(h))
        # Feed hidden layer (value function)
        h_value = F.relu(self.lin_value(h))
        # Head: Value function
        value = self.value(h_value).reshape(-1)
        # Head: Policy
        pi = [Categorical(logits=branch(h_policy)) for branch in self.policy_branches]

        return pi, value, recurrent_cell

    def get_conv_output(self, shape:tuple) -> int:
        """Computes the output size of the convolutional layers by feeding a dummy tensor.

        Arguments:
            shape {tuple} -- Input shape of the data feeding the first convolutional layer

        Returns:
            {int} -- Number of output features returned by the utilized convolutional layers
        """
        o = self.convs(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
 
    def init_recurrent_cell_states(self, num_sequences:int, device:torch.device) -> tuple:
        """Initializes the recurrent cell states (hxs, cxs) as zeros.

        Arguments:
            num_sequences {int} -- The number of sequences determines the number of the to be generated initial recurrent cell states.
            device {torch.device} -- Target device.

        Returns:
            {tuple} -- Depending on the used recurrent layer type, just hidden states (gru) or both hidden states and
                     cell states are returned using initial values.
        """
        hxs = torch.zeros((num_sequences), self.recurrence["hidden_state_size"], dtype=torch.float32, device=device).unsqueeze(0)
        cxs = None
        if "lstm" in self.recurrence["layer_type"]:
            cxs = torch.zeros((num_sequences), self.recurrence["hidden_state_size"], dtype=torch.float32, device=device).unsqueeze(0)
        return hxs, cxs