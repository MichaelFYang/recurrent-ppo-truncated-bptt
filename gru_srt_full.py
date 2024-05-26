import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseSpatialRecurrentTransformer(nn.Module):
    def __init__(self, feature_size, pose_size, factor):
        super(BaseSpatialRecurrentTransformer, self).__init__()
        self.feat_size = feature_size
        self.pose_size = pose_size
        # Feature Normalization
        self.norm = nn.LayerNorm(feature_size)
        # Gate linear layer
        self.gate_fc = nn.Linear(feature_size, feature_size)
        # Linear layer to generate transformation matrix from pose
        self.affine_layers = nn.Sequential(
            nn.Linear(pose_size, feature_size),
            nn.Tanh(),
            nn.Linear(feature_size, 6),
        )
        # Init bias as identity matrix
        self.affine_layers[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def _create_upscale_layers(self, channels, factor):
        layers = []
        for _ in range(factor):
            out_channels = channels // 2
            layers.extend([
                nn.ConvTranspose2d(channels, out_channels, kernel_size=2, stride=2, bias=False),
                nn.InstanceNorm2d(out_channels),
            ])
            channels = out_channels
        return nn.Sequential(*layers)

    def _create_downscale_layers(self, channels, factor):
        layers = []
        for _ in range(factor):
            out_channels = channels * 2
            layers.append(nn.Conv2d(channels, out_channels, kernel_size=2, stride=2, bias=False))
            channels = out_channels
        return nn.Sequential(*layers)

    def get_theta(self, p):
        theta = self.affine_layers(p)
        return theta.view(-1, 2, 3)

class SpatialRecurrentTransformer(BaseSpatialRecurrentTransformer):
    def __init__(self, feature_size, pose_size, factor):
        super(SpatialRecurrentTransformer, self).__init__(feature_size, pose_size, factor)
        self.upscale_seq = self._create_upscale_layers(feature_size, factor)
        self.downscale_seq = self._create_downscale_layers(feature_size // 2**factor, factor)

    def forward(self, p, h):
        b, d = h.size()

        # Normalize the input feature vector
        h = self.norm(h)

        # Map 1D feature vector to a 2D grid
        h_s = self.upscale_seq(h.view(b, d, 1, 1))

        # Spatial affine transformation matrix from pose
        theta = self.get_theta(p)

        # Apply spatial transformation to the grid
        grid = F.affine_grid(theta, h_s.size(), align_corners=False)
        h_s = F.grid_sample(h_s, grid, align_corners=False)

        # Map 2D grid back to 1D feature vector
        h_s = self.downscale_seq(h_s)
        h_s = h_s.view(b, d)
        
        # Gated Vector
        h_g = self.gate_fc(h)

        # Gated residual connection
        h = h_s * torch.tanh(h_g)

        return h
    
class GRUSRT(nn.Module):
    def __init__(self, input_size, hidden_size, pose_size, num_layers=1, batch_first=False, grid_factor=3):
        super(GRUSRT, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.rnn = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=False)
        self.srt_layers = nn.ModuleList([SpatialRecurrentTransformer(hidden_size, pose_size, grid_factor) 
                                         for _ in range(num_layers)])

    def forward(self, x, pose, state=None):
        if self.batch_first:
            x = x.permute(1, 0, 2)
            pose = pose.permute(1, 0, 2)
        
        l, b, _ = x.shape
        if state is None:
            state = torch.zeros((self.num_layers, b, self.hidden_size), device=x.device)  # (J, B, D)
        
        out_seq = []
        for i in range(l):
            state_list = []
            for j in range(self.num_layers):
                new_state = self.srt_layers[j](pose[i], state[j])
                state_list.append(new_state)
            state = torch.stack(state_list, dim=0)
            out, state = self.rnn(x[i:i+1], state)
            out_seq.append(out)
        out_seq = torch.cat(out_seq, dim=0)
        if self.batch_first:
            out_seq = out_seq.permute(1, 0, 2)
        return out_seq, state
    
if __name__ == '__main__':
    batch_size = 3
    input_size = 15
    hidden_size = 128
    pose_size = 12
    output_size = 3
    seq_len = 10
    batch_first = False
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    x = torch.randn(seq_len, batch_size, input_size).to(device)
    pose = torch.randn(seq_len, batch_size, pose_size).to(device)
    if batch_first:
        x = x.permute(1, 0, 2)
        pose = pose.permute(1, 0, 2)
    print(f'Input shape: {x.shape}')
    print(f'Pose shape: {pose.shape}')
    print(f'Batch first: {batch_first}')
        
    gru_srt = GRUSRT(input_size, hidden_size, pose_size, num_layers=2, batch_first=batch_first).to(device)
    
    out, state = gru_srt(x, pose, None)
    print(f'Output shape: {out.shape}')
    print(f'State shape: {state.shape}')