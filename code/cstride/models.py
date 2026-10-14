"""Checkpoint-compatible C-STRIDE encoder and decoder models."""
import torch
from .mmnn import MMNN
from .modulation import LatentToModulation, Bias

class FourierEmbedding(torch.nn.Module):
    def __init__(self, num_frequencies, input_dim):
        super(FourierEmbedding, self).__init__()
        self.num_frequencies = num_frequencies
        self.input_dim = input_dim
        self.encoding = torch.nn.Linear(input_dim, num_frequencies, bias=False)

    def forward(self, x):
        # x shape: (batch_size, *, input_dim)
        # Apply Fourier embedding
        x_projected = self.encoding(2 * torch.pi * x)
        return torch.cat([x, torch.sin(x_projected), torch.cos(x_projected)], dim=-1)

class ModulatedFMMNN(MMNN):
    def __init__(self, dim_in, dim_out, rank, width, num_layers, device='cpu', ResNet=False, fixWb=True, init_scaling=True,
                 modulate_scale=False, modulate_shift=True, use_latent=False, latent_dim=64, modulation_net_dim_hidden=64, modulation_net_num_layers=1):
        '''
        ModulatedFMMNN model definition

        Args:
        dim_in (int): Dimension of input.
        dim_out (int): Dimension of output.
        rank (int): Rank of FMMNN.
        width (int): Width of FMMNN.
        num_layers (int): Number of layers.
        device (str): Device to run the model on.
        ResNet (bool): Whether to use ResNet connections.
        fixWb (bool): Whether to fix weights and biases.
        init_scaling (bool): Whether to initialize scaling.

        modulate_scale (bool): Whether to modulate with scales.
        modulate_shift (bool): Whether to modulate with shifts.
        use_latent (bool): If true, use a latent vector which is mapped to
            modulations, otherwise use modulations directly.
        latent_dim (int): Dimension of latent vector.
        modulation_net_dim_hidden (int): Number of hidden dimensions of
            modulation network.
        modulation_net_num_layers (int): Number of layers in modulation network.
            If this is set to 1 will correspond to a linear layer.
        '''

        super().__init__(
            ranks=[dim_in] + [rank] * (num_layers-1) + [dim_out],
            widths=[width] * num_layers,
            device=device,
            ResNet=ResNet,
            fixWb=fixWb,
            init_scaling=init_scaling
        )

        assert modulate_scale or modulate_shift

        self.rank = rank
        self.device = torch.device(device)
        self.ResNet = ResNet
        self.fixWb = fixWb
        self.init_scaling = init_scaling
        self.modulate_scale = modulate_scale
        self.modulate_shift = modulate_shift
        # self.activations = [Rational().to(self.device) for _ in range(self.depth)]

        num_modulations = rank * (num_layers - 1)
        if self.modulate_scale and self.modulate_shift:
            num_modulations *= 2

        if use_latent:
            self.modulation_net = LatentToModulation(
                latent_dim,
                num_modulations,
                modulation_net_dim_hidden,
                modulation_net_num_layers,
            )
        else:
            self.modulation_net = Bias(num_modulations)

        if not use_latent:
            if self.modulate_shift and self.modulate_scale:
                self.modulation_net.bias.data = torch.cat(
                    (
                        torch.ones(num_modulations // 2),
                        torch.zeros(num_modulations // 2),
                    ),
                    dim=0,
                )
            elif self.modulate_scale:
                self.modulation_net.bias.data = torch.ones(num_modulations)
            else:
                self.modulation_net.bias.data = torch.zeros(num_modulations)

        self.num_modulations = num_modulations

    def modulate_forward(self, x, latent):
        '''
        Forward pass of the model
        Args:
            x (torch.Tensor): Shape (batch_size, *, dim_in), where * refers to
                any spatial dimensions, e.g. (height, width), (height * width,)
                or (depth, height, width) etc.
            latent (torch.Tensor): Shape (batch_size, latent_dim). If
                use_latent=False, then latent_dim = num_modulations.

        Returns:
            Output features of shape (batch_size, *, dim_out).
        '''

        x_shape = x.shape[:-1]
        x = x.view(x.shape[0], -1, x.shape[-1])
        modulations = self.modulation_net(latent)
        mid_idx = (
            self.num_modulations // 2
            if (self.modulate_scale and self.modulate_shift)
            else 0
        )
        idx = 0
        
        for j in range(self.depth):

            if self.ResNet:
                if 0 < j < self.depth-1:
                    x_id = x + 0

            x = self.fcs[2*j](x)

            x = torch.sin(x)
            # x = SinTu()(x)
            # x = self.activations[j](x)

            x = self.fcs[2*j+1](x) 

            if self.ResNet:
                if 0 < j < self.depth-1:
                    if x.shape[1] <= x_id.shape[1]:
                        x = x + x_id[:,:x.shape[1]]
                    else:
                        x0 = torch.zeros_like(x)
                        x0[:,:x_id.shape[1]] = x_id
                        x = x + x0

            if j < self.depth - 1:
                if self.modulate_scale:
                    # Shape (batch_size, 1, rank). Note that we add 1 so
                    # modulations remain zero centered
                    scale = modulations[:, idx: idx +
                                        self.rank].unsqueeze(1) + 1.0
                else:
                    scale = 1.0

                if self.modulate_shift:
                    # Shape (batch_size, 1, rank)
                    shift = modulations[
                        :, mid_idx + idx: mid_idx + idx + self.rank
                    ].unsqueeze(1)
                else:
                    shift = 0.0

                x = x * scale + shift

        return x.view(*x_shape, x.shape[-1])

class CSTRIDE(torch.nn.Module):

    def __init__(self, input_size, output_size, hidden_size = 64, hidden_layers = 2, coord_size = 1, fmmnn_depth = 6, fmmnn_width = 256, fmmnn_rank = 16, hypernet_depth = 1, hypernet_width = 128, device = 'cpu', num_fourier = 0):
        '''
        C-STRIDE model definition


        Inputs
            input size (e.g. number of sensors)
            output size (e.g. number of variables)
            size of LSTM hidden layers (default to 64)
            number of LSTM hidden layers (default to 2)
            size of the coordinates (default to 1)
            depth of the FMMNN (default to 6)
            width of the FMMNN (default to 256)
            rank of the FMMNN (default to 16)
            depth of the hypernet (default to 1)
            width of the hypernet (default to 128)
            number of Fourier frequencies (default to 0)
        '''

        super(CSTRIDE,self).__init__()

        self.lstm = torch.nn.LSTM(input_size = input_size,
                                  hidden_size = hidden_size,
                                  num_layers = hidden_layers,
                                  batch_first=True)
        
        if num_fourier > 0:
            self.fourier_embedding = FourierEmbedding(num_fourier, coord_size)

        self.inr = ModulatedFMMNN(
            dim_in=coord_size + num_fourier * 2,
            dim_out=output_size,
            rank=fmmnn_rank,
            width=fmmnn_width,
            num_layers=fmmnn_depth,
            device=device,
            use_latent=True,
            latent_dim=hidden_size,
            modulation_net_dim_hidden=hypernet_width,
            modulation_net_num_layers=hypernet_depth,
        )

        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size
        self.num_fourier = num_fourier
        self.device = torch.device(device)

    def decode(self, coords, latent):
        self.inr.to(device=self.device)
        if self.num_fourier > 0:
            coords = self.fourier_embedding(coords)
        output = self.inr.modulate_forward(coords, latent)
        return output

    def forward(self, x, coords, return_hidden_sequence=False):
        '''
        Forward pass of the model
        Inputs
            x: observations (e.g. sensors) with shape (batch_size, lag, nsensors)
            coords: spatial coordinates with shape (batch_size, *, coord_size)
        Outputs
            output: output data (e.g. variables) with shape (batch_size, *, output_size)
        '''
        
        h_0 = torch.zeros((self.hidden_layers, x.size(0), self.hidden_size), dtype=torch.float)
        c_0 = torch.zeros((self.hidden_layers, x.size(0), self.hidden_size), dtype=torch.float)
        if self.device.type == 'cuda':
            h_0 = h_0.cuda(device=self.device)
            c_0 = c_0.cuda(device=self.device)

        hidden_sequence, (latent, _) = self.lstm(x, (h_0, c_0))
        latent = latent[-1].view(-1, self.hidden_size)

        self.inr.to(device=self.device)
        if self.num_fourier > 0:
            coords = self.fourier_embedding(coords)
        output = self.inr.modulate_forward(coords, latent)

        if return_hidden_sequence:
            return output, latent, hidden_sequence
        return output, latent

class CSTRIDEForecast(CSTRIDE):
    """
    Next-step forecasting variant of C-STRIDE.

    The architecture matches C-STRIDE; autoregressive behavior is
    defined by the forecast training and evaluation loops.
    """

    pass
