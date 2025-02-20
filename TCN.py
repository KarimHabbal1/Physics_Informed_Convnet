import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNetAutoencoder(nn.Module):
    def __init__(self, input_shape, conv_layers, fc_layers, latent_dims=(2,)):
        super(ConvNetAutoencoder, self).__init__()
        self.input_shape = input_shape
        self.latent_dims = latent_dims
        
        # Define encoder convolutional layers
        self.encoder_conv_layers = self.create_conv_layers(1, conv_layers, batch_norm=True)
        
        # Calculate shape of the output of the last conv layer to correctly pass it into the first FC layer
        conv_output_shape = self.calculate_conv_output_shape(input_shape, conv_layers)
        
        # Define encoder and decoder fully connected layers
        self.encoder_fc_layers = self.create_fc_layers([conv_output_shape] + fc_layers + [np.prod(latent_dims)])
        self.decoder_fc_layers = self.create_fc_layers([np.prod(latent_dims)] + list(reversed(fc_layers)) + [conv_output_shape])
        
        # Define decoder convolutional layers (mirror of the encoder but with ConvTranspose layers)
        reversed_conv_layers = list(reversed(conv_layers))
        self.decoder_conv_layers = self.create_conv_layers(reversed_conv_layers[0][0], reversed_conv_layers, batch_norm=False, decode=True)
        
        # Final layer to produce one-channel output
        self.output_layer = nn.ConvTranspose2d(reversed_conv_layers[-1][0], 1, kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        x = x.view(x.size(0), 1, *self.input_shape)
        x = self.encoder_conv_layers(x)
        self.enc_conv_output_shape = x.shape[1:]  # Ignore the batch dimension
        x = x.view(x.size(0), -1)  # Flatten for the FC layers
        x = self.encoder_fc_layers(x)
        x = self.decoder_fc_layers(x)
        x = x.view(-1, *self.enc_conv_output_shape)
        x = self.decoder_conv_layers(x)
        x = self.output_layer(x)
        return x
    
    def create_conv_layers(self, in_channels, layer_specs, batch_norm=False, decode=False):
        layers = []
        for spec in layer_specs:
            out_channels, kernel_size, stride, padding = spec
            if decode:
                layers += [nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)]
            if batch_norm:
                layers += [nn.BatchNorm2d(out_channels)]
            layers += [nn.ReLU(inplace=True)]
            in_channels = out_channels
        return nn.Sequential(*layers)


    def create_fc_layers(self, layer_specs, final_act=False):
        layers = []
        for i in range(len(layer_specs) - 1):
            layers += [nn.Linear(layer_specs[i], layer_specs[i+1])]
            if i < len(layer_specs) - 2 or final_act:
                layers += [nn.ELU(inplace=True)]
        return nn.Sequential(*layers)
    
    def calculate_conv_output_shape(self, input_shape, conv_layers):
        output = torch.zeros(1, 1, *input_shape)
        for layer in self.encoder_conv_layers:
            if not isinstance(layer, nn.ReLU) and not isinstance(layer, nn.BatchNorm2d):
                output = layer(output)
        return int(np.prod(output.size()[1:]))





class ConvNetAutoencoder_basic(nn.Module):
    def __init__(self, input_shape):
        super(ConvNetAutoencoder_basic, self).__init__()
        self.encoder_conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
        )
        # Dynamically calculate the output shape of the encoder conv layers
        self._get_conv_output(input_shape)

        self.encoder_fc_layers = nn.Sequential(
            nn.Linear(self.num_flat_features, 128),
            nn.ELU(),
            nn.Linear(128, 2),  # Assuming a 2-dimensional latent space
        )

        self.decoder_fc_layers = nn.Sequential(
            nn.Linear(2, 128),
            nn.ELU(),
            nn.Linear(128, self.num_flat_features),
        )

        self.decoder_conv_layers = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
        )

    def _get_conv_output(self, shape):
        bs = 1  # Batch size of 1 for the test input
        input = torch.autograd.Variable(torch.rand(bs, *shape))
        output_feat = self.encoder_conv_layers(input)
        self.final_conv_shape = output_feat.shape[1:]  # Store the shape after the last conv layer
        n_size = output_feat.data.view(bs, -1).size(1)
        self.num_flat_features = n_size

    def forward(self, x):
        x = x.view(x.size(0), 1, x.size(1), x.size(2))
        x = self.encoder_conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.encoder_fc_layers(x)
        x = self.decoder_fc_layers(x)
        # Use stored shape for dynamic reshaping
        x = x.view(x.size(0), *self.final_conv_shape)
        x = self.decoder_conv_layers(x)
        return x




class ConvNetVAE(nn.Module):
    def __init__(self, input_shape, conv_layers, fc_layers, latent_dim=2):
        super(ConvNetVAE, self).__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        
        # Encoder convolutional layers
        self.encoder_conv_layers = self.create_conv_layers(1, conv_layers, batch_norm=True)
        
        # Calculate output shape for FC layers
        conv_output_shape = self.calculate_conv_output_shape(input_shape, conv_layers)
        
        # Encoder fully connected layers
        self.encoder_fc_layers = self.create_fc_layers([conv_output_shape] + fc_layers + [latent_dim * 2])

        # Decoder fully connected layers
        self.decoder_fc_layers = self.create_fc_layers([latent_dim] + list(reversed(fc_layers)) + [conv_output_shape])

        # Decoder convolutional layers
        reversed_conv_layers = list(reversed(conv_layers))
        self.decoder_conv_layers = self.create_conv_layers(reversed_conv_layers[0][0], reversed_conv_layers, batch_norm=False, decode=True)
        
        # Final output layer
        self.output_layer = nn.ConvTranspose2d(reversed_conv_layers[-1][0], 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = x.view(x.size(0), 1, *self.input_shape)
        x = self.encoder_conv_layers(x)
        self.enc_conv_output_shape = x.shape[1:]
        x = x.view(x.size(0), -1)
        x = self.encoder_fc_layers(x)
        
        # Split into mean and log-variance vectors
        mu, log_var = torch.split(x, self.latent_dim, dim=1)
        
        # Sample z using the reparameterization trick
        z = self.reparameterize(mu, log_var)
        
        # Decode
        x = self.decoder_fc_layers(z)
        x = x.view(-1, *self.enc_conv_output_shape)
        x = self.decoder_conv_layers(x)
        x = self.output_layer(x)
        return x, mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def create_conv_layers(self, in_channels, layer_specs, batch_norm=False, decode=False):
        layers = []
        for spec in layer_specs:
            out_channels, kernel_size, stride, padding = spec
            if decode:
                layers += [nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)]
            if batch_norm:
                layers += [nn.BatchNorm2d(out_channels)]
            layers += [nn.ReLU(inplace=True)]
            in_channels = out_channels
        return nn.Sequential(*layers)

    def create_fc_layers(self, layer_specs):
        layers = []
        for i in range(len(layer_specs) - 1):
            layers += [nn.Linear(layer_specs[i], layer_specs[i + 1])]
            layers += [nn.ELU(inplace=True)]  # Optional: you might want to control this
        return nn.Sequential(*layers)

    def calculate_conv_output_shape(self, input_shape, conv_layers):
        output = torch.zeros(1, 1, *input_shape)
        for layer in self.encoder_conv_layers:
            if not isinstance(layer, nn.ReLU) and not isinstance(layer, nn.BatchNorm2d):
                output = layer(output)
        return int(np.prod(output.size()[1:]))




def pad_to_target(img, target_height, target_width):
    """
    Pad an image tensor to the target height and width with zeros.
    
    Args:
    - img (Tensor): The input image tensor of shape [H, W].
    - target_height (int): The target height.
    - target_width (int): The target width.
    
    Returns:
    - Tensor: Padded image tensor.
    """
    height, width = img.shape
    pad_height = target_height - height
    pad_width = target_width - width
    
    # Ensure padding is non-negative
    pad_height = max(pad_height, 0)
    pad_width = max(pad_width, 0)
    
    # Calculate padding for top/bottom and left/right
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    
    # Pad the image
    # pad() takes a tuple (pad_left, pad_right, pad_top, pad_bottom)
    padded_img = F.pad(img, (pad_left, pad_right, pad_top, pad_bottom), 'constant', 0)
    return padded_img