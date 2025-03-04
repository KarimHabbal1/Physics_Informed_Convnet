import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torchdiffeq import odeint as odeint_torch


def jacobian(y, x, create_graph=False, retain_graph=True, allow_unused=False):
    jac = []
    for i in range(y.shape[-1]):
        jac_i = torch.autograd.grad(
            y[..., i], x, torch.ones_like(y[..., i]), allow_unused=allow_unused, retain_graph=retain_graph, create_graph=create_graph)[0]
        jac.append(jac_i)
    return torch.stack(jac, dim=-1)

class ConvNetAutoencoder(nn.Module):
    def __init__(self, input_shape):
        super(ConvNetAutoencoder, self).__init__()
        self.encoder_conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
        )
        
        # Compute conv output shape
        self._get_conv_output(input_shape)

        self.encoder_fc_layers = nn.Sequential(
            nn.Linear(self.num_flat_features, 128),
            nn.ELU(),
            nn.Linear(128, 2),  # 2D latent space
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
        bs = 1  # Batch size for testing shape
        input_tensor = torch.randn(bs, *shape)
        with torch.no_grad():
            output_feat = self.encoder_conv_layers(input_tensor)
        self.final_conv_shape = output_feat.shape[1:]
        self.num_flat_features = output_feat.flatten(start_dim=1).size(1)

    def forward(self, x):
        x = self.encoder_conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        latent_vars = self.encoder_fc_layers(x)
        x = self.decoder_fc_layers(latent_vars)
        x = x.view(x.size(0), *self.final_conv_shape)
        x = self.decoder_conv_layers(x)
        return x, latent_vars


class SAE(nn.Module):
    def __init__(self, dt, t_int, tau, coefs, input_dim=2, latent_dim=2, library_dim=6, poly_order=2, l=None):
        super(SAE, self).__init__()
        self.dt = dt
        self.t_int = t_int
        self.tau = tau
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.library_dim = library_dim
        self.poly_order = poly_order
        self.l = l if l else {'l1': 1e-1, 'l2': 1e-1, 'l3': 1e-1, 'l4': 1e-1, 'l5': 1e-1, 'l6': 1e-1}
        self.coefficients = nn.Parameter(coefs)

    def phi(self, x): 
        library = [torch.ones(x.size(0), 1)]
        for i in range(self.latent_dim):
            library.append(x[:, i].unsqueeze(1)) 
        
        if self.poly_order == 2:
            for i in range(self.latent_dim):
                for j in range(i, self.latent_dim):
                    library.append((x[:, i] * x[:, j]).unsqueeze(1)) 

        return torch.cat(library, dim=1)

    def SINDy_num(self, time, x):
        return torch.matmul(self.phi(x), self.coefficients)

    def integrate(self, x0, t):
        return odeint_torch(self.SINDy_num, x0, t)

    def loss(self, latent_vars, dvdt, criterion):
        dxdt_SINDy = self.SINDy_num(torch.tensor(np.linspace(0, self.dt*len(latent_vars), len(latent_vars))), latent_vars)
        loss = criterion(dvdt, dxdt_SINDy) * self.l['l3']
        loss += torch.norm(self.coefficients, 1) * self.l['l5']
        return loss


images_tensor = torch.load('/Users/karim/desktop/eece499/TCN_SINDy/image_tensors.pt')

autoencoder = ConvNetAutoencoder(input_shape=(1, 556, 200))
coefs = torch.zeros(6, 2, requires_grad=True, dtype=torch.float32)
sindy_model = SAE(dt=0.01, t_int=10, tau=5, coefs=coefs)
# Define loss function and optimizers
recon_loss_fn = nn.MSELoss()
optimizer = optim.Adam(list(autoencoder.parameters()) + list(sindy_model.parameters()), lr=0.001)

num_epochs = 20

for epoch in range(num_epochs):
    autoencoder.train()
    sindy_model.train()
    optimizer.zero_grad()

    # Forward pass
    reconstructed, latent_vars = autoencoder(images_tensor)

    # Fix size mismatch
    reconstructed = reconstructed[:, :, :images_tensor.shape[2], :images_tensor.shape[3]]

    # Compute losses
    recon_loss = recon_loss_fn(reconstructed, images_tensor)
    
    # Compute numerical derivative of latent variables (finite differences)
    dvdt = (latent_vars[1:] - latent_vars[:-1]) / 0.01  # Assuming dt=0.01

    sindy_loss = sindy_model.loss(latent_vars[:-1], dvdt, recon_loss_fn)

    total_loss = recon_loss + sindy_loss

    # Backpropagation
    total_loss.backward()
    optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Recon Loss: {recon_loss.item():.4f}, SINDy Loss: {sindy_loss.item():.4f}")

# Evaluation
autoencoder.eval()
sindy_model.eval()

with torch.no_grad():
    reconstructed, latent_vars = autoencoder(images_tensor)
    dvdt = (latent_vars[1:] - latent_vars[:-1]) / 0.01
    sindy_pred = sindy_model.SINDy_num(torch.tensor(np.linspace(0, 0.01 * len(latent_vars), len(latent_vars))), latent_vars)

    print(f"Final SINDy Prediction: {sindy_pred}")
