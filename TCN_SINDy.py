import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torchdiffeq import odeint as odeint_torch 


class ConvNetAutoencoderSINDy(nn.Module):
    def __init__(self, input_shape, latent_dim=2, poly_order=2):
        super(ConvNetAutoencoderSINDy, self).__init__()
        self.encoder_conv_layer = nn.Sequential(
            nn.Conv2d(1,16,kernel_size=3,stride=2,padding=1),
            nn.ReLU(),
            nn.Conv2d(16,32,kernel_size=3,stride=2,padding=1),
            nn.ReLU(),
            nn.Conv2d(32,64,kernel_size=3,stride=2,padding=1),
            nn.ReLU()
        )

        self.get_conv_output(input_shape)

        self.fully_connected_encoder=nn.Sequential(
            nn.Linear(self.num_flat_features,128),
            nn.ELU(),
            nn.Linear(128,2)
        )

        #NOW I WANT TO USE SINDY ON THESE TWO LATENT VARIABLES
        #I NEED TO BE ABLE TO GET THE FLATTENED TENSOR AND TAKE IT AS MY X

        self.fully_connected_decoder=nn.Sequential(
            nn.Linear(2,128),
            nn.ELU(),
            nn.Linear(128,self.num_flat_features)
        )

        self.decoder_conv_layer = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
        )


    def get_conv_output(self,input_shape):
        input_tensor=torch.randn(1,*input_shape)
        with torch.no_grad():
            output_feat = self.encoder_conv_layers(input_tensor)
        self.final_conv_shape=output_feat.shape[1:]
        n_size=output_feat.detach().flatten(start_dim=1).size(1)
        self.num_flat_features = n_size

    def conv_decode(self, v):
        decoded = self.decoder_conv_layer(v)
        return decoded

    def conv_encode(self, v):
        encoded = self.encoder_conv_layer(v)
        return encoded
    
    def fc_decode(self, x):
        decoded = self.fully_connected_decoder(x)
        return decoded

    def fc_encode(self, x):
        encoded = self.fully_connected_encoder(x)
        return encoded

    def phi(self, x): 

        library = [torch.ones(x.size(0), 1)] 
        for i in range(self.latent_dim):
            library.append(x[:, i].unsqueeze(1)) 
        
        if self.poly_order == 2:
            for i in range(self.latent_dim):
                for j in range(i, self.latent_dim):
                    library.append((x[:, i] * x[:, j]).unsqueeze(1)) 
        
        for i in range(self.latent_dim):
            library.append(torch.sin(x[:, i]).unsqueeze(1))

        return torch.cat(library, dim=1)

    def phi_t(self, x, t): #For t i will use either index of frame or timestamp code for timestamp is in video processing
        library = [torch.ones(x.size(0), 1)]  # Constant term
        
        # Add linear terms
        for i in range(self.latent_dim):
            library.append(x[:, i].unsqueeze(1))
        
        # Add polynomial terms (second order)
        if self.poly_order >= 2:
            for i in range(self.latent_dim):
                for j in range(i, self.latent_dim):
                    library.append((x[:, i] * x[:, j]).unsqueeze(1))
        
        # Add sine terms only for each variable, without interactions
        for i in range(self.latent_dim):
            library.append(torch.sin(x[:, i]).unsqueeze(1))
        
        # Include time-dependent terms if t is not None
        if t is not None:
            t = t.view(-1, 1)  # Ensure t is a column vector
            # Time-dependent terms for each entry in the library
            library_with_t = [entry * t for entry in library]
            library_with_t2 = [entry * torch.pow(t, 2) for entry in library]
            library.extend(library_with_t)
            library.extend(library_with_t2)

        return torch.cat(library, dim=1) 
    
    def SINDy_num(self, x): #phi(x)coeff
        dxdt = torch.matmul(self.phi(x), self.coefficients)
        return dxdt
    
    def SINDy_num_t(self, t, x):
        dxdt = torch.matmul(self.phi_t(x,t), self.coefficients)
        return dxdt
    
    def integrate(self, x0, t):
        try:
            x_pred = odeint_torch(self.SINDy_num, x0, t) 
        except AssertionError as error:
            print(error)
            return None 
        return x_pred
    
    def forward(self, x):
        encoded = self.encoder_conv_layer(x)  # Apply convolutional layers
        encoded = encoded.view(encoded.size(0), -1)  # Flatten the features for the fully connected layer
        latent = self.fully_connected_encoder(encoded)  # Get the latent space representation

        # Decoding the latent representation
        decoded = self.fully_connected_decoder(latent)
        decoded = decoded.view(decoded.size(0), 64, *self.final_conv_shape[1:])  # Reshape to match the input of the decoder conv layers
        decoded = self.decoder_conv_layers(decoded)  # Apply decoder convolutional layers

        return decoded, latent



    











'''
images_tensor = torch.load('/Users/karim/desktop/eece499/TCN_SINDy/image_tensors.pt')
autoencoder = ConvNetAutoencoder(input_shape=(1, 556, 200))
coefs = torch.zeros(6, 2, requires_grad=True, dtype=torch.float32)
sindy_model = SINDy(dt=0.01, t_int=10, tau=5, coefs=coefs)
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
'''