import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class KoopmanAutoencoder(nn.Module):
    def __init__(self, input_shape, latent_dim=2):
        super(KoopmanAutoencoder, self).__init__()
        
        # Encoder (CNN + Fully Connected Layers)
        self.encoder_conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
        )

        #Calculate output size after CNN
        self._get_conv_output(input_shape)

        self.encoder_fc_layers = nn.Sequential(
            nn.Linear(self.num_flat_features, 128),
            nn.ELU(),
            nn.Linear(128, latent_dim),  #Latent space (Koopman eigenfunctions)
        )

        #Koopman operator K
        self.K = nn.Linear(latent_dim, latent_dim, bias=False)  #Koopman matrix

        #Decoder (Fully Connected + CNN)
        self.decoder_fc_layers = nn.Sequential(
            nn.Linear(latent_dim, 128),
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
        input_tensor = torch.randn(1, *shape)  # Generate a random tensor without Variable
        with torch.no_grad():  # Disable gradient tracking since this is only for shape calculation
            output_feat = self.encoder_conv_layers(input_tensor)
        self.final_conv_shape = output_feat.shape[1:]  # Store the shape after the last conv layer
        n_size = output_feat.detach().flatten(start_dim=1).size(1)  # Compute flattened size safely
        self.num_flat_features = n_size

    def forward(self, x, apply_koopman=False, m=0):
        x = self.encoder_conv_layers(x)
        x = x.view(x.size(0), -1) #Flattening
        x = self.encoder_fc_layers(x)
        latent_vars = x.clone()  # Store latent variables before decoding
        
        x = self.decoder_fc_layers(x)
        #Use stored shape for dynamic reshaping
        x = x.view(x.size(0), *self.final_conv_shape)
        x = self.decoder_conv_layers(x)

        #APPLYING KOOPMAN IF SPECIFIED TO DO SO
        if apply_koopman:
            
            latent_vars_at_m = latent_vars
            for i in range(m):
                latent_vars_at_m = self.K(latent_vars_at_m)
            
            decoded_k = self.decoder_fc_layers(latent_vars_at_m)
            decoded_k = decoded_k.view(decoded_k.size(0), *self.final_conv_shape)
            decoded_k = self.decoder_conv_layers(decoded_k)

            return decoded_k,latent_vars_at_m #x and latent vars without K while latent_vars_at_m and decoded_k with K
        
        return x,latent_vars
    

# Initialize model
model = KoopmanAutoencoder(input_shape=(1, 556, 200))

# Define loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
images_tensor=images_tensor = torch.load('/Users/karim/desktop/eece499/TCN_SINDy/image_tensors.pt')


#X_current and X_future assuming m = 1 for simplicity
X_current = images_tensor[:-1]  #all but the last
X_future = images_tensor[1:]    #all but the first

num_epochs_phase1 = 2
num_epochs_phase2 = 2


def train_autoencoder():
    model.train()
    for epoch in range(num_epochs_phase1):
        optimizer.zero_grad()
        
        reconstructed, latent_vars = model(images_tensor)
        reconstructed = reconstructed[:, :, :images_tensor.shape[2], :images_tensor.shape[3]]
        
        loss = loss_fn(reconstructed, images_tensor)
        loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch}, Loss: {loss.item():.4f}")


def train_koopman():
    model.train()
    for epoch in range(num_epochs_phase2):
        for i in range(len(X_current)):
            optimizer.zero_grad()

            x_at_i_plus_m, latent_at_i_plus_m = model(X_future[i].unsqueeze(0))  
            
            reconstructed_at_i_plus_m_with_K, latent_at_i_plus_m_with_K = model(X_current[i].unsqueeze(0),True,1)
            reconstructed_at_i_plus_m_with_K = reconstructed_at_i_plus_m_with_K[:, :, :images_tensor.shape[2], :images_tensor.shape[3]]
            
            loss = loss_fn(latent_at_i_plus_m,latent_at_i_plus_m_with_K) + loss_fn(X_future[i].unsqueeze(0),reconstructed_at_i_plus_m_with_K)
            loss.backward()
            optimizer.step()

    print(f"Epoch: {epoch}, Loss: {loss.item():.4f}")


train_autoencoder()
train_koopman()










