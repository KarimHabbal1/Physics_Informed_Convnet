import torch
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

        # Calculate output size after CNN
        self._get_conv_output(input_shape)

        self.encoder_fc_layers = nn.Sequential(
            nn.Linear(self.num_flat_features, 128),
            nn.ELU(),
            nn.Linear(128, latent_dim),  # Latent space (Koopman eigenfunctions)
        )

        #Koopman operator K
        self.K = nn.Linear(latent_dim, latent_dim, bias=False)  # Koopman matrix

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
        bs = 1
        input_tensor = torch.randn(bs, *shape)
        with torch.no_grad():
            output_feat = self.encoder_conv_layers(input_tensor)
        self.final_conv_shape = output_feat.shape[1:]
        self.num_flat_features = output_feat.numel() // bs

    def forward(self, x):
        #Encode input
        x = self.encoder_conv_layers(x)
        x = x.view(x.size(0), -1)
        latent_vars = self.encoder_fc_layers(x)

        #Koopman evolution (Linear mapping in latent space)
        latent_next = self.K(latent_vars)

        #Decode from latent representation
        x = self.decoder_fc_layers(latent_vars)
        x = x.view(x.size(0), *self.final_conv_shape)
        x = self.decoder_conv_layers(x)
        
        return x, latent_vars, latent_next

def koopman_loss(reconstructed, latent_vars, latent_next, images_tensor, loss_fn):
    #Reconstruction loss (MSE between input and output)
    recon_loss = loss_fn(reconstructed, images_tensor)

    #Koopman loss (Kφ(x) ≈ φ(x_next))
    koopman_loss = torch.mean((latent_next - latent_vars) ** 2)  # Mean Squared Error

    #Total loss: weighted sum
    total_loss = recon_loss + 0.1 * koopman_loss
    return total_loss

# Initialize model
model = KoopmanAutoencoder(input_shape=(1, 556, 200))

# Define loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
images_tensor=images_tensor = torch.load('/Users/karim/desktop/eece499/TCN_SINDy/image_tensors.pt')

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    reconstructed, latent_vars, latent_next = model(images_tensor)

    reconstructed = reconstructed[:, :, :images_tensor.shape[2], :images_tensor.shape[3]]

    # Compute loss
    loss = koopman_loss(reconstructed, latent_vars, latent_next, images_tensor, loss_fn)

    # Backpropagation
    loss.backward()
    optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Total Loss: {loss.item():.4f}")



model.eval()
with torch.no_grad():
    _, latent_vars, _ = model(images_tensor)

