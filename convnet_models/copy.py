import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

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
        # Output Dimension = ((Input dimension +2*padding - kernel_size)/stride) + 1
        # stride is how much we shift after each filter 
        # padding is adding zeros around the edges of your matrix before applying the filter
        # Final dimension is (64,69,25)

        #Dynamically calculate the output shape of the encoder conv layers
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
        bs = 1  # Batch size of 1 for testing output shape
        input_tensor = torch.randn(bs, *shape)  # Generate a random tensor without Variable
        with torch.no_grad():  # Disable gradient tracking since this is only for shape calculation
            output_feat = self.encoder_conv_layers(input_tensor)
        self.final_conv_shape = output_feat.shape[1:]  # Store the shape after the last conv layer
        n_size = output_feat.detach().flatten(start_dim=1).size(1)  # Compute flattened size safely
        self.num_flat_features = n_size

    def forward(self, x):
        x = self.encoder_conv_layers(x)
        x = x.view(x.size(0), -1) #Flattening
        x = self.encoder_fc_layers(x)
        latent_vars = x.clone()  # Store latent variables before decoding
        x = self.decoder_fc_layers(x)
        #Use stored shape for dynamic reshaping
        x = x.view(x.size(0), *self.final_conv_shape)
        x = self.decoder_conv_layers(x)
        return x,latent_vars


images_tensor = torch.load('/Users/karim/desktop/eece499/TCN_SINDy/image_tensors.pt')
images_tensor = F.pad(images_tensor, (0, 0, 2, 2))  # (left, right, top, bottom)


model = ConvNetAutoencoder_basic(input_shape=(1, 560, 200))
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 1
losses = []
patience = 1
min_delta = 0.01  
latent_epoch_data = []
converged = False

for epoch in range(num_epochs):
    if converged:
        break
    model.train()
    optimizer.zero_grad()
    reconstructed, latent_vars = model(images_tensor)
    loss = loss_fn(reconstructed, images_tensor)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    print(f"Epoch: {epoch}, Loss: {loss.item():.4f}")

    if epoch > 0 and (losses[-2] - losses[-1] < min_delta):
        if patience > 0:
            patience -= 1
        else:
            print("Early stopping as the model has converged.")
            converged=True

    if epoch % 50 == 0:
        latent_epoch_data.append(latent_vars.detach().cpu().numpy())



latent_values = []

model.eval()
# Forward pass to get the latent variable
with torch.no_grad():
    xe = model.encoder_conv_layers(images_tensor)
    xe = xe.view(xe.size(0), -1)
    latent_variable = model.encoder_fc_layers(xe)

# Convert the lists to numpy arrays for plotting
latent_values = latent_variable.numpy() 


plt.plot(losses, label='Loss per Epoch')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss as a Function of Epochs')
plt.legend()
plt.show()


for index, latent_vars_at_epoch in enumerate(latent_epoch_data):
    plt.figure()
    for var_idx in range(latent_vars_at_epoch.shape[1]):  
        plt.plot(latent_vars_at_epoch[:, var_idx], label=f'Latent Variable {var_idx + 1}')
    plt.xlabel('Frame')
    plt.ylabel('Latent Variable')
    plt.title(f'Latent Variable as a Function of Frames at Epoch {index * 50}')
    plt.legend()
    plt.show()



# Plot the latent variable as a function of epochs
plt.plot(latent_values)
plt.xlabel('Frame')
plt.ylabel('Latent Variable')
plt.title('Latent Variable as a Function of Frames After optimization')
plt.show()

