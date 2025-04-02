import os
import torch
from torchvision import transforms
from PIL import Image

image_dir = '/Users/karim/desktop/eece499/TCN_SINDy/processed_frames'

transform = transforms.Compose([
    transforms.Grayscale(),  
    transforms.ToTensor()    
])

def load_images_to_tensors(image_dir, transform):
    image_tensors = []
    files = sorted(os.listdir(image_dir), key=lambda x: int(os.path.splitext(x)[0]))
    for filename in files:
        if filename.endswith(".png"):
            image_path = os.path.join(image_dir, filename)
            image = Image.open(image_path)
            image_tensor = transform(image)
            image_tensors.append(image_tensor)
    return torch.stack(image_tensors)  #To have them in batch format all the frames are in one batch

images_tensor = load_images_to_tensors(image_dir, transform)
print(images_tensor.shape)
#Shape is 1114,1,556,200    (1114 frames, 1 channel, 556 Height, 200 Width)

tensor_file_path = '/Users/karim/desktop/eece499/TCN_SINDy/image_tensors.pt'
torch.save(images_tensor, tensor_file_path)
#Saved results in image_tensors