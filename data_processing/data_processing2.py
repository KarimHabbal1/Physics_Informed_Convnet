import torch
import torch.nn.functional as F
import cv2
import numpy as np

def pad_to_target(img, target_height, target_width):
    current_height, current_width = img.shape[-2:]
    pad_height = (target_height - current_height) if current_height < target_height else 0
    pad_width = (target_width - current_width) if current_width < target_width else 0

    # Calculate padding to be added to each side
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    # Apply symmetric padding
    padded_img = F.pad(img, (pad_left, pad_right, pad_top, pad_bottom), 'constant', 0)
    return padded_img

# Load video and process frames directly to tensor with padding
cap = cv2.VideoCapture('/Users/karim/desktop/eece499/spring-osc-trim.mov')
frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tensor_frame = torch.from_numpy(gray_frame).unsqueeze(0).float() / 255.0  # Convert to tensor and normalize
    padded_frame = pad_to_target(tensor_frame, 200, 560)  # Apply padding
    frames.append(padded_frame)
cap.release()

# Stack all frames into a single tensor
images_tensor = torch.stack(frames)

print("Expected Shape: [num_frames, 1, 200, 560]")
print(f"Actual Shape: {images_tensor.shape}")