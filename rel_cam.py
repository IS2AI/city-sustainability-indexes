import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import models, transforms
import torch.nn.functional as F

# -----------------------
# 1. Setup
# -----------------------
device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")

# Load your trained model
model = models.resnet50(pretrained=False)
in_features = model.fc.in_features
model.fc = torch.nn.Linear(in_features, 1)

# Load checkpoint (handles DataParallel or plain)
checkpoint = torch.load("workspace/satellite_images/models/index_prediction/overall_arcadis/best.pth", map_location=device)
if hasattr(checkpoint, "state_dict"):
    checkpoint = checkpoint.state_dict()
new_state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
model.load_state_dict(new_state_dict, strict=False)
model.to(device)
model.eval()

# -----------------------
# 2. Hooks for Relevance-CAM
# -----------------------
target_layer = model.layer4[-1]
activations = []
gradients = []

def forward_hook(module, input, output):
    activations.append(output)

def backward_hook(module, grad_input, grad_output):
    gradients.append(grad_output[0])

target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)

# -----------------------
# 3. Transforms & paths
# -----------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

root_dir = "workspace/dataset/reg_240/test_b"
output_dir = "relevance_cam_outputs_overall"
os.makedirs(output_dir, exist_ok=True)

# Collect all image paths
image_paths = []
for dirpath, _, filenames in os.walk(root_dir):
    for filename in filenames:
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
            image_paths.append(os.path.join(dirpath, filename))

# -----------------------
# 4. Process in batches
# -----------------------
batch_size = 16  # Adjust to fit GPU memory

for i in range(0, len(image_paths), batch_size):
    batch_paths = image_paths[i:i + batch_size]
    original_images = [Image.open(p).convert("RGB") for p in batch_paths]
    batch_tensors = torch.stack([transform(img) for img in original_images]).to(device)

    # Clear previous hook data
    activations.clear()
    gradients.clear()

    # Forward
    outputs = model(batch_tensors)

    # Backward for regression
    model.zero_grad()
    grads = torch.ones_like(outputs)
    outputs.backward(grads)

    # -----------------------
    # 5. Relevance-CAM computation
    # -----------------------
    acts = activations[0]      # [B, C, H, W]
    grads_val = gradients[0]   # [B, C, H, W]

    # Element-wise product of activations and gradients
    numerator = torch.sum(acts * grads_val, dim=(2, 3), keepdim=True)
    denominator = torch.sum(grads_val, dim=(2, 3), keepdim=True) + 1e-8
    weights = numerator / denominator

    # Weighted sum of activations
    cams = torch.relu(torch.sum(weights * acts, dim=1, keepdim=True))  # [B, 1, H, W]

    # Normalize CAMs per image
    cams_min = cams.view(cams.size(0), -1).min(dim=1)[0].view(-1, 1, 1, 1)
    cams_max = cams.view(cams.size(0), -1).max(dim=1)[0].view(-1, 1, 1, 1)
    cams = (cams - cams_min) / (cams_max - cams_min + 1e-8)

    # -----------------------
    # 6. Save results
    # -----------------------
    for j, cam in enumerate(cams):
        cam_up = F.interpolate(
            cam.unsqueeze(0),
            size=(original_images[j].height, original_images[j].width),
            mode='bilinear',
            align_corners=False
        ).squeeze().detach().cpu().numpy()

        # Masked image (threshold 0.5)
        mask = np.uint8(cam_up > 0.5)
        masked_image = np.array(original_images[j]) * mask[:, :, None]

        # Plot
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(original_images[j])
        axs[0].set_title("Original Image")
        axs[0].axis('off')

        axs[1].imshow(original_images[j])
        axs[1].imshow(cam_up, cmap='jet', alpha=0.5)
        axs[1].set_title("Relevance-CAM Overlay")
        axs[1].axis('off')

        axs[2].imshow(masked_image)
        axs[2].set_title("Masked Image")
        axs[2].axis('off')

        plt.tight_layout()
        save_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(batch_paths[j]))[0]}_relcam.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

