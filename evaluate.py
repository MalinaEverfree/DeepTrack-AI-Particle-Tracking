import numpy as np
import torch

from model import ParticleTrackerCNN
from simulation import build_pipeline

# --- Load the trained model ---
model = ParticleTrackerCNN()
model.load_state_dict(torch.load("tracker_model.pt"))
model.eval()   # tell PyTorch we're evaluating, not training

# --- Generate fresh test particles the model has never seen ---
pipeline = build_pipeline()

n_samples = 200
images = []
true_labels = []

for _ in range(n_samples):
    image, label = pipeline.update().resolve()
    image = np.array(image).astype("float32")
    image = image / (image.max() + 1e-8)
    image = np.transpose(image, (2, 0, 1))
    images.append(image)
    true_labels.append(label)

images = torch.from_numpy(np.stack(images))
true_labels = np.stack(true_labels)

# --- Predict ---
with torch.no_grad():           # tells PyTorch not to bother tracking gradients here
    predictions = model(images).numpy()

# --- Convert normalized [0,1] error back into real pixels ---
frame_size = 64
errors_normalized = np.abs(predictions - true_labels)   # shape: (200, 4)

print(f"x error:      {errors_normalized[:, 0].mean() * frame_size:.3f} pixels")
print(f"y error:      {errors_normalized[:, 1].mean() * frame_size:.3f} pixels")
print(f"z error:      {errors_normalized[:, 2].mean():.4f}  (normalized 0-1 scale)")
print(f"radius error: {errors_normalized[:, 3].mean():.4f}  (normalized 0-1 scale)")

import matplotlib.pyplot as plt

# Grab 8 examples to look at
n_show = 8
fig, axes = plt.subplots(2, 4, figsize=(14, 7))

for ax, img_tensor, pred, true in zip(axes.ravel(), images[:n_show], predictions[:n_show], true_labels[:n_show]):
    img = img_tensor.numpy().squeeze()   # remove the "channel" dimension so matplotlib can show it as grayscale
    ax.imshow(img, cmap="gray")

    px, py = pred[0] * frame_size, pred[1] * frame_size
    tx, ty = true[0] * frame_size, true[1] * frame_size

    ax.scatter([tx], [ty], c="lime", marker="+", s=150, label="true")
    ax.scatter([px], [py], c="red", marker="x", s=100, label="predicted")
    ax.axis("off")

axes.ravel()[0].legend(loc="upper right", fontsize=8)
plt.tight_layout()
plt.savefig("evaluation_examples.png", dpi=150)
print("Saved evaluation_examples.png -- open it from the VS Code file explorer")
plt.show()
