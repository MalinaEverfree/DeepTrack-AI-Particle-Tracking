import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import ParticleTrackerCNN
from dataset import ParticleDataset

def quick_eval(model, pipeline, n=50):
    model.eval()
    images, true_labels = [], []
    with torch.no_grad():
        for _ in range(n):
            image, label = pipeline.update().resolve()
            image = np.array(image).astype("float32")
            image = image / (image.max() + 1e-8)
            image = np.transpose(image, (2, 0, 1))
            images.append(image)
            true_labels.append(label)
        images_t = torch.from_numpy(np.stack(images))
        preds = model(images_t).numpy()
    true_labels = np.stack(true_labels)
    err_px = np.abs(preds[:, :2] - true_labels[:, :2]) * 64
    model.train()   # switch back to training mode!
    return err_px.mean()

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Training on:", device)

dataset = ParticleDataset(length=3200)          # 3200 fresh images "per epoch"
loader = DataLoader(dataset, batch_size=32, shuffle=False)
# shuffle=False is fine here -- every sample is already random by construction,
# there's no fixed order to shuffle.

model = ParticleTrackerCNN().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# --- Training loop ---
num_epochs = 120
for epoch in range(num_epochs):
    running_loss = 0.0
    loss_history = []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()              # clear old gradients from last batch
        predictions = model(images)        # forward pass
        loss = criterion(predictions, labels)
        loss.backward()                    # compute gradients (backpropagation)
        optimizer.step()                   # nudge the weights

        running_loss += loss.item()
        loss_history.append(loss.item())
    avg_loss = running_loss / len(loader)
    loss_history.append(avg_loss)
    print(f"Epoch {epoch+1}/{num_epochs} - loss: {avg_loss:.5f}")


torch.save(model.state_dict(), "tracker_model.pt")
print("Model saved to tracker_model.pt")

import json

with open("training_log.json", "w") as f:
    json.dump(loss_history, f)
