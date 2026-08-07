import numpy as np
import torch
from torch.utils.data import Dataset

from simulation import build_pipeline


class ParticleDataset(Dataset):
    def __init__(self, length=3200):
        # `length` isn't a real dataset size -- it just tells PyTorch's
        # DataLoader how many samples make up "one epoch." Since our
        # simulator can generate forever, this number is really just
        # "how many freshly-simulated images per epoch."
        self.length = length
        self.pipeline = build_pipeline()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Note: idx is ignored on purpose -- every call generates a
        # BRAND NEW random particle, regardless of which index PyTorch asked for.
        image, label = self.pipeline.update().resolve()

        image = np.array(image).astype("float32")
        image = image / (image.max() + 1e-8)          # normalize brightness to [0,1]
        image = np.transpose(image, (2, 0, 1))          # HWC -> CHW (PyTorch's expected order)

        return torch.from_numpy(image), torch.from_numpy(label)
    
if __name__ == "__main__":
    ds = ParticleDataset(length=10)
    image, label = ds[0]
    print("Image tensor shape:", image.shape)   # should be torch.Size([1, 64, 64])
    print("Label:", label)