import torch
import torch.nn as nn

class ParticleTrackerCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Convolution blocks: each one is Conv -> BatchNorm -> ReLU -> Pool
        # Change these four lines in __init__:
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)      # was 16
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)     # was 16->32
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)    # was 32->64
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)   # was 64->128
        self.bn4 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()

        # After 4 rounds of pooling, a 64x64 image becomes 4x4.
        # 128 channels * 4 * 4 pixels = 2048 numbers going into the dense layers.
        self.fc1 = nn.Linear(256 * 4 * 4, 64)   # was 128*4*4 -> 32
        self.fc2 = nn.Linear(64, 64)             # was 32 -> 32
        self.fc3 = nn.Linear(64, 4)

        self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.pool(self.relu(self.bn4(self.conv4(x))))

        x = x.view(x.size(0), -1)   # flatten everything except the batch dimension

        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.sigmoid(self.fc3(x))
        return x

if __name__ == "__main__":
    model = ParticleTrackerCNN()
    print(model)

    # Sanity check: does a fake batch of images produce the right output shape?
    fake_batch = torch.randn(8, 1, 64, 64)   # 8 images, 1 channel, 64x64
    output = model(fake_batch)
    print("Output shape:", output.shape)   # should be torch.Size([8, 4])