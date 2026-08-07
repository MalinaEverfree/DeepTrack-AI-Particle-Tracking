import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

from model import ParticleTrackerCNN

# --- EDIT ONLY THIS LINE WHEN YOU SWITCH IMAGES ---
FRAME_PATH = r"C:\Users\mayus\Pictures\Cavity Lab Intern\LaserSpot01_17_7.png"

CROP_SIZE = 120
MODEL_INPUT_SIZE = 64

# --- Step 1: click on the particle to get its coordinates ---
img_full = Image.open(FRAME_PATH)
arr_full = np.array(img_full)

fig, ax = plt.subplots()
ax.imshow(arr_full)
ax.set_title("Click on the trapped particle, then close this window")

clicked = {}
def onclick(event):
    clicked["x"], clicked["y"] = event.xdata, event.ydata
    print(f"Clicked: x={event.xdata:.1f}, y={event.ydata:.1f}")

fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()   # script pauses here until you close this window

if "x" not in clicked:
    raise RuntimeError("You closed the window without clicking on the particle first.")

CENTER_X, CENTER_Y = clicked["x"], clicked["y"]

# --- Step 2: crop, preprocess, predict (same as before) ---
def load_and_crop(frame_path, center_x, center_y, crop_size):
    img = Image.open(frame_path).convert("L")
    arr = np.array(img).astype("float32")
    half = crop_size // 2
    padded = np.pad(arr, half + 1, mode="edge")
    x0p = int(center_x - half) + half + 1
    y0p = int(center_y - half) + half + 1
    return padded[y0p:y0p + crop_size, x0p:x0p + crop_size]

def preprocess(crop, target_size):
    img = Image.fromarray(crop).resize((target_size, target_size))
    arr = np.array(img).astype("float32")
    return arr / (arr.max() + 1e-8)

model = ParticleTrackerCNN()
model.load_state_dict(torch.load("tracker_model.pt"))
model.eval()

crop = load_and_crop(FRAME_PATH, CENTER_X, CENTER_Y, CROP_SIZE)
processed = preprocess(crop, MODEL_INPUT_SIZE)

img_tensor = torch.from_numpy(processed).unsqueeze(0).unsqueeze(0)
with torch.no_grad():
    pred = model(img_tensor).numpy()[0]

x_norm, y_norm, z_norm, r_norm = pred
x_px, y_px = x_norm * MODEL_INPUT_SIZE, y_norm * MODEL_INPUT_SIZE

print(f"Predicted position: x={x_px:.1f}px, y={y_px:.1f}px")
print(f"Predicted z (normalized):      {z_norm:.3f}")
print(f"Predicted radius (normalized): {r_norm:.3f}")

fig2, ax2 = plt.subplots()
ax2.imshow(processed, cmap="gray")
ax2.scatter([x_px], [y_px], c="red", marker="x", s=150, label="predicted")
ax2.legend()
ax2.set_title("Model's prediction on your real cropped frame")
plt.savefig("../results/real_frame_prediction.png", dpi=150)
print("Saved real_frame_prediction.png")
plt.show()
