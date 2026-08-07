import os
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageSequence

from model import ParticleTrackerCNN

# --- EDIT THIS ---
FRAME_PATH = r"C:\Users\mayus\Pictures\Cavity Lab Intern\ThorCam\LaserSpot01_17_5.tif"

CROP_SIZE = 120
MODEL_INPUT_SIZE = 64


def crop_from_array(arr, center_x, center_y, crop_size):
    half = crop_size // 2
    padded = np.pad(arr.astype("float32"), half + 1, mode="edge")
    x0p = int(center_x - half) + half + 1
    y0p = int(center_y - half) + half + 1
    return padded[y0p:y0p + crop_size, x0p:x0p + crop_size]


def preprocess(crop, target_size):
    img = Image.fromarray(crop).resize((target_size, target_size))
    arr = np.array(img).astype("float32")
    return arr / (arr.max() + 1e-8)


# --- Load model ---
model = ParticleTrackerCNN()
model.load_state_dict(torch.load("tracker_model.pt"))
model.eval()

# --- Load every page out of the stack ---
stack = Image.open(FRAME_PATH)
frames = [np.array(frame.convert("L")) for frame in ImageSequence.Iterator(stack)]
print(f"Found {len(frames)} frames in the stack")

if len(frames) == 0:
    raise RuntimeError("No frames found -- check FRAME_PATH is correct.")

# --- Click once on the FIRST frame only ---
fig, ax = plt.subplots()
ax.imshow(frames[0], cmap="gray")
ax.set_title("Click on the particle in the FIRST frame, then close this window")

clicked = {}
def onclick(event):
    clicked["x"], clicked["y"] = event.xdata, event.ydata
    print(f"Starting position: x={event.xdata:.1f}, y={event.ydata:.1f}")

fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()

if "x" not in clicked:
    raise RuntimeError("You closed the window without clicking first.")

current_x, current_y = clicked["x"], clicked["y"]

# --- Loop through every frame ---
# Crop stays FIXED at the original click position for every frame --
# no feedback loop, so no compounding drift.
results = []
for i, frame_arr in enumerate(frames):
    crop = crop_from_array(frame_arr, current_x, current_y, CROP_SIZE)
    processed = preprocess(crop, MODEL_INPUT_SIZE)

    img_tensor = torch.from_numpy(processed).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        pred = model(img_tensor).numpy()[0]

    x_norm, y_norm, z_norm, r_norm = pred

    crop_offset_x = current_x - CROP_SIZE / 2
    crop_offset_y = current_y - CROP_SIZE / 2
    scale = CROP_SIZE / MODEL_INPUT_SIZE

    abs_x = crop_offset_x + (x_norm * MODEL_INPUT_SIZE) * scale
    abs_y = crop_offset_y + (y_norm * MODEL_INPUT_SIZE) * scale

    results.append({
        "frame": i,
        "x_px": abs_x, "y_px": abs_y,
        "z_norm": z_norm, "r_norm": r_norm,
    })

    if i % 10 == 0:
        print(f"Frame {i}/{len(frames)}: x={abs_x:.1f}, y={abs_y:.1f}")

# --- Save results to CSV ---
with open("trajectory.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["frame", "x_px", "y_px", "z_norm", "r_norm"])
    writer.writeheader()
    writer.writerows(results)
print("Saved trajectory.csv")

# --- Plot the trajectory ---
xs = [r["x_px"] for r in results]
ys = [r["y_px"] for r in results]

plt.figure()
plt.plot(xs, ys, marker="o", markersize=3, linewidth=1)
plt.scatter([xs[0]], [ys[0]], c="green", s=100, label="start", zorder=5)
plt.scatter([xs[-1]], [ys[-1]], c="red", s=100, label="end", zorder=5)
plt.xlabel("x (pixels)")
plt.ylabel("y (pixels)")
plt.title("Tracked particle trajectory")
plt.legend()
plt.gca().invert_yaxis()
plt.savefig("trajectory_plot.png", dpi=150)
print("Saved trajectory_plot.png")
plt.show()
