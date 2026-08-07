# Optical Tweezer Particle Tracker

Uses DeepTrack2 to generate synthetic training images (so we don't need to hand-label real ones) and trains a PyTorch CNN to predict a trapped particle's x, y, z, and radius from a microscope image. Also works on real video.

## Setup

Needs Python 3.11 (not 3.14 - PyTorch/DeepTrack2 don't support it yet).

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Structure

```
src/            all the code
models/         trained model weights land here
results/        plots and trajectory CSVs land here
```

## Running things

Everything's run from inside `src/`:

- `python simulation.py` - quick check that the simulator works
- `python train.py` - trains the model, saves to ../models/
- `python evaluate.py` - tests it on fresh simulated data
- `python single_frame.py` - run on one real image (click to mark the particle)
- `python track_video.py` - run on a real .tif video, click once on the first frame, outputs a trajectory

## Notes / things to know

- Model is trained on `MieSphere` particles (more realistic scattering physics) - tried the simpler `Sphere` first but it made training images look unrealistically clean compared to real footage.
- Widening the CNN (more filters per layer) cut error by ~45% - capacity was the main bottleneck early on, not training time.
- `track_video.py` keeps the crop fixed at wherever you first click, instead of following the prediction frame-to-frame. Tried the follow version first and it drifted way off the particle over a full video - small errors kept stacking up. Fixed crop avoids that.
- Real videos consistently show messier predictions for the first ~40-50 frames before settling down, even when the particle was already trapped before recording started. Probably camera/recording settling, not real motion - worth excluding those frames from analysis.
- No ground truth for real video, obviously, so real-video results are checked by eye (does the prediction land on the particle, does the trajectory look physically reasonable), not a precise error number.
- Only handles one particle in frame at a time.

## Could still improve

- Test on more real videos
- Handle multiple particles (would need a different model, like a U-Net)
- z and radius are consistently harder to predict than x/y (~1.5-1.7x worse) - haven't fully fixed this