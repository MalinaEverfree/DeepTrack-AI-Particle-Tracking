"""
simulation.py
==============
Builds the synthetic optical-tweezer imaging pipeline using DeepTrack2 2.x.
"""

import numpy as np
import deeptrack as dt

from config import OPTICS_CONFIG, PARTICLE_CONFIG, NOISE_CONFIG


def build_optics():
    return dt.Brightfield(
        NA=OPTICS_CONFIG["NA"],
        wavelength=OPTICS_CONFIG["wavelength"],
        resolution=OPTICS_CONFIG["resolution"],
        magnification=OPTICS_CONFIG["magnification"],
        output_region=OPTICS_CONFIG["output_region"],
    )


def build_particle():
    lo, hi = PARTICLE_CONFIG["position_range_px"]
    zlo, zhi = PARTICLE_CONFIG["z_range_um"]
    rlo, rhi = PARTICLE_CONFIG["radius_range_m"]

    return dt.MieSphere(
        position=lambda: np.random.uniform(lo, hi, size=2),
        z=lambda: np.random.uniform(zlo, zhi),
        radius=lambda: np.random.uniform(rlo, rhi),
        refractive_index=PARTICLE_CONFIG["refractive_index"],
        medium_refractive_index=PARTICLE_CONFIG["medium_refractive_index"],
    )


def build_noise():
    lo, hi = NOISE_CONFIG["snr_range"]
    return dt.Poisson(snr=lambda: np.random.uniform(lo, hi))


def build_pipeline():
    optics = build_optics()
    particle = build_particle()
    noise = build_noise()

    image_pipeline = optics(particle) >> noise

    # This function reads ground truth straight off the `particle` object's
    # own properties (NOT off the resolved image), so it doesn't depend on
    # DeepTrack2's "Image object" behavior at all.
    def get_label(image):
        frame_size = OPTICS_CONFIG["output_region"][2]

        pos = np.array(particle.position()) / frame_size

        z = particle.z()
        z_norm = (z - zlo_g) / (zhi_g - zlo_g)

        r = particle.radius()
        r_norm = (r - rlo_g) / (rhi_g - rlo_g)

        return np.array([pos[0], pos[1], z_norm, r_norm], dtype=np.float32)

    # Pull these ranges once, outside get_label, so we're not re-reading
    # the config dict on every single call (minor tidiness, not required).
    zlo_g, zhi_g = PARTICLE_CONFIG["z_range_um"]
    rlo_g, rhi_g = PARTICLE_CONFIG["radius_range_m"]

    pipeline = image_pipeline & (image_pipeline >> dt.Lambda(lambda: get_label))
    return pipeline


if __name__ == "__main__":
    pipeline = build_pipeline()
    image, label = pipeline.update().resolve()
    print("Image shape:", np.array(image).shape)
    print("Label [x, y, z, r] (normalized 0-1):", label)