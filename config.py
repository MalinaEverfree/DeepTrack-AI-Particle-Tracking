OPTICS_CONFIG = {
    "NA": 1.0,
    "wavelength": 532e-9,
    "resolution": 1.0e-6,
    "magnification": 10,
    "output_region": (0, 0, 64, 64),
}

PARTICLE_CONFIG = {
    "radius_range_m": (0.5e-6, 1.5e-6),
    "refractive_index": 1.45,
    "position_range_px": (20, 44),
    "z_range_um": (-5, 5),
    "medium_refractive_index": 1.33,
}

NOISE_CONFIG = {
    "snr_range": (5, 20),
}