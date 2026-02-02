#!/usr/bin/env python3
"""Generate synthetic M42 dataset (16 subs) + guiding logs + guide videos, then stack.

Assumptions:
- 2032mm SCT, SQM~20 background modeled as constant offset + noise.
- Linear 16-bit FITS, 2048x2048.
- Seeing modeled as Gaussian blur; guiding errors add anisotropic blur + shift.
"""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter, shift


@dataclass
class SimConfig:
    subs: int = 16
    width: int = 2048
    height: int = 2048
    exposure_s: float = 60.0
    guide_fps: int = 30
    guide_seconds: int = 5
    focal_length_mm: float = 2032.0
    pixel_size_um: float = 3.45
    sqm: float = 20.0


def pixel_scale_arcsec(config: SimConfig) -> float:
    return 206.265 * (config.pixel_size_um / 1000.0) / config.focal_length_mm


def gaussian2d(x, y, cx, cy, sigma_x, sigma_y, amp):
    return amp * np.exp(-(((x - cx) ** 2) / (2 * sigma_x**2) + ((y - cy) ** 2) / (2 * sigma_y**2)))


def make_base_nebula(config: SimConfig, rng: np.random.Generator) -> np.ndarray:
    y, x = np.mgrid[0 : config.height, 0 : config.width]
    x = x.astype(np.float32)
    y = y.astype(np.float32)

    cx = config.width * 0.52
    cy = config.height * 0.48

    nebula = gaussian2d(x, y, cx, cy, 220, 180, 9000)
    nebula += gaussian2d(x, y, cx + 220, cy - 120, 380, 260, 3500)
    nebula += gaussian2d(x, y, cx - 260, cy + 160, 520, 420, 2800)

    # Fractal-ish dust structure
    noise1 = gaussian_filter(rng.random((config.height, config.width)), sigma=60)
    noise2 = gaussian_filter(rng.random((config.height, config.width)), sigma=180)
    dust = 0.6 * noise1 + 0.4 * noise2
    dust = (dust - dust.min()) / (dust.max() - dust.min() + 1e-6)
    nebula += 2500 * dust

    # Dust lanes subtract
    lanes = gaussian_filter(rng.random((config.height, config.width)), sigma=120)
    lanes = (lanes - lanes.min()) / (lanes.max() - lanes.min() + 1e-6)
    nebula -= 1200 * lanes
    nebula = np.clip(nebula, 0, None)
    return nebula


def add_star_field(image: np.ndarray, config: SimConfig, rng: np.random.Generator) -> np.ndarray:
    stars = image.copy()
    psf_size = 17
    half = psf_size // 2
    xk, yk = np.mgrid[-half : half + 1, -half : half + 1]
    psf = np.exp(-(xk**2 + yk**2) / (2 * 1.1**2))
    psf /= psf.sum()

    star_count = 1400
    for _ in range(star_count):
        x = rng.integers(half, config.width - half)
        y = rng.integers(half, config.height - half)
        flux = 10 ** rng.uniform(2.0, 4.2)
        stars[y - half : y + half + 1, x - half : x + half + 1] += flux * psf
    return stars


def generate_guiding_series(duration_s: int, rng: np.random.Generator):
    t = np.arange(duration_s)
    ra = 0.6 * np.sin(t / 16.0) + 0.25 * np.sin(t / 5.0) + rng.normal(0, 0.12, size=t.size)
    dec = 0.5 * np.cos(t / 18.0) + 0.2 * np.sin(t / 7.0) + rng.normal(0, 0.1, size=t.size)
    return t, ra, dec


def write_phd2_log(path: Path, t, ra, dec, profile_name: str):
    lines = []
    lines.append("PHD2 version 2.6.13")
    lines.append("Log start 2026-02-01 22:00:00")
    lines.append("Guiding Begins at 2026-02-01 22:00:02")
    lines.append(f"Equipment Profile = \"{profile_name}\"")
    lines.append("Mount = \"Mount\"")
    lines.append("Camera = \"Simulator\"")
    lines.append("Guider = \"Hysteresis\"")
    lines.append("Exposure = 1.0 s")
    lines.append("Image scale = 0.35 arc-sec/px")
    lines.append("")
    lines.append(
        "Frame,Time,Mount,dx,dy,RARawDistance,DECRawDistance,RAGuideDistance,DECGuideDistance,"
        "RADuration,RADirection,DECDuration,DECDirection,XStep,YStep,StarMass,SNR,ErrorCode"
    )
    for i in range(t.size):
        ra_raw = ra[i]
        dec_raw = dec[i]
        ra_guide = ra_raw if abs(ra_raw) > 0.2 else 0.0
        dec_guide = dec_raw if abs(dec_raw) > 0.2 else 0.0
        ra_dur = int(abs(ra_guide) * 120)
        dec_dur = int(abs(dec_guide) * 140)
        ra_dir = "E" if ra_guide > 0 else ("W" if ra_guide < 0 else "")
        dec_dir = "N" if dec_guide > 0 else ("S" if dec_guide < 0 else "")
        star_mass = 30000 + (i * 37) % 20000
        snr = 28.0 + 5.0 * math.sin(i / 20.0)
        lines.append(
            f"{i+1},{t[i]:.3f},\"Mount\",{ra_raw:.3f},{dec_raw:.3f},{ra_raw:.3f},{dec_raw:.3f},"
            f"{ra_guide:.3f},{dec_guide:.3f},{ra_dur},{ra_dir},{dec_dur},{dec_dir},,,{star_mass},{snr:.2f},0"
        )
    path.write_text("\n".join(lines) + "\n")


def write_ser(path: Path, frames: np.ndarray):
    # SER header 178 bytes; minimal INDI recorder variant
    # FileID (14 bytes) + 2 pad
    file_id = b"INDI-RECORDER"  # 14 bytes
    header = bytearray(178)
    header[0:14] = file_id
    # 7 uint32 values
    def put_u32(offset, value):
        header[offset : offset + 4] = int(value).to_bytes(4, "little", signed=False)

    width = frames.shape[2]
    height = frames.shape[1]
    frame_count = frames.shape[0]
    put_u32(14 + 0 * 4, 0)  # lu_id
    put_u32(14 + 1 * 4, 0)  # color_id (mono)
    put_u32(14 + 2 * 4, 0)  # little endian flag (ignored)
    put_u32(14 + 3 * 4, width)
    put_u32(14 + 4 * 4, height)
    put_u32(14 + 5 * 4, 16)  # pixel depth
    put_u32(14 + 6 * 4, frame_count)

    # Observer/Instrument/Telescope fields
    header[42:82] = b"Synthetic Observer".ljust(40, b"\0")
    header[82:122] = b"Synthetic GuideCam".ljust(40, b"\0")
    header[122:162] = b"SCT 2032mm".ljust(40, b"\0")

    # Datetime fields left zeroed
    with path.open("wb") as f:
        f.write(header)
        # Write frames as little-endian u16
        f.write(frames.astype("<u2").tobytes())


def build_dataset(base_dir: Path, config: SimConfig):
    base_dir.mkdir(parents=True, exist_ok=True)
    subs_dir = base_dir / "subs"
    phd2_dir = base_dir / "phd2"
    guide_dir = base_dir / "secondary"
    subs_dir.mkdir(exist_ok=True)
    phd2_dir.mkdir(exist_ok=True)
    guide_dir.mkdir(exist_ok=True)

    rng = np.random.default_rng(20260201)

    base_nebula = make_base_nebula(config, rng)
    base = add_star_field(base_nebula, config, rng)

    scale = pixel_scale_arcsec(config)
    results = []

    for i in range(config.subs):
        sub_rng = np.random.default_rng(1000 + i)
        t, ra, dec = generate_guiding_series(int(config.exposure_s), sub_rng)
        rms_ra = float(np.sqrt(np.mean(ra**2)))
        rms_dec = float(np.sqrt(np.mean(dec**2)))

        seeing_fwhm = float(sub_rng.normal(2.6, 0.4))
        seeing_fwhm = max(1.5, seeing_fwhm)
        seeing_sigma = seeing_fwhm / (2.355 * scale)
        track_sigma_x = rms_ra / (2.355 * scale)
        track_sigma_y = rms_dec / (2.355 * scale)

        blurred = gaussian_filter(base, sigma=(seeing_sigma + track_sigma_y, seeing_sigma + track_sigma_x))
        mean_shift_x = float(np.mean(ra) / scale)
        mean_shift_y = float(np.mean(dec) / scale)
        shifted = shift(blurred, shift=(mean_shift_y, mean_shift_x), order=1, mode="nearest")

        # Background from SQM~20 (approx) and noise
        background = 800.0 + 50.0 * (20.0 - config.sqm)
        image = shifted + background
        # Shot noise and read noise
        noisy = sub_rng.poisson(np.clip(image, 0, None)).astype(np.float32)
        noisy += sub_rng.normal(0.0, 6.0, size=image.shape)

        # Scale to 16-bit
        noisy -= noisy.min()
        scale_factor = 50000.0 / max(noisy.max(), 1.0)
        img16 = np.clip(noisy * scale_factor, 0, 65535).astype(np.uint16)

        fits_path = subs_dir / f"sub_{i+1:03d}.fits"
        hdr = fits.Header()
        hdr["FOCALLEN"] = config.focal_length_mm
        hdr["EXPTIME"] = config.exposure_s
        hdr["PIXSCALE"] = scale
        hdr["SQM"] = config.sqm
        hdr["SEEING"] = seeing_fwhm
        hdr["SIMGUID"] = True
        fits.PrimaryHDU(img16, header=hdr).writeto(fits_path, overwrite=True)

        phd2_path = phd2_dir / f"PHD2_GuideLog_sub_{i+1:03d}.txt"
        write_phd2_log(phd2_path, t, ra, dec, profile_name=f"SCT-2032-sub-{i+1:03d}")

        # Secondary guide video (single bright star + jitter)
        frames = int(config.guide_seconds * config.guide_fps)
        g_height, g_width = 480, 640
        gy, gx = np.mgrid[0:g_height, 0:g_width]
        gx = gx.astype(np.float32)
        gy = gy.astype(np.float32)
        base_star = gaussian2d(gx, gy, g_width * 0.5, g_height * 0.5, 2.5, 2.5, 20000)
        base_star += gaussian2d(gx, gy, g_width * 0.6, g_height * 0.45, 2.0, 2.0, 8000)

        # High-frequency jitter
        hf = sub_rng.normal(0, 0.15, size=frames)
        hf2 = sub_rng.normal(0, 0.12, size=frames)
        t_frames = np.linspace(0, config.guide_seconds, frames)
        ra_frame = np.interp(t_frames, t, ra) + hf
        dec_frame = np.interp(t_frames, t, dec) + hf2

        guide_frames = np.zeros((frames, g_height, g_width), dtype=np.uint16)
        for fi in range(frames):
            sx = ra_frame[fi] / scale * 0.35
            sy = dec_frame[fi] / scale * 0.35
            frame = shift(base_star, shift=(sy, sx), order=1, mode="nearest")
            frame = frame + sub_rng.normal(0, 15.0, size=frame.shape)
            frame = np.clip(frame, 0, 65535).astype(np.uint16)
            guide_frames[fi] = frame

        ser_path = guide_dir / f"guide_sub_{i+1:03d}.ser"
        write_ser(ser_path, guide_frames)

        results.append(
            {
                "sub": i + 1,
                "fits": str(fits_path),
                "phd2": str(phd2_path),
                "ser": str(ser_path),
                "seeing_fwhm": seeing_fwhm,
                "rms_ra": rms_ra,
                "rms_dec": rms_dec,
            }
        )

    (base_dir / "metadata.json").write_text(json.dumps({"config": asdict(config), "subs": results}, indent=2))


def compute_weights(base_dir: Path) -> dict[int, float]:
    weights = {}
    tracefield = Path("zig-out/bin/tracefield")
    if not tracefield.exists():
        subprocess.check_call(["zig", "build"], cwd=Path.cwd())
    for sub in range(1, 17):
        fits_path = base_dir / "subs" / f"sub_{sub:03d}.fits"
        phd2_path = base_dir / "phd2" / f"PHD2_GuideLog_sub_{sub:03d}.txt"
        ser_path = base_dir / "secondary" / f"guide_sub_{sub:03d}.ser"
        out = subprocess.check_output(
            [
                str(tracefield),
                "--fits",
                str(fits_path),
                "--phd2",
                str(phd2_path),
                "--ser",
                str(ser_path),
            ]
        )
        payload = json.loads(out.decode("utf-8"))
        weights[sub] = float(payload["weight"])
    (base_dir / "weights.json").write_text(json.dumps(weights, indent=2))
    return weights


def stack_outputs(base_dir: Path, weights: dict[int, float]):
    subs = []
    for sub in range(1, 17):
        fits_path = base_dir / "subs" / f"sub_{sub:03d}.fits"
        subs.append(fits.getdata(fits_path).astype(np.float32))
    stack = np.mean(np.stack(subs, axis=0), axis=0)

    w = np.array([weights[i] for i in range(1, 17)], dtype=np.float32)
    w = w / np.sum(w)
    weighted = np.tensordot(w, np.stack(subs, axis=0), axes=(0, 0))

    fits.PrimaryHDU(stack.astype(np.uint16)).writeto(base_dir / "stack_plain.fits", overwrite=True)
    fits.PrimaryHDU(weighted.astype(np.uint16)).writeto(base_dir / "stack_weighted.fits", overwrite=True)


def main():
    base_dir = Path("testdata/m42")
    config = SimConfig()
    build_dataset(base_dir, config)
    weights = compute_weights(base_dir)
    stack_outputs(base_dir, weights)


if __name__ == "__main__":
    main()
