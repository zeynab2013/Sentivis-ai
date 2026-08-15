"""Real super-resolution backends for SENTIVIS.

Priority:
1. Real-ESRGAN compact (``realesr-general-x4v3``) — preferred on CPU / ~2 GB VRAM
2. Real-ESRGAN RRDBNet x2plus — if weights present
3. OpenCV EDSR DNN — if ``EDSR_x2.pb`` / ``EDSR_x4.pb`` present
4. Never claim Lanczos alone as AI super-resolution

Hardware policy:
- Lazy load, tiled inference, CUDA→CPU fallback, unload after use when requested
"""

from __future__ import annotations

import hashlib
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from core.logging import get_logger

logger = get_logger(__name__)

# Official Real-ESRGAN release weights (xinntao/Real-ESRGAN).
_WEIGHT_URLS: dict[str, str] = {
    "realesr-general-x4v3.pth": (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/"
        "realesr-general-x4v3.pth"
    ),
    "RealESRGAN_x2plus.pth": (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/"
        "RealESRGAN_x2plus.pth"
    ),
}

_MODEL_CACHE: dict[str, object] = {}
_DISK_CACHE: dict[str, Path] = {}


@dataclass(frozen=True)
class UpscaleResult:
    """Outcome of a super-resolution attempt."""

    pixels: NDArray[np.uint8]
    backend: str  # realesrgan | opencv_dnn | none | failed
    model_name: str
    scale: int
    device: str
    tile_size: int
    input_size: tuple[int, int]  # (w, h)
    output_size: tuple[int, int]  # (w, h)
    true_sr: bool
    message: str = ""


def upscale(
    pixels: NDArray[np.uint8],
    *,
    models_dir: Path,
    min_dimension: int,
    scale: int = 2,
    tile_size: int = 128,
    tile_overlap: int = 8,
    device: str = "auto",
    max_output_side: int = 2048,
    allow_download: bool = True,
) -> UpscaleResult:
    """Run genuine learned SR when possible; otherwise return unchanged pixels."""
    height, width = int(pixels.shape[0]), int(pixels.shape[1])
    input_size = (width, height)
    min_dim = min(height, width)

    # Already large enough — skip SR (HIGH-res or already upscaled).
    if min_dim >= min_dimension and min(height, width) >= 720:
        return UpscaleResult(
            pixels=pixels,
            backend="none",
            model_name="",
            scale=1,
            device="n/a",
            tile_size=tile_size,
            input_size=input_size,
            output_size=input_size,
            true_sr=False,
            message="input already large enough",
        )

    scale = int(max(2, min(4, scale)))
    # Cap absurd output sizes (e.g. tiny 200px → avoid multi-megapixel blowups).
    projected = max(width, height) * scale
    if projected > max_output_side:
        # Prefer 2×; if still too large, skip.
        if max(width, height) * 2 <= max_output_side:
            scale = 2
        else:
            return UpscaleResult(
                pixels=pixels,
                backend="none",
                model_name="",
                scale=1,
                device="n/a",
                tile_size=tile_size,
                input_size=input_size,
                output_size=input_size,
                true_sr=False,
                message="projected output exceeds max_output_side",
            )

    logger.info(
        "[SR] Input: %sx%s | requested_scale=%sx | tile=%s | device_pref=%s",
        width,
        height,
        scale,
        tile_size,
        device,
    )

    realesr = _try_realesrgan(
        pixels,
        models_dir=models_dir,
        scale=scale,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        device_pref=device,
        allow_download=allow_download,
    )
    if realesr is not None:
        return realesr

    dnn = _try_opencv_dnn(pixels, models_dir=models_dir, scale=scale)
    if dnn is not None:
        return dnn

    logger.info("[SR] Validation: no learned SR backend available")
    return UpscaleResult(
        pixels=pixels,
        backend="failed",
        model_name="",
        scale=1,
        device="n/a",
        tile_size=tile_size,
        input_size=input_size,
        output_size=input_size,
        true_sr=False,
        message="no Real-ESRGAN / EDSR weights available",
    )


def lanczos_upscale(pixels: NDArray[np.uint8], scale: int) -> NDArray[np.uint8]:
    """Interpolation baseline for fair SR validation (not labeled as AI SR)."""
    height, width = pixels.shape[:2]
    scale = max(1, int(scale))
    pil = Image.fromarray(pixels).resize(
        (width * scale, height * scale),
        Image.Resampling.LANCZOS,
    )
    return np.asarray(pil, dtype=np.uint8)


def laplacian_variance(pixels: NDArray[np.uint8]) -> float:
    """Sharpness proxy used for SR validation."""
    try:
        import cv2
    except Exception:  # noqa: BLE001
        gray = np.mean(pixels.astype(np.float32), axis=2) if pixels.ndim == 3 else pixels.astype(np.float32)
        gy, gx = np.gradient(gray)
        return float(np.var(gx) + np.var(gy))
    gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY) if pixels.ndim == 3 else pixels
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def validate_sr_against_baseline(
    original: NDArray[np.uint8],
    enhanced: NDArray[np.uint8],
    *,
    scale: int,
) -> tuple[bool, str]:
    """Accept SR when output is a valid upscaled image without severe corruption.

    Does NOT reject solely because an experimental sharpness metric failed to
    rise — resolution change makes raw Laplacian comparisons unreliable.
    """
    if enhanced is None or enhanced.size == 0:
        return False, "empty output"
    if not np.isfinite(enhanced.astype(np.float32)).all():
        return False, "NaN/Inf pixels"
    if enhanced.ndim != 3 or enhanced.shape[2] != 3:
        return False, "invalid channel layout"
    if enhanced.shape[0] <= original.shape[0] or enhanced.shape[1] <= original.shape[1]:
        return False, "output resolution did not increase"
    expected_h = original.shape[0] * max(1, int(scale))
    expected_w = original.shape[1] * max(1, int(scale))
    # Allow small rounding differences from model padding.
    if enhanced.shape[0] < expected_h * 0.9 or enhanced.shape[1] < expected_w * 0.9:
        return False, f"output too small ({enhanced.shape[1]}x{enhanced.shape[0]} < ~{expected_w}x{expected_h})"

    # Identical content at higher res after nearest-like expand is not real SR.
    down = np.asarray(
        Image.fromarray(enhanced).resize(
            (original.shape[1], original.shape[0]),
            Image.Resampling.LANCZOS,
        ),
        dtype=np.uint8,
    )
    if np.array_equal(down, original):
        return False, "output identical to input when downsampled"

    # Severe chroma collapse only.
    orig_chroma = float(np.mean(np.std(original.reshape(-1, 3).astype(np.float32), axis=1)))
    down_chroma = float(np.mean(np.std(down.reshape(-1, 3).astype(np.float32), axis=1)))
    if orig_chroma > 8.0 and down_chroma < orig_chroma * 0.55:
        return False, "severe color/chroma corruption"

    # Extreme mean shift (washed out / black).
    if abs(float(down.mean()) - float(original.mean())) > 45.0:
        return False, "extreme brightness corruption"

    return True, "valid_upscaled_output"


def _resolve_device(device_pref: str) -> str:
    pref = (device_pref or "auto").lower().strip()
    try:
        import torch
    except Exception:  # noqa: BLE001
        return "cpu"
    if pref == "cpu":
        return "cpu"
    if pref == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        try:
            props = torch.cuda.get_device_properties(0)
            # Stay on CUDA only when there is a usable amount of free memory.
            if props.total_memory >= 1.5 * 1024**3:
                return "cuda"
        except Exception:  # noqa: BLE001
            pass
    return "cpu"


def _ensure_weights(models_dir: Path, filename: str, *, allow_download: bool) -> Path | None:
    target_dir = models_dir / "realesrgan"
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / filename
    if path.is_file() and path.stat().st_size > 1_000_000:
        return path
    if not allow_download:
        return None
    url = _WEIGHT_URLS.get(filename)
    if not url:
        return None
    tmp = path.with_suffix(".partial")
    try:
        logger.info("[SR] Downloading weights %s …", filename)
        urllib.request.urlretrieve(url, tmp)  # noqa: S310 — pinned GitHub release URL
        if tmp.stat().st_size < 1_000_000:
            tmp.unlink(missing_ok=True)
            return None
        tmp.replace(path)
        logger.info("[SR] Weights ready: %s (%.1f MB)", path.name, path.stat().st_size / 1e6)
        return path
    except Exception as exc:  # noqa: BLE001
        logger.warning("[SR] Weight download failed for %s: %s", filename, exc)
        tmp.unlink(missing_ok=True)
        return None


def _try_realesrgan(
    pixels: NDArray[np.uint8],
    *,
    models_dir: Path,
    scale: int,
    tile_size: int,
    tile_overlap: int,
    device_pref: str,
    allow_download: bool,
) -> UpscaleResult | None:
    # Prefer compact general model (small, CPU-friendly), then x2plus RRDB.
    compact = _ensure_weights(models_dir, "realesr-general-x4v3.pth", allow_download=allow_download)
    x2 = _ensure_weights(models_dir, "RealESRGAN_x2plus.pth", allow_download=allow_download)

    device = _resolve_device(device_pref)
    try:
        import torch

        cuda_ok = bool(torch.cuda.is_available())
        vram = ""
        if cuda_ok:
            try:
                free, total = torch.cuda.mem_get_info(0)
                vram = f"free={free/1e9:.2f}GB total={total/1e9:.2f}GB"
            except Exception:  # noqa: BLE001
                props = torch.cuda.get_device_properties(0)
                vram = f"total={props.total_memory/1e9:.2f}GB"
    except Exception as exc:  # noqa: BLE001
        cuda_ok = False
        vram = f"torch_unavailable:{exc}"

    print(
        f"[SR DEBUG]\n"
        f"Model weights compact: {compact is not None} path={compact}\n"
        f"Model weights x2plus: {x2 is not None} path={x2}\n"
        f"Device pref/resolved: {device_pref}/{device}\n"
        f"CUDA available: {cuda_ok}\n"
        f"VRAM: {vram or 'n/a'}",
        flush=True,
    )

    attempts: list[tuple[str, Path, str]] = []
    # Prefer compact (lightweight, CPU-friendly) first; then x2plus RRDB.
    if compact is not None:
        attempts.append(("realesr-general-x4v3", compact, "srvgg"))
    if x2 is not None:
        attempts.append(("RealESRGAN_x2plus", x2, "rrdb_x2"))

    tile_candidates = [tile_size]
    for smaller in (96, 64):
        if smaller < tile_size and smaller not in tile_candidates:
            tile_candidates.append(smaller)

    for model_name, weights, kind in attempts:
        for try_device in ((device, "cpu") if device != "cpu" else ("cpu",)):
            for try_tile in tile_candidates:
                try:
                    print(
                        f"[SR DEBUG]\n"
                        f"Model loaded: attempting\n"
                        f"Model name: {model_name}\n"
                        f"Model path: {weights}\n"
                        f"Device: {try_device}\n"
                        f"Tile: {try_tile}\n"
                        f"Inference started: TRUE",
                        flush=True,
                    )
                    out = _infer_realesrgan(
                        pixels,
                        weights=weights,
                        kind=kind,
                        outscale=scale,
                        tile_size=try_tile,
                        tile_overlap=tile_overlap,
                        device=try_device,
                    )
                    expected_h = pixels.shape[0] * scale
                    expected_w = pixels.shape[1] * scale
                    if out.shape[0] < expected_h * 0.9 or out.shape[1] < expected_w * 0.9:
                        raise RuntimeError(
                            f"SR output size {out.shape[1]}x{out.shape[0]} "
                            f"< expected ~{expected_w}x{expected_h} "
                            f"(model did not upscale)"
                        )
                    print(
                        f"[SR DEBUG]\n"
                        f"Inference completed: TRUE\n"
                        f"Enhanced image exists: TRUE\n"
                        f"Enhanced image size: {out.shape[1]}x{out.shape[0]}",
                        flush=True,
                    )
                    logger.info(
                        "[SR] Model: %s | Scale: %sx | Device: %s | Tile: %s | Output: %sx%s",
                        model_name,
                        scale,
                        try_device,
                        try_tile,
                        out.shape[1],
                        out.shape[0],
                    )
                    return UpscaleResult(
                        pixels=out,
                        backend="realesrgan",
                        model_name=model_name,
                        scale=scale,
                        device=try_device,
                        tile_size=try_tile,
                        input_size=(pixels.shape[1], pixels.shape[0]),
                        output_size=(out.shape[1], out.shape[0]),
                        true_sr=True,
                        message="ok",
                    )
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"[SR DEBUG]\n"
                        f"Inference completed: FALSE\n"
                        f"Exception: {type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    logger.warning(
                        "[SR] Inference failed model=%s device=%s tile=%s err=%s",
                        model_name,
                        try_device,
                        try_tile,
                        exc,
                        exc_info=True,
                    )
                    _clear_torch_cache()
                    # On OOM, try smaller tile / next device; otherwise try next tile then model.
                    msg = str(exc).lower()
                    if "out of memory" in msg or "cuda" in msg and "memory" in msg:
                        continue
                    # Dimension / arch errors: don't waste time on smaller tiles.
                    if "did not upscale" in msg or "size mismatch" in msg:
                        break
                    continue
    return None


def _infer_realesrgan(
    pixels: NDArray[np.uint8],
    *,
    weights: Path,
    kind: str,
    outscale: int,
    tile_size: int,
    tile_overlap: int,
    device: str,
) -> NDArray[np.uint8]:
    import torch

    from vision.enhancement.sr_arch import RRDBNet, SRVGGNetCompact

    # Include arch revision so a fixed RRDBNet is not reused from a broken cache.
    cache_key = f"{weights.resolve()}:{kind}:{device}:arch_v2_two_upsamples"
    model = _MODEL_CACHE.get(cache_key)
    if model is None:
        print(f"[SR DEBUG]\nLoading weights into new net: {weights.name} kind={kind}", flush=True)
        state = torch.load(str(weights), map_location="cpu")
        if isinstance(state, dict) and "params_ema" in state:
            state = state["params_ema"]
        elif isinstance(state, dict) and "params" in state:
            state = state["params"]
        if isinstance(state, dict):
            state = {str(k).replace("module.", ""): v for k, v in state.items()}

        if kind == "srvgg":
            net: torch.nn.Module = SRVGGNetCompact(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_conv=32,
                upscale=4,
                act_type="prelu",
            )
            net_scale = 4
        else:
            net = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=2,
            )
            net_scale = 2

        net.load_state_dict(state, strict=True)
        net.eval()
        net = net.to(device)
        _MODEL_CACHE[cache_key] = (net, net_scale)
        model = _MODEL_CACHE[cache_key]

    net, net_scale = model  # type: ignore[misc]
    # Model native scale may be 4; we still request outscale (usually 2) by
    # running native SR then downsampling if needed — preserves learned priors
    # better than forcing a mismatched PixelShuffle.
    with torch.inference_mode():
        rgb = pixels.astype(np.float32) / 255.0
        tensor = torch.from_numpy(np.transpose(rgb, (2, 0, 1))).unsqueeze(0).to(device)
        if tile_size and min(tensor.shape[-2], tensor.shape[-1]) > tile_size:
            output = _tiled_forward(net, tensor, tile=tile_size, overlap=tile_overlap)
        else:
            output = net(tensor)
        output = output.clamp(0, 1).squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        out_u8 = (output * 255.0).round().astype(np.uint8)

    if net_scale != outscale:
        target = (pixels.shape[1] * outscale, pixels.shape[0] * outscale)
        out_u8 = np.asarray(
            Image.fromarray(out_u8).resize(target, Image.Resampling.LANCZOS),
            dtype=np.uint8,
        )
    return out_u8


def _tiled_forward(
    model: object,
    tensor: "torch.Tensor",
    *,
    tile: int,
    overlap: int,
) -> "torch.Tensor":
    import torch

    b, c, h, w = tensor.shape
    # Probe scale with a tiny corner crop.
    with torch.inference_mode():
        probe = model(tensor[:, :, :16, :16])
    scale = probe.shape[-1] // 16
    out_h, out_w = h * scale, w * scale
    output = torch.zeros((b, c, out_h, out_w), device=tensor.device, dtype=tensor.dtype)
    weight = torch.zeros_like(output)
    step = max(1, tile - overlap)
    for y in range(0, h, step):
        for x in range(0, w, step):
            y1 = min(y + tile, h)
            x1 = min(x + tile, w)
            y0 = max(0, y1 - tile)
            x0 = max(0, x1 - tile)
            tile_in = tensor[:, :, y0:y1, x0:x1]
            tile_out = model(tile_in)
            oy0, ox0 = y0 * scale, x0 * scale
            oy1, ox1 = y1 * scale, x1 * scale
            output[:, :, oy0:oy1, ox0:ox1] += tile_out
            weight[:, :, oy0:oy1, ox0:ox1] += 1.0
    return output / weight.clamp_min(1e-3)


def _try_opencv_dnn(
    pixels: NDArray[np.uint8],
    *,
    models_dir: Path,
    scale: int,
) -> UpscaleResult | None:
    """Optional OpenCV EDSR when contrib ``dnn_superres`` or TF pb is available."""
    sr_dir = models_dir / "super_resolution"
    candidates = [
        (sr_dir / f"EDSR_x{scale}.pb", scale),
        (sr_dir / "EDSR_x2.pb", 2),
        (sr_dir / "EDSR_x4.pb", 4),
    ]
    for proto, model_scale in candidates:
        if not proto.is_file():
            continue
        try:
            import cv2

            if hasattr(cv2, "dnn_superres"):
                sr = cv2.dnn_superres.DnnSuperResImpl_create()
                sr.readModel(str(proto))
                sr.setModel("edsr", int(model_scale))
                bgr = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
                out_bgr = sr.upsample(bgr)
                out = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
                if model_scale != scale:
                    out = np.asarray(
                        Image.fromarray(out).resize(
                            (pixels.shape[1] * scale, pixels.shape[0] * scale),
                            Image.Resampling.LANCZOS,
                        ),
                        dtype=np.uint8,
                    )
                return UpscaleResult(
                    pixels=out.astype(np.uint8),
                    backend="opencv_dnn",
                    model_name=proto.name,
                    scale=scale,
                    device="cpu",
                    tile_size=0,
                    input_size=(pixels.shape[1], pixels.shape[0]),
                    output_size=(out.shape[1], out.shape[0]),
                    true_sr=True,
                    message="opencv dnn_superres",
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("[SR] OpenCV EDSR failed: %s", exc)
    return None


def _clear_torch_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def unload_sr_models() -> None:
    """Drop cached SR nets to free RAM/VRAM between pipeline stages."""
    _MODEL_CACHE.clear()
    _clear_torch_cache()


def image_bytes_hash(pixels: NDArray[np.uint8]) -> str:
    return hashlib.sha1(pixels.tobytes()).hexdigest()[:16]
