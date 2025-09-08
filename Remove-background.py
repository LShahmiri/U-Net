import os
from pathlib import Path
import cv2
import numpy as np

# =========================
# CONFIG — EDIT THESE THREE
# =========================
IMAGES_ROOT = r'/butterflies-images/'     #
MASKS_ROOT  = r'/mask-outputed/'   
OUTPUT_ROOT = r'/removed background/'   

# Choose background mode:
BACKGROUND = "white"   # "white" or "transparent"
# =========================

IMG_EXTS  = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
MASK_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

def find_mask_for(stem: str, mask_dir: Path) -> Path | None:
    """Find a mask file in mask_dir that shares the same stem (name without extension)."""
    for ext in MASK_EXTS:
        p = mask_dir / f"{stem}{ext}"
        if p.exists():
            return p
    # Fallback: case-insensitive search by stem
    stem_lower = stem.lower()
    for p in mask_dir.iterdir():
        if p.is_file() and p.suffix.lower() in MASK_EXTS and p.stem.lower() == stem_lower:
            return p
    return None

def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def load_image(path: Path) -> np.ndarray | None:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"[WARN] Could not read image: {path}")
    return img

def load_mask(path: Path) -> np.ndarray | None:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        print(f"[WARN] Could not read mask: {path}")
    return m

def to_bool_mask(mask_gray: np.ndarray, target_shape_hw: tuple[int, int]) -> np.ndarray:
    # Resize if needed (nearest to preserve binary edges)
    if mask_gray.shape[:2] != target_shape_hw:
        mask_gray = cv2.resize(mask_gray, (target_shape_hw[1], target_shape_hw[0]), interpolation=cv2.INTER_NEAREST)
    # Threshold to {0,1}
    _, thresh = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)
    return (thresh // 255).astype(np.uint8)

def apply_mask_color_on_white(img: np.ndarray, mask01: np.ndarray) -> np.ndarray:
    # Ensure 3 channels BGR
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:  # drop existing alpha for this mode
        img = img[:, :, :3]
    mask01 = mask01[..., None]  # (H,W,1)
    white_bg = np.full_like(img, 255)
    return img * mask01 + white_bg * (1 - mask01)

def apply_mask_color_with_alpha(img: np.ndarray, mask01: np.ndarray) -> np.ndarray:
    # Return BGRA with alpha = mask * 255
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        bgr = img[:, :, :3]
    else:
        bgr = img
    alpha = (mask01 * 255).astype(np.uint8)
    return np.dstack([bgr, alpha])

def main():
    images_root = Path(IMAGES_ROOT)
    masks_root  = Path(MASKS_ROOT)
    output_root = Path(OUTPUT_ROOT)

    if not images_root.exists():
        raise SystemExit(f"IMAGES_ROOT not found: {images_root}")
    if not masks_root.exists():
        raise SystemExit(f"MASKS_ROOT not found: {masks_root}")

    img_count = 0
    ok_count = 0
    missing_mask = 0
    failed = 0

    for img_path in images_root.rglob("*"):
        if not img_path.is_file() or img_path.suffix.lower() not in IMG_EXTS:
            continue
        img_count += 1

        # Build relative path and corresponding mask dir/file
        rel = img_path.relative_to(images_root)
        mask_dir = masks_root / rel.parent
        mask_path = find_mask_for(img_path.stem, mask_dir)

        if mask_path is None:
            print(f"[MISS] No mask for: {rel}")
            missing_mask += 1
            continue

        img = load_image(img_path)
        mask_gray = load_mask(mask_path)
        if img is None or mask_gray is None:
            failed += 1
            continue

        mask01 = to_bool_mask(mask_gray, img.shape[:2])

        if BACKGROUND.lower() == "transparent":
            out_img = apply_mask_color_with_alpha(img, mask01)
            # Always save as .png to keep alpha
            out_rel = rel.with_suffix(".png")
        else:  # white
            out_img = apply_mask_color_on_white(img, mask01)
            # Keep original extension
            out_rel = rel

        out_path = output_root / out_rel
        ensure_dir(out_path)

        # Write file
        success = cv2.imwrite(str(out_path), out_img)
        if not success:
            print(f"[FAIL] Could not write: {out_rel}")
            failed += 1
        else:
            ok_count += 1
            # Optional: progress print every 100
            if ok_count % 100 == 0:
                print(f"[OK] {ok_count} saved...")

    print("\n===== SUMMARY =====")
    print(f"Total images scanned : {img_count}")
    print(f"Segmented & saved    : {ok_count}")
    print(f"Missing masks        : {missing_mask}")
    print(f"Failed saves/reads   : {failed}")

if __name__ == "__main__":
    main()
