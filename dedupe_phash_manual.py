# dedupe_phash_manual.py
"""
Dedupe de frames sin usar imagehash.
- Extrae frames con ffmpeg (si no existen).
- Calcula pHash manual (DCT).
- Compara Hamming entre hashes.
- Opcional: verifica con SSIM (scikit-image + opencv).
Python 3.10, Windows-friendly.

Salidas:
- frames_unique: set global de frames únicos (canonical)
- segments.csv: ocurrencias en el tiempo (start/end idx) apuntando al canonical name
"""

import sys
import re
import csv
import subprocess
from pathlib import Path
from PIL import Image
import numpy as np
import shutil
from tqdm import tqdm

# SSIM opcional
try:
    import cv2
    from skimage.metrics import structural_similarity as ssim
    HAS_SSIM = True
except Exception:
    HAS_SSIM = False

def extract_frames(ffmpeg_path, input_file, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / 'frame_%06d.png')
    cmd = [ffmpeg_path, '-y', '-i', str(input_file), '-vsync', 'vfr', pattern]
    print('Running:', ' '.join(cmd))
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        print('ffmpeg failed. stderr:\n', r.stderr.decode(errors='ignore'))
        raise RuntimeError('ffmpeg error')

def phash_image(path_or_pil, hash_size=16, highfreq_factor=4):
    """
    Compute pHash (DCT) returning a numpy uint8 array of bits.
    hash_size: width/height of low-frequency block (e.g. 16 -> 256 bits)
    highfreq_factor: factor to increase pre-dct size (commonly 4)
    """
    if isinstance(path_or_pil, (str, Path)):
        img = Image.open(path_or_pil).convert('L')
    else:
        img = path_or_pil.convert('L')

    img_size = hash_size * highfreq_factor
    img = img.resize((img_size, img_size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32)
    # compute DCT (use numpy fft's dct via cv2 or implement dct via scipy if available)
    try:
        import cv2  # noqa
        dct = cv2.dct(arr)
    except Exception:
        # fallback: simple DCT via numpy (not as efficient but works)
        # 2D DCT via separable 1D using np.fft (approx)
        def dct_1d(x):
            return np.real(np.fft.fft(np.concatenate([x, x[::-1]]))[:x.shape[0]])
        # naive separable (slow); prefer cv2 if disponible
        dct = np.zeros_like(arr)
        for i in range(arr.shape[0]):
            dct[i, :] = dct_1d(arr[i, :])
        for j in range(arr.shape[1]):
            dct[:, j] = dct_1d(dct[:, j])

    # take top-left low frequencies
    dctlow = dct[:hash_size, :hash_size]
    med = np.median(dctlow)
    bits = (dctlow > med).astype(np.uint8).flatten()  # array of 0/1
    return bits  # shape hash_size*hash_size


def hamming_distance_bits(a_bits, b_bits):
    return int(np.count_nonzero(a_bits != b_bits))


def gray_for_ssim(path: Path):
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return im


def parse_frame_index_from_name(path: Path) -> int:
    m = re.search(r'(\d+)', path.stem)
    if not m:
        raise ValueError(f'Cannot parse frame index from {path.name}')
    return int(m.group(1))


def ssim_ok(cur_gray, prev_gray, thresh: float) -> tuple[bool, float]:
    # Devuelve (pasa, score)
    if cur_gray is None or prev_gray is None:
        return True, 1.0

    if cur_gray.shape != prev_gray.shape:
        hmin = min(cur_gray.shape[0], prev_gray.shape[0])
        wmin = min(cur_gray.shape[1], prev_gray.shape[1])
        cur_r = cv2.resize(cur_gray, (wmin, hmin))
        prev_r = cv2.resize(prev_gray, (wmin, hmin))
    else:
        cur_r, prev_r = cur_gray, prev_gray

    score = ssim(prev_r, cur_r)
    return (score >= thresh), float(score)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--input', required=True)
    ap.add_argument('--ffmpeg', default='ffmpeg')
    ap.add_argument('--workdir', default='work_frames_manual')
    ap.add_argument('--skip_extract', action='store_true')
    ap.add_argument('--hash_size', type=int, default=16)
    ap.add_argument('--hf', type=int, default=4)
    ap.add_argument('--hash_thresh', type=int, default=20)
    ap.add_argument('--use_ssim', action='store_true')
    ap.add_argument('--ssim_thresh', type=float, default=0.92)
    ap.add_argument('--max_frames', type=int, default=0)
    args = ap.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        print('Input not found:', inp)
        sys.exit(1)

    if args.use_ssim and not HAS_SSIM:
        print("Warning: pediste --use_ssim pero no esta disponible. Se usa solo pHash.")

    work = Path(args.workdir)
    frames_all = work / 'frames_all'
    frames_unique = work / 'frames_unique'
    frames_dup = work / 'duplicates'
    frames_all.mkdir(parents=True, exist_ok=True)
    frames_unique.mkdir(parents=True, exist_ok=True)
    frames_dup.mkdir(parents=True, exist_ok=True)

    if not args.skip_extract:
        print('Extracting frames...')
        extract_frames(args.ffmpeg, inp, frames_all)
    else:
        print('Skipping extraction, using', frames_all)

    files = sorted([p for p in frames_all.iterdir() if p.suffix.lower() in ('.png', '.jpg', '.jpeg')])
    if args.max_frames > 0:
        files = files[:args.max_frames]
    if not files:
        print('No frames found in', frames_all)
        sys.exit(1)

    # Global dedup
    seen_hashes: list[np.ndarray] = []
    seen_names: list[str] = []
    seen_gray: list[object] = []  # np.ndarray si SSIM, sino vacío

    saved_unique = 0

    # Segments por runs (en orden temporal) apuntando a canonical name
    segments: list[dict] = []
    cur_seg: dict | None = None

    print(
        'Processing', len(files),
        'frames; hash_size=', args.hash_size,
        'hf=', args.hf,
        'thresh=', args.hash_thresh,
        'use_ssim=', args.use_ssim and HAS_SSIM
    )

    for p in tqdm(files):
        try:
            frame_idx = parse_frame_index_from_name(p)
            bits = phash_image(p, hash_size=args.hash_size, highfreq_factor=args.hf)
        except Exception as e:
            print('phash error on', p, e)
            continue

        # Buscar match global (mejor candidato)
        match_j = None
        best_dist = None

        for j, sh in enumerate(seen_hashes):
            dist = hamming_distance_bits(bits, sh)
            if dist <= args.hash_thresh:
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    match_j = j

        canonical_name = None
        is_dup = False

        if match_j is not None:
            # Opcional SSIM para validar el match elegido
            if args.use_ssim and HAS_SSIM:
                try:
                    cur_gray = gray_for_ssim(p)
                    ok, _score = ssim_ok(cur_gray, seen_gray[match_j], args.ssim_thresh)
                    if ok:
                        is_dup = True
                        canonical_name = seen_names[match_j]
                except Exception:
                    # si SSIM falla, lo consideramos dup para no explotar
                    is_dup = True
                    canonical_name = seen_names[match_j]
            else:
                is_dup = True
                canonical_name = seen_names[match_j]

        if not is_dup:
            # Nuevo unique global (canonical)
            shutil.copy2(p, frames_unique / p.name)
            seen_hashes.append(bits)
            seen_names.append(p.name)
            if args.use_ssim and HAS_SSIM:
                try:
                    seen_gray.append(gray_for_ssim(frames_unique / p.name))
                except Exception:
                    seen_gray.append(None)
            saved_unique += 1
            canonical_name = p.name
        else:
            # Duplicate global: opcional guardarlo en duplicates para debug
            shutil.copy2(p, frames_dup / p.name)

        # --- Segments por run contiguo en el tiempo (usando canonical_name) ---
        if cur_seg is None:
            cur_seg = {
                'start_idx': frame_idx,
                'start_name': canonical_name,
                'end_idx': frame_idx,
                'end_name': canonical_name,
            }
        else:
            if canonical_name == cur_seg['start_name']:
                cur_seg['end_idx'] = frame_idx
                cur_seg['end_name'] = canonical_name
            else:
                segments.append(cur_seg)
                cur_seg = {
                    'start_idx': frame_idx,
                    'start_name': canonical_name,
                    'end_idx': frame_idx,
                    'end_name': canonical_name,
                }

    if cur_seg is not None:
        segments.append(cur_seg)

    # escribir segments.csv
    seg_path = work / 'segments.csv'
    with seg_path.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['start_idx', 'start_name', 'end_idx', 'end_name'])
        for s in segments:
            w.writerow([s['start_idx'], s['start_name'], s['end_idx'], s['end_name']])

    print('Done. Saved unique frames:', saved_unique)
    print('Unique dir:', frames_unique.resolve())
    print('Duplicates dir:', frames_dup.resolve())
    print('Segments CSV:', seg_path.resolve())
    print('Segments count:', len(segments))


if __name__ == '__main__':
    main()
