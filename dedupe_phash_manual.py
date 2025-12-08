# dedupe_phash_manual.py
"""
Dedupe de frames sin usar imagehash.
- Extrae frames con ffmpeg (si no existen).
- Calcula pHash manual (DCT).
- Compara Hamming entre hashes.
- Opcional: verifica con SSIM (scikit-image + opencv).
Python 3.10, Windows-friendly.
"""

import os
import sys
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

def extract_frames(ffmpeg_path, input_file, out_dir):
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
        import cv2
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

def bits_to_hexstr(bits):
    s = ''.join(str(int(b)) for b in bits)
    h = hex(int(s, 2))[2:].rjust(len(s)//4, '0')
    return h

def hamming_distance_bits(a_bits, b_bits):
    # a_bits, b_bits are uint8 arrays
    return int(np.count_nonzero(a_bits != b_bits))

def gray_for_ssim(path):
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return im

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('-i','--input', required=True)
    ap.add_argument('--ffmpeg', default='ffmpeg')
    ap.add_argument('--workdir', default='work_frames_manual')
    ap.add_argument('--skip_extract', action='store_true')
    ap.add_argument('--hash_size', type=int, default=16)            # 16 -> 256-bit
    ap.add_argument('--hf', type=int, default=4)                    # highfreq factor
    ap.add_argument('--hash_thresh', type=int, default=20)          # hamming threshold
    ap.add_argument('--use_ssim', action='store_true')
    ap.add_argument('--ssim_thresh', type=float, default=0.92)
    ap.add_argument('--max_frames', type=int, default=0)
    args = ap.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        print('Input not found:', inp); sys.exit(1)

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

    files = sorted([p for p in frames_all.iterdir() if p.suffix.lower() in ('.png','.jpg','.jpeg')])
    if args.max_frames > 0:
        files = files[:args.max_frames]
    if not files:
        print('No frames found in', frames_all); sys.exit(1)

    seen_hashes = []
    seen_gray = []  # for SSIM
    saved = 0

    print('Processing', len(files), 'frames; hash_size=', args.hash_size, 'hf=', args.hf, 'thresh=', args.hash_thresh)
    for p in tqdm(files):
        try:
            bits = phash_image(p, hash_size=args.hash_size, highfreq_factor=args.hf)
        except Exception as e:
            print('phash error on', p, e); continue

        is_dup = False
        for idx, sh in enumerate(seen_hashes):
            dist = hamming_distance_bits(bits, sh)
            if dist <= args.hash_thresh:
                # possible duplicate: optional SSIM verify
                if args.use_ssim and HAS_SSIM:
                    try:
                        cur = gray_for_ssim(p)
                        prev = seen_gray[idx]
                        # resize to same
                        if cur.shape != prev.shape:
                            hmin = min(cur.shape[0], prev.shape[0]); wmin = min(cur.shape[1], prev.shape[1])
                            cur_r = cv2.resize(cur, (wmin, hmin))
                            prev_r = cv2.resize(prev, (wmin, hmin))
                        else:
                            cur_r, prev_r = cur, prev
                        score = ssim(prev_r, cur_r)
                        if score >= args.ssim_thresh:
                            is_dup = True
                            break
                        else:
                            continue
                    except Exception:
                        is_dup = True
                        break
                else:
                    is_dup = True
                    break

        if is_dup:
            # move duplicate
            shutil.copy2(p, frames_dup / p.name)
            continue

        # unique -> copy
        shutil.copy2(p, frames_unique / p.name)
        seen_hashes.append(bits)
        if args.use_ssim and HAS_SSIM:
            seen_gray.append(gray_for_ssim(frames_unique / p.name))
        saved += 1

    print('Done. Saved unique frames:', saved)
    print('Unique dir:', frames_unique.resolve())
    print('Duplicates dir:', frames_dup.resolve())

if __name__ == '__main__':
    main()
