# dedupe_frames.py
"""
Extrae frames (si hace falta) y elimina duplicados visuales dejando solo los esenciales.
Compatible con Windows. Python 3.10.
Instalá: pillow imagehash opencv-python scikit-image tqdm
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from PIL import Image
import imagehash
import shutil
from tqdm import tqdm

# SSIM (opcional)
try:
    import cv2
    from skimage.metrics import structural_similarity as ssim
    HAS_SSIM = True
except Exception:
    HAS_SSIM = False

def run_ffmpeg_extract(input_path, out_dir, ffmpeg_path='ffmpeg'):
    os.makedirs(out_dir, exist_ok=True)
    out_pattern = str(Path(out_dir) / "frame_%06d.png")
    cmd = [ffmpeg_path, '-y', '-i', str(input_path), '-vsync', 'vfr', out_pattern]
    print("Ejecutando:", ' '.join(cmd))
    res = subprocess.run(cmd, capture_output=True)
    if res.returncode != 0:
        print("ffmpeg falló. stdout/stderr:\n", res.stdout.decode(errors='ignore'), res.stderr.decode(errors='ignore'))
        raise RuntimeError("ffmpeg error")
    return out_dir

def compute_phash(path, hash_size=16):
    # hash_size 16 -> 256-bit phash (mejor sensibilidad en subtítulos)
    try:
        img = Image.open(path).convert('L')
        return imagehash.phash(img, hash_size=hash_size)
    except Exception as e:
        print("Error abriendo imagen para hash:", path, e)
        return None

def image_to_gray_cv(path):
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return im

def main():
    ap = argparse.ArgumentParser(description="Extraer y deduplicar frames (pHash [+ opcional SSIM])")
    ap.add_argument('-i','--input', required=True, help="Archivo de video (ruta)")
    ap.add_argument('-w','--workdir', default="work_frames", help="Carpeta de trabajo (frames_all, frames_unique)")
    ap.add_argument('--ffmpeg', default='ffmpeg', help="Ruta a ffmpeg (si no está en PATH)")
    ap.add_argument('--skip_extract', action='store_true', help="Si ya extrajiste frames y están en workdir/frames_all")
    ap.add_argument('--hash_threshold', type=int, default=6, help="Hamming dist máx para considerar 'igual' (pHash). Por defecto 6")
    ap.add_argument('--use_ssim', action='store_true', help="Habilita verificación SSIM en borde (requiere scikit-image + opencv)")
    ap.add_argument('--ssim_threshold', type=float, default=0.92, help="Umbral SSIM (0..1) si use_ssim True")
    ap.add_argument('--max_frames', type=int, default=0, help="Si >0, procesa solo los primeros N frames extraídos (útil para pruebas)")
    ap.add_argument('--keep_duplicates', action='store_true', help="No borra duplicados, los mueve a workdir/duplicates en vez de borrarlos")
    args = ap.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print("No existe el archivo input:", input_path)
        sys.exit(1)

    workdir = Path(args.workdir)
    frames_all = workdir / "frames_all"
    frames_unique = workdir / "frames_unique"
    frames_dup = workdir / "duplicates"

    frames_all.mkdir(parents=True, exist_ok=True)
    frames_unique.mkdir(parents=True, exist_ok=True)
    frames_dup.mkdir(parents=True, exist_ok=True)

    # 1) extraer frames si es necesario
    if not args.skip_extract:
        print("Extrayendo frames a:", frames_all)
        run_ffmpeg_extract(input_path, frames_all, ffmpeg_path=args.ffmpeg)
    else:
        print("Saltando extracción (--skip_extract). Usando frames en:", frames_all)

    # 2) listar y ordenar frames
    all_files = sorted([p for p in frames_all.iterdir() if p.suffix.lower() in ('.png','.jpg','.jpeg')])
    if args.max_frames > 0:
        all_files = all_files[:args.max_frames]

    if not all_files:
        print("No se encontraron frames en", frames_all)
        sys.exit(1)

    # 3) deduplicar usando pHash y opcionalmente SSIM
    saved_hashes = []
    saved_hash_objs = []  # imagehash objects
    saved_cv_gray = []    # imágenes grises guardadas (para SSIM comparaciones)
    saved_paths = []

    hash_thresh = args.hash_threshold
    use_ssim = args.use_ssim and HAS_SSIM
    if args.use_ssim and not HAS_SSIM:
        print("Advertencia: use_ssim solicitado pero opencv/scikit-image no está disponible. Continuo sin SSIM.")
        use_ssim = False

    print(f"Procesando {len(all_files)} frames, hash_threshold={hash_thresh}, use_ssim={use_ssim}")

    for idx, fp in enumerate(tqdm(all_files, desc="Frames")):
        try:
            h = compute_phash(fp)
            if h is None:
                continue
        except Exception as e:
            print("Error hash:", fp, e)
            continue

        is_dup = False
        # comparar con hashes guardados
        for si, sh in enumerate(saved_hash_objs):
            dist = h - sh
            if dist <= hash_thresh:
                # candidato a duplicado visual
                if use_ssim:
                    # calculamos SSIM entre current y saved si es necesario (refina)
                    try:
                        cur_gray = image_to_gray_cv(fp)
                        prev_gray = saved_cv_gray[si]
                        # si tamaños distintos, redimensionar al menor
                        if cur_gray.shape != prev_gray.shape:
                            h_min = min(cur_gray.shape[0], prev_gray.shape[0])
                            w_min = min(cur_gray.shape[1], prev_gray.shape[1])
                            cur_r = cv2.resize(cur_gray, (w_min, h_min), interpolation=cv2.INTER_AREA)
                            prev_r = cv2.resize(prev_gray, (w_min, h_min), interpolation=cv2.INTER_AREA)
                        else:
                            cur_r = cur_gray; prev_r = prev_gray
                        score = ssim(prev_r, cur_r)
                        if score >= args.ssim_threshold:
                            is_dup = True
                            break
                        else:
                            # not considered duplicate by SSIM; continue checking other saved hashes
                            continue
                    except Exception as e:
                        # en caso de fallo ssim, fallback a considerar duplicado por hash
                        print("Warn: SSIM falló:", e)
                        is_dup = True
                        break
                else:
                    # sin SSIM, consideramos duplicado por pHash
                    is_dup = True
                    break

        if is_dup:
            # mover o borrar duplicados según flag
            if args.keep_duplicates:
                target = frames_dup / fp.name
                shutil.copy2(fp, target)
            # opcional: borrar el frame original para ahorrar espacio (comento por seguridad)
            # fp.unlink()
            continue

        # si llegamos acá, es único: lo copiamos a unique y guardamos sus datos
        target = frames_unique / fp.name
        shutil.copy2(fp, target)
        saved_hash_objs.append(h)
        saved_paths.append(str(target))
        if use_ssim:
            saved_cv_gray.append(image_to_gray_cv(target))

    print("Hecho. Frames únicos guardados en:", frames_unique)
    if args.keep_duplicates:
        print("Duplicados copiados a:", frames_dup)

if __name__ == "__main__":
    main()
