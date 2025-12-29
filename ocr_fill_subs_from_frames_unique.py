# ocr_fill_subs_from_frames_unique.py
from __future__ import annotations
import argparse
from pathlib import Path

from PIL import Image

try:
    import pytesseract
except ImportError:
    pytesseract = None

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, *args, **kwargs):
        return x


def ocr_frames(frames_dir: Path, lang: str) -> list[str]:
    exts = (".png", ".jpg", ".jpeg")
    frames = sorted([p for p in frames_dir.iterdir() if p.suffix.lower() in exts])

    if not frames:
        raise SystemExit(f"No se encontraron imágenes en {frames_dir}")

    if pytesseract is None:
        raise SystemExit("pytesseract no está instalado. Agregalo al entorno.")

    texts: list[str] = []
    for p in tqdm(frames, desc="OCR frames"):
        try:
            img = Image.open(p)
            raw = pytesseract.image_to_string(img, lang=lang)
        except Exception as e:
            print(f"Error OCR en {p}: {e}")
            raw = ""

        # normalizar saltos de línea y limpiar espacios
        raw = raw.replace("\r\n", "\n").replace("\r", "\n")
        lines = [ln.strip() for ln in raw.splitlines()]
        lines = [ln for ln in lines if ln]  # eliminar líneas vacías
        text = "\n".join(lines)
        texts.append(text)

    return texts


def escape_ass_text(text: str) -> str:
    # escapamos lo mínimo necesario y convertimos \n en \N
    text = text.replace("\\", r"\\")
    text = text.replace("{", r"\{").replace("}", r"\}")
    text = text.replace("\n", r"\N")
    return text


def update_ass(ass_in: Path, ass_out: Path, texts: list[str]) -> None:
    if not ass_in.exists():
        print(f"AVISO: no se encontró {ass_in}, se saltea .ass")
        return

    lines = ass_in.read_text(encoding="utf-8", errors="ignore").splitlines(keepends=True)

    dialogue_idx: list[int] = []
    for i, line in enumerate(lines):
        if line.lstrip().startswith("Dialogue:"):
            dialogue_idx.append(i)

    if not dialogue_idx:
        print(f"AVISO: no se encontraron líneas 'Dialogue:' en {ass_in}")
        return

    n = min(len(dialogue_idx), len(texts))
    print(f"Actualizando {n} eventos en .ass (de {len(dialogue_idx)} diálogos)")

    for k in range(n):
        i = dialogue_idx[k]
        original_line = lines[i].rstrip("\n").rstrip("\r")

        parts = original_line.split(",", 9)  # 10 campos máx.: 0..9
        if len(parts) < 10:
            # formato raro, no tocamos
            continue

        new_text = texts[k]
        if not new_text:
            # si OCR no encontró nada, dejamos el texto original
            continue

        parts[9] = escape_ass_text(new_text)
        new_line = ",".join(parts) + "\n"
        lines[i] = new_line

    ass_out.write_text("".join(lines), encoding="utf-8")
    print(f"Escrito .ass OCR: {ass_out}")


def update_srt(srt_in: Path, srt_out: Path, texts: list[str]) -> None:
    if not srt_in.exists():
        print(f"AVISO: no se encontró {srt_in}, se saltea .srt")
        return

    content = srt_in.read_text(encoding="utf-8", errors="ignore")

    # separar bloques por líneas en blanco
    blocks = [b for b in content.split("\n\n") if b.strip()]
    new_blocks: list[str] = []

    text_idx = 0
    for block in blocks:
        lines = block.splitlines()
        if len(lines) < 2:
            new_blocks.append(block)
            continue

        if "-->" not in lines[1]:
            # no parece un bloque de subtítulo estándar
            new_blocks.append(block)
            continue

        if text_idx >= len(texts):
            new_blocks.append(block)
            continue

        new_text = texts[text_idx]
        text_idx += 1

        if new_text:
            text_lines = new_text.split("\n")
            new_block = "\n".join([lines[0], lines[1]] + text_lines)
        else:
            # sin texto OCR, mantenemos el bloque como estaba
            new_block = block

        new_blocks.append(new_block)

    new_content = "\n\n".join(new_blocks) + "\n"
    srt_out.write_text(new_content, encoding="utf-8")
    print(f"Escrito .srt OCR: {srt_out}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Hace OCR a frames_unique y reemplaza el texto de los .ass/.srt "
            "generados por gen_subs_from_frames_unique.py"
        )
    )
    ap.add_argument(
        "-i",
        "--input",
        help="Video original (solo para deducir el nombre base si no usás --subs-base)",
    )
    ap.add_argument(
        "--workdir",
        default="work_frames_manual",
        help="Carpeta de trabajo (por defecto work_frames_manual)",
    )
    ap.add_argument(
        "--subs-base",
        default=None,
        help="Ruta base sin extensión de los subtítulos ya generados "
             "(ej: work_frames_manual/video_dedup)",
    )
    ap.add_argument(
        "--frames-dir",
        default=None,
        help="Carpeta con frames únicos (por defecto workdir/frames_unique)",
    )
    ap.add_argument(
        "--lang",
        default="eng",
        help="Código de idioma para Tesseract (ej: spa, eng, spa+eng)",
    )
    args = ap.parse_args()

    workdir = Path(args.workdir)

    # frames_unique
    if args.frames_dir:
        frames_dir = Path(args.frames_dir)
    else:
        frames_dir = workdir / "frames_unique"

    if not frames_dir.exists():
        raise SystemExit(f"No existe frames_dir: {frames_dir}")

    # base de subtítulos
    if args.subs_base:
        base = Path(args.subs_base)
    else:
        if not args.input:
            raise SystemExit(
                "Falta --subs-base o -i/--input para deducir la ruta base de subtítulos."
            )
        video = Path(args.input)
        base_name = video.with_suffix("").name + "_dedup"
        base = workdir / base_name

    ass_in = base.with_suffix(".ass")
    srt_in = base.with_suffix(".srt")

    # salidas con sufijo _ocr
    base_ocr = base.with_name(base.name + "_ocr")
    ass_out = base_ocr.with_suffix(".ass")
    srt_out = base_ocr.with_suffix(".srt")

    print(f"frames_dir: {frames_dir}")
    print(f"Subs base: {base}")
    print(f"Entrada .ass: {ass_in}")
    print(f"Entrada .srt: {srt_in}")
    print(f"Salida .ass OCR: {ass_out}")
    print(f"Salida .srt OCR: {srt_out}")
    print(f"Idioma Tesseract: {args.lang}")

    texts = ocr_frames(frames_dir, args.lang)

    update_ass(ass_in, ass_out, texts)
    update_srt(srt_in, srt_out, texts)


if __name__ == "__main__":
    main()
