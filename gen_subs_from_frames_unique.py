# gen_subs_from_frames_unique.py
from __future__ import annotations
import re
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class FrameEvent:
    index: int           # índice 1-based para el subtítulo
    frame_idx: int       # número de frame a partir del nombre
    fname: str
    start: float
    end: float


def sec_to_ass(t: float) -> str:
    if t < 0:
        t = 0.0
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    cs = int(round((t - int(t)) * 100))  # centésimas
    if cs == 100:
        cs = 0
        s += 1
        if s == 60:
            s = 0
            m += 1
            if m == 60:
                m = 0
                h += 1
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


def sec_to_srt(t: float) -> str:
    if t < 0:
        t = 0.0
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int(round((t - int(t)) * 1000))
    if ms == 1000:
        ms = 0
        s += 1
        if s == 60:
            s = 0
            m += 1
            if m == 60:
                m = 0
                h += 1
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def run_ffprobe_times(video: Path) -> List[float]:
    cmd = [
        "ffprobe",
        "-hide_banner",
        "-loglevel", "error",
        "-select_streams", "v:0",
        "-show_frames",
        "-show_entries", "frame=pkt_pts_time,best_effort_timestamp_time",
        "-of", "csv=p=0",
        str(video)
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {r.stderr}")

    times: List[float] = []
    for line in r.stdout.splitlines():
        # típico: "frame,0.033333,0.033333" o "frame,0.033333"
        parts = [p for p in line.strip().split(",") if p]
        if not parts:
            continue
        for p in reversed(parts):
            try:
                t = float(p)
                times.append(t)
                break
            except ValueError:
                continue
    return times


def run_ffprobe_duration(video: Path) -> Optional[float]:
    cmd = [
        "ffprobe",
        "-hide_banner",
        "-loglevel", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video)
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return None
    try:
        return float(r.stdout.strip())
    except ValueError:
        return None


def parse_frame_index(path: Path) -> int:
    # asume algo tipo frame_000123.png -> 123
    m = re.search(r"(\d+)", path.stem)
    if not m:
        raise ValueError(f"Cannot parse frame index from {path.name}")
    return int(m.group(1))


def build_events(video: Path, frames_all_dir: Path, frames_unique_dir: Path) -> List[FrameEvent]:
    all_frames = sorted(
        [p for p in frames_all_dir.iterdir()
         if p.suffix.lower() in (".png", ".jpg", ".jpeg")]
    )
    if not all_frames:
        raise RuntimeError(f"No frames in {frames_all_dir}")

    unique_frames = sorted(
        [p for p in frames_unique_dir.iterdir()
         if p.suffix.lower() in (".png", ".jpg", ".jpeg")]
    )
    if not unique_frames:
        raise RuntimeError(f"No frames in {frames_unique_dir}")

    print(f"Total frames_all: {len(all_frames)}, frames_unique: {len(unique_frames)}")

    times = run_ffprobe_times(video)
    if len(times) < len(all_frames):
        print(f"Warning: ffprobe returned {len(times)} frames, "
              f"but frames_all has {len(all_frames)}. Truncating.")

    times = times[:len(all_frames)]
    idx_to_time = {i + 1: times[i] for i in range(len(times))}

    unique_info: List[Tuple[int, Path, float]] = []
    for p in unique_frames:
        idx = parse_frame_index(p)
        t = idx_to_time.get(idx)
        if t is None:
            raise KeyError(f"No timestamp for frame index {idx} (file {p.name})")
        unique_info.append((idx, p, t))

    # ordenar por tiempo, por las dudas
    unique_info.sort(key=lambda x: x[2])

    # gap típico entre frames
    gaps = [unique_info[i + 1][2] - unique_info[i][2]
            for i in range(len(unique_info) - 1)]
    median_gap = sorted(gaps)[len(gaps) // 2] if gaps else 2.0

    dur_video = run_ffprobe_duration(video)

    events: List[FrameEvent] = []
    for i, (idx, p, t) in enumerate(unique_info):
        if i < len(unique_info) - 1:
            end = unique_info[i + 1][2]
        else:
            if dur_video is not None:
                # extender hasta el final del video, asegurando al menos 0.1s
                end = max(t + 0.1, min(dur_video, t + median_gap))
            else:
                end = t + median_gap

        if end <= t:
            end = t + 0.1

        events.append(
            FrameEvent(
                index=i + 1,
                frame_idx=idx,
                fname=p.name,
                start=t,
                end=end,
            )
        )

    return events


def write_ass(path: Path, events: List[FrameEvent], title: str = "dedupe_phash_manual"):
    with path.open("w", encoding="utf-8") as f:
        f.write("[Script Info]\n")
        f.write(f"Title: {title}\n")
        f.write("ScriptType: v4.00+\n")
        f.write("Collisions: Normal\n")
        f.write("PlayResX: 1920\n")
        f.write("PlayResY: 1080\n")
        f.write("Timer: 100.0000\n")

        f.write("\n[V4+ Styles]\n")
        f.write(
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
            "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
            "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
            "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        )
        f.write(
            "Style: Default,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,"
            "&H00000000,-1,0,0,0,100,100,0,0,1,2,0,2,10,10,10,1\n"
        )

        f.write("\n[Events]\n")
        f.write("Format: Layer, Start, End, Style, Name, MarginL, MarginR, "
                "MarginV, Effect, Text\n")

        for ev in events:
            start = sec_to_ass(ev.start)
            end = sec_to_ass(ev.end)
            text = ev.fname  # placeholder: nombre del frame
            line = f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}\n"
            f.write(line)


def write_srt(path: Path, events: List[FrameEvent]):
    with path.open("w", encoding="utf-8") as f:
        for ev in events:
            start = sec_to_srt(ev.start)
            end = sec_to_srt(ev.end)
            f.write(f"{ev.index}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"{ev.fname}\n\n")


def main():
    import argparse

    ap = argparse.ArgumentParser(
        description="Genera .ass y .srt a partir de frames_unique de dedupe_phash_manual.py"
    )
    ap.add_argument(
        "-i", "--input", required=True,
        help="Video original (el mismo que usaste en dedupe_phash_manual.py)",
    )
    ap.add_argument(
        "--workdir", default="work_frames_manual",
        help="Carpeta de trabajo con frames_all/ y frames_unique/",
    )
    ap.add_argument(
        "--frames-all", default=None,
        help="Override carpeta frames_all (por defecto workdir/frames_all)",
    )
    ap.add_argument(
        "--frames-unique", default=None,
        help="Override carpeta frames_unique (por defecto workdir/frames_unique)",
    )
    ap.add_argument(
        "-o", "--output-base", default=None,
        help="Ruta base para los .ass y .srt (sin extensión)",
    )
    args = ap.parse_args()

    video = Path(args.input)
    if not video.exists():
        raise SystemExit(f"Video no encontrado: {video}")

    work = Path(args.workdir)

    frames_all_dir = Path(args.frames_all) if args.frames_all else work / "frames_all"
    frames_unique_dir = (
        Path(args.frames_unique) if args.frames_unique else work / "frames_unique"
    )

    if not frames_all_dir.exists():
        raise SystemExit(f"No existe frames_all: {frames_all_dir}")
    if not frames_unique_dir.exists():
        raise SystemExit(f"No existe frames_unique: {frames_unique_dir}")

    events = build_events(video, frames_all_dir, frames_unique_dir)

    if not args.output_base:
        out_base = video.with_suffix("").name + "_dedup"
        out_base_path = work / out_base
    else:
        out_base_path = Path(args.output_base)

    ass_path = out_base_path.with_suffix(".ass")
    srt_path = out_base_path.with_suffix(".srt")

    ass_path.parent.mkdir(parents=True, exist_ok=True)

    write_ass(ass_path, events)
    write_srt(srt_path, events)

    print(f"Escrito: {ass_path}")
    print(f"Escrito: {srt_path}")


if __name__ == "__main__":
    main()
