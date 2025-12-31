# gen_subs_from_frames_unique.py
from __future__ import annotations
import re
import csv
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict


@dataclass
class FrameEvent:
    index: int           # índice 1-based para el subtítulo
    frame_idx: int       # número de frame a partir del nombre
    fname: str
    start: float
    end: float
    end_frame_idx: Optional[int] = None
    end_fname: Optional[str] = None


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
        str(video),
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
                times.append(float(p))
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


def read_segments_csv(path: Path) -> list[tuple[int, str, int, str]]:
    segs: list[tuple[int, str, int, str]] = []
    with path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            segs.append((
                int(row["start_idx"]), row["start_name"],
                int(row["end_idx"]), row["end_name"],
            ))
    return segs


def build_idx_to_time(video: Path, frames_all: list[Path]) -> tuple[Dict[int, float], list[int], float, Optional[float]]:
    frames_all_sorted = sorted(frames_all, key=parse_frame_index)

    times = run_ffprobe_times(video)
    if len(times) < len(frames_all_sorted):
        print(
            f"Warning: ffprobe returned {len(times)} frames, "
            f"but frames_all has {len(frames_all_sorted)}. Truncating frames_all."
        )
        frames_all_sorted = frames_all_sorted[:len(times)]
    times = times[:len(frames_all_sorted)]

    idx_to_time: Dict[int, float] = {}
    for p, t in zip(frames_all_sorted, times):
        idx_to_time[parse_frame_index(p)] = t

    keys_sorted = sorted(idx_to_time.keys())

    gaps = [times[i + 1] - times[i] for i in range(len(times) - 1)]
    gaps = [g for g in gaps if g > 0]
    median_gap = sorted(gaps)[len(gaps) // 2] if gaps else 2.0

    dur_video = run_ffprobe_duration(video)
    return idx_to_time, keys_sorted, median_gap, dur_video


def next_time_after(end_idx: int, idx_to_time: Dict[int, float], keys_sorted: list[int]) -> Optional[float]:
    # Primer timestamp con idx > end_idx (no asume end_idx+1)
    lo = 0
    hi = len(keys_sorted)
    while lo < hi:
        mid = (lo + hi) // 2
        if keys_sorted[mid] <= end_idx:
            lo = mid + 1
        else:
            hi = mid
    if lo >= len(keys_sorted):
        return None
    return idx_to_time[keys_sorted[lo]]


def build_overrides_from_segments(
    segs: list[tuple[int, str, int, str]],
    unique_name_set: set[str],
) -> Dict[str, tuple[int, int]]:
    """
    Devuelve overrides por nombre de archivo (start_name) SOLO si existe en frames_unique/.
    Si hay varias filas para el mismo nombre (reapariciones), toma la de menor start_idx.
    """
    overrides: Dict[str, tuple[int, int]] = {}
    for s_idx, s_name, e_idx, _e_name in segs:
        if s_name not in unique_name_set:
            continue

        # normaliza rango
        if e_idx < s_idx:
            s_idx, e_idx = e_idx, s_idx

        if s_name not in overrides:
            overrides[s_name] = (s_idx, e_idx)
        else:
            prev_s, _prev_e = overrides[s_name]
            if s_idx < prev_s:
                overrides[s_name] = (s_idx, e_idx)

    return overrides


def build_events(video: Path, frames_all_dir: Path, frames_unique_dir: Path) -> List[FrameEvent]:
    all_frames = [
        p for p in frames_all_dir.iterdir()
        if p.suffix.lower() in (".png", ".jpg", ".jpeg")
    ]
    if not all_frames:
        raise RuntimeError(f"No frames in {frames_all_dir}")

    unique_frames = sorted(
        [p for p in frames_unique_dir.iterdir()
         if p.suffix.lower() in (".png", ".jpg", ".jpeg")],
        key=parse_frame_index,
    )
    if not unique_frames:
        raise RuntimeError(f"No frames in {frames_unique_dir}")

    unique_name_set = {p.name for p in unique_frames}

    print(f"Total frames_all: {len(all_frames)}, frames_unique: {len(unique_frames)}")

    idx_to_time, keys_sorted, median_gap, dur_video = build_idx_to_time(video, all_frames)

    # overrides desde segments.csv (si existe)
    seg_path = frames_unique_dir.parent / "segments.csv"
    overrides: Dict[str, tuple[int, int]] = {}
    if seg_path.exists():
        print(f"Usando overrides de: {seg_path}")
        segs = read_segments_csv(seg_path)
        overrides = build_overrides_from_segments(segs, unique_name_set)
        print(f"Overrides aplicables: {len(overrides)} / {len(unique_frames)}")

    # base: eventos salen SOLO de frames_unique (en orden por índice del nombre)
    base_info: list[tuple[str, int]] = []
    for p in unique_frames:
        base_info.append((p.name, parse_frame_index(p)))

    # construir start/end por evento
    # start/end en "índices de frame" (luego lo pasamos a tiempo)
    items: list[tuple[str, int, int]] = []
    for i, (name, base_idx) in enumerate(base_info):
        if name in overrides:
            s_idx, e_idx = overrides[name]
        else:
            s_idx = base_idx
            # end por defecto: hasta antes del siguiente unique
            if i < len(base_info) - 1:
                next_base_idx = base_info[i + 1][1]
                e_idx = max(s_idx, next_base_idx - 1)
            else:
                e_idx = s_idx

        items.append((name, s_idx, e_idx))

    # ordenar por start_idx (por si editaste algún start_idx)
    items.sort(key=lambda x: (x[1], x[0]))

    events: List[FrameEvent] = []
    for i, (name, s_idx, e_idx) in enumerate(items):
        t_start = idx_to_time.get(s_idx)
        if t_start is None:
            raise KeyError(f"No timestamp for start_idx={s_idx} (name={name})")

        t_end = next_time_after(e_idx, idx_to_time, keys_sorted)
        if t_end is None:
            if dur_video is not None:
                t_end = max(t_start + 0.1, min(dur_video, t_start + median_gap))
            else:
                t_end = t_start + median_gap

        if t_end <= t_start:
            t_end = t_start + 0.1

        events.append(
            FrameEvent(
                index=i + 1,
                frame_idx=s_idx,
                fname=name,          # SOLO nombre del archivo de frames_unique
                start=t_start,
                end=t_end,
                end_frame_idx=e_idx,
                end_fname=name,
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
            f.write(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{ev.fname}\n")


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
        description="Genera .ass y .srt usando SOLO frames_unique; segments.csv solo corrige start/end por nombre"
    )
    ap.add_argument("-i", "--input", required=True, help="Video original")
    ap.add_argument("--workdir", default="work_frames_manual", help="Carpeta de trabajo")
    ap.add_argument("--frames-all", default=None, help="Override frames_all")
    ap.add_argument("--frames-unique", default=None, help="Override frames_unique")
    ap.add_argument("-o", "--output-base", default=None, help="Ruta base sin extension")
    args = ap.parse_args()

    video = Path(args.input)
    if not video.exists():
        raise SystemExit(f"Video no encontrado: {video}")

    work = Path(args.workdir)
    frames_all_dir = Path(args.frames_all) if args.frames_all else work / "frames_all"
    frames_unique_dir = Path(args.frames_unique) if args.frames_unique else work / "frames_unique"

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
