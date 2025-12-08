# tesseract

## Prerequisites

- Python 3.10.0

## Install

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Locating ffmpeg (PowerShell)

```powershell
where.exe ffmpeg
```

## Usage (PowerShell)

1. Extract all frames and perform deduplication (recommended):

```powershell
python dedupe_phash_manual.py -i "my_video.mkv" --ffmpeg "C:\ffmpeg\bin\ffmpeg.exe"
```

2. Run with SSIM verification (safer but slower):

```powershell
python dedupe_phash_manual.py -i "my_video.mkv" --use_ssim --ssim_thresh 0.92
```

3. Test only the first 500 frames (useful for tuning thresholds):

```powershell
python dedupe_phash_manual.py -i "my_video.mkv" --max_frames 500
```

4. Generate .ass and .str files:

```powershell
python gen_subs_from_frames_unique.py -i "my_video.mkv" --workdir work_frames_manual
```

5. Perform OCR and generate subtitles:

```powershell
python ocr_fill_subs_from_frames_unique.py -i "my_video.mkv" --workdir work_frames_manual --lang spa+eng
```
