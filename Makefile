# Makefile para pipelines Python (Windows / Unix)
# Uso: make <target> [VAR=value ...]
# Ejemplo PowerShell:
#   make dedupe VIDEO='mi_video.mkv' FFMPEG='C:/ffmpeg/bin/ffmpeg.exe'
#   make dedupe USE_SSIM=1 MAX_FRAMES=500

# ---------- Configurables ----------
VIDEO ?= mi_video.mkv
WORKDIR ?= work_frames_manual
# FFMPEG vacío por defecto -> el script usa 'ffmpeg' del PATH
FFMPEG ?=
LANG ?= spa+eng
SSIM_THRESH ?= 0.92
MAX_FRAMES ?=
USE_SSIM ?=
# -----------------------------------

# Detect venv python path según sistema
ifeq ($(OS),Windows_NT)
VENV_PY := .venv/Scripts/python.exe
else
VENV_PY := .venv/bin/python
endif

# Si no existe .venv todavía, usar python del sistema para crearla
PY := python

.PHONY: help venv install run run_pose dedupe dedupe_ssim test500 gen_subs ocr clean start

.DEFAULT_GOAL := help

help:
	$(info Targets disponibles:)
	$(info   make venv           - crea .venv e instala requirements.txt)
	$(info   make install        - alias de venv)
	$(info   make dedupe         - corre dedupe_phash_manual.py (ver flags abajo))
	$(info   make gen_subs       - corre gen_subs_from_frames_unique.py)
	$(info   make ocr            - corre ocr_fill_subs_from_frames_unique.py)
	$(info   make start          - corre dedupe - gen_subs - ocr en cadena)
	$(info   make clean          - borra .venv, workdir y artefactos)
	$(info )
	$(info Variables que podés pasar:)
	$(info   VIDEO, FFMPEG, WORKDIR, LANG, USE_SSIM=1, SSIM_THRESH, MAX_FRAMES)
	$(info   (si FFMPEG se omite, se usa "ffmpeg" del PATH))
	@:

venv:
	@echo 'Creando virtualenv (.venv) y instalando requirements...'
	$(PY) -m venv .venv
	$(VENV_PY) -m pip install --upgrade pip
	$(VENV_PY) -m pip install -r requirements.txt

install: venv

# Flags condicionales
SSIM_FLAGS := $(if $(USE_SSIM),--use_ssim --ssim_thresh $(SSIM_THRESH),)
MAXFRAMES_FLAG := $(if $(MAX_FRAMES),--max_frames $(MAX_FRAMES),)
FFMPEG_FLAG := $(if $(FFMPEG),--ffmpeg "$(FFMPEG)",)

dedupe:
	@echo 'Corriendo dedupe (phash)...'
	$(VENV_PY) dedupe_phash_manual.py -i "$(VIDEO)" $(FFMPEG_FLAG) --workdir $(WORKDIR) $(SSIM_FLAGS) $(MAXFRAMES_FLAG)

gen_subs:
	@echo 'Generando .ass y .str desde frames únicos...'
	$(VENV_PY) gen_subs_from_frames_unique.py -i "$(VIDEO)" --workdir $(WORKDIR)

ocr:
	@echo 'Haciendo OCR y llenando subtítulos...'
	$(VENV_PY) ocr_fill_subs_from_frames_unique.py -i "$(VIDEO)" --workdir $(WORKDIR) --lang $(LANG)

start:
	$(MAKE) dedupe
	$(MAKE) gen_subs
	$(MAKE) ocr
	@echo 'Pipeline completa (dedupe -^> gen_subs -^> ocr).'

# Alias para casos comunes
dedupe_ssim:
	$(MAKE) dedupe USE_SSIM=1

test500:
	$(MAKE) dedupe MAX_FRAMES=500

clean:
	@echo 'Limpiando .venv, workdir y caches...'
	-@rm -rf .venv $(WORKDIR) __pycache__ *.pyc || true
