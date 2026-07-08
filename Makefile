# Ferramentas: make <alvo>
PY ?= python3

.PHONY: help install install-manifold run web test lint clean

help:
	@echo "Alvos:"
	@echo "  install           deps core"
	@echo "  install-manifold  + PHATE/UMAP"
	@echo "  run               pipeline (data/*.npz, preview.png, manifold_data.json)"
	@echo "  web               gera data/viewer_webgpu.html a partir do JSON"
	@echo "  test              pytest"
	@echo "  lint              ruff"
	@echo "  clean             remove data/ e caches"

install:
	$(PY) -m pip install -r requirements.txt

install-manifold:
	$(PY) -m pip install -r requirements.txt phate umap-learn

run:
	cd python && $(PY) run_pipeline.py --sessions 180 --grid 192 --out ../data

web:
	cd web && $(PY) build_viewer.py --data ../data/manifold_data.json --out ../data/viewer_webgpu.html

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check python tests || true

clean:
	rm -rf data __pycache__ */__pycache__ */*/__pycache__ .pytest_cache
