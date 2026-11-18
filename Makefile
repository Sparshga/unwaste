.PHONY: setup data inspect test demo synthetic serve
setup:      ## create venv and install
	python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
data:       ## download FreshRetailNet-50K
	./scripts/download_data.sh
inspect:    ## show dataset columns
	python scripts/inspect_data.py
test:       ## unit + smoke tests
	python -m pytest -q
demo:       ## full run on real data, 1 city, 20 stores, 40 items, with robustness sweep
	python -m unwaste.pipeline --stores 20 --products 40 --sweep
synthetic:  ## same pipeline on stand-in data (no download needed)
	python -m unwaste.pipeline --synthetic --stores 12 --products 20 --sweep
serve:      ## open the manager screen at http://localhost:8000
	uvicorn unwaste.api:app --port 8000
