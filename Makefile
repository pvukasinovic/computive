.PHONY: install test lint clean

install:
	pip install -e ".[dev]"

test:
	python -m pytest tests/ -v

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

format:
	ruff check --fix src/ tests/
	ruff format src/ tests/

rtl-lint:
	verilator --lint-only --language 1800-2017 -Wall \
		-Irtl/include \
		rtl/compute/*.sv rtl/memory/*.sv rtl/layer/*.sv rtl/interface/*.sv

clean:
	rm -rf build/ dist/ *.egg-info src/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
