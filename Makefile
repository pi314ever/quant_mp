format: ruff
	ruff format
	ruff check --fix
	ruff check --select I --fix

ruff:
	pip install ruff

uv-install:
	@if !command -v uv >/dev/null 2>&1; then \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	fi

export_requirements: uv-install
	uv export --format requirements.txt --locked > requirements.txt
