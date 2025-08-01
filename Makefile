format: ruff
	ruff format
	ruff check --fix
	ruff check --select I --fix

ruff:
	pip install ruff
