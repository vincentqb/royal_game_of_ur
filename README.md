# Royal Game of Ur

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh

uv run python ur/play_one.py
uv run python ur/play_many.py
uv run python ur/play_learn.py
```

Development

```
uvx pre-commit install
uvx pre-commit autoupdate
uvx ruff format .
uvx ruff check --fix .
uvx ty check
uvx codespell

uv venv
source .venv/bin/activate
uv pip install .
```
