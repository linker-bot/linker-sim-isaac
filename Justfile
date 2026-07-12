set shell := ["bash", "-uc"]

alias check := quality

whitespace:
    git diff --check
    git diff --cached --check

format:
    uv run --frozen --extra dev ruff format .

format-check:
    uv run --frozen --extra dev ruff format --check .

lint:
    uv run --frozen --extra dev ruff check .

check-docs:
    uv run --frozen --extra dev python scripts/check_markdown_links.py

test:
    uv run --frozen --extra dev coverage erase
    uv run --frozen --extra dev coverage run -m pytest -q
    uv run --frozen --extra dev coverage report

validate-config:
    uv run --frozen --extra dev python scripts/validate_config.py --runtime-profile default_single_scene
    uv run --frozen --extra dev python scripts/validate_config.py --runtime-profile default_tiled_scene

quality: whitespace format-check lint check-docs test validate-config
