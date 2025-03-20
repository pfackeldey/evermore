# Contributing

We welcome contributions to the project. Please follow the guidelines below.

## Development environment

We use [uv](https://docs.astral.sh/uv/) to manage the development environment.

Setup a virtual environment (once):

```shell
uv sync
```

### Testing

Use pytest to run the unit checks:

```bash
uv run pytest .
```

### Linting

We use `ruff` to lint the code. Run the following command to check the code:

```bash
uv run ruff check . --fix --show-fixes
```

### Check all files

**Recommended before creating a commit**: to run all checks against all files,
run the following command:

```bash
pre-commit run --all-files
```

### Build the documentation

To build the documentation, run the following command:

```bash
uv run sphinx-build -M html ./docs ./docs/_build -W --keep-going
```
