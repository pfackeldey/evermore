# Contributing

We welcome contributions to the project. Please follow the guidelines below.

## Development environment

We use [pixi](https://pixi.sh/latest/) to manage the development environment.

```shell
pixi install
pixi shell
pixi run postinstall
```

### Testing

Use pytest to run the unit checks:

```bash
pixi run test
```

### Linting

We use `ruff` to lint the code. Run the following command to check the code:

```bash
pixi run lint
```

### Check all files

**Recommended before creating a commit**: to run all checks against all files,
run the following command:

```bash
pixi run checkall
```
