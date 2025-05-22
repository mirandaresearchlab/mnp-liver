### Installing project requirements through uv

After [installing uv globally](https://docs.astral.sh/uv/getting-started/installation/), you can create a local project-specific environment by running the following command:

```
uv sync
```

### Using the project environment

Here, you have two options. Once you are in the directory of your project, either activate the environment through:
```
source .venv/bin/activate
```

or run python code through
```
uv run python your_script.py
```