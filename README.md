### Installing project requirements through uv

After install uv globally, in your computer, you can create a local project-specific environment by running the following command:

```
uv sync
```

### Using the project environment

You have two options. Once you are in the directory of your project, either activate the environment through:
```
source .venv/bin/activate
```

or run python through
```
uv run python your_script.py
```