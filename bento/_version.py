import tomli

def get_version():
    with open("pyproject.toml", "rb") as f:
        pyproject = tomli.load(f)
    return pyproject["project"]["version"]

__version__ = get_version()

