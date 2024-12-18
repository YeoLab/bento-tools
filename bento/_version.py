import tomli

def get_version():
    import os
    import pathlib
    
    package_dir = pathlib.Path(__file__).parent.parent
    pyproject_path = os.path.join(package_dir, "pyproject.toml")
    
    with open(pyproject_path, "rb") as f:
        pyproject = tomli.load(f)
    return pyproject["project"]["version"]

__version__ = get_version()

