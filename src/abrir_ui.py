import sys
import importlib.abc
import importlib.resources.abc

importlib.abc.Traversable = importlib.resources.abc.Traversable


from mlflow.cli import cli

if __name__ == "__main__":
    
    sys.argv = ["mlflow", "ui", "--backend-store-uri", "sqlite:///mlruns.db"]
    cli()