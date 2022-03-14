import importlib
import os

transforms_dir = os.path.dirname(__file__)
for file in os.listdir(transforms_dir):
    path = os.path.join(transforms_dir, file)
    if (
        not file.startswith("_")
        and not file.startswith(".")
        and (file.endswith(".py") or os.path.isdir(path))
    ):
        name = file[: file.find(".py")] if file.endswith(".py") else file
        importlib.import_module("espresso.data.feature_transforms." + name)
