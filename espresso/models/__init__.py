# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import os


# automatically import any Python files in the models/ directory
for file in os.listdir(os.path.dirname(__file__)):
    if not file.startswith("_") and not file.startswith(".") and file.endswith(".py"):
        model_name = file[:file.find(".py")]
        importlib.import_module("espresso.models." + model_name)
