# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

from fairseq.models import import_models

# automatically import any Python files in the models/ directory
models_dir = os.path.dirname(__file__)
import_models(models_dir, "espresso.models")
