# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

from .learned_relative_positional_embedding import LearnedRelativePositionalEmbedding
from .sinusoidal_relative_positional_embedding import (
    SinusoidalRelativePositionalEmbedding,
)


def RelativePositionalEmbedding(
    embedding_dim: int,
    padding_idx: Optional[int] = None,
    max_size: Optional[int] = None,
    learned: bool = False,
):
    if learned:
        # if padding_idx is specified then offset the embedding ids by
        # this index and adjust num_embeddings appropriately
        assert max_size is not None
        m = LearnedRelativePositionalEmbedding(
            embedding_dim, padding_idx, max_size=max_size
        )
    else:
        m = SinusoidalRelativePositionalEmbedding(
            embedding_dim,
            padding_idx,
            init_size=1024,
            max_size=max_size,
            scale_embedding=True,
        )
    return m
