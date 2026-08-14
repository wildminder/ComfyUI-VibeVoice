"""Compatibility layer for different transformers versions.

Handles the module structure changes between transformers 4.x and 5.x.
"""

import transformers
from packaging import version

_transformers_version = version.parse(transformers.__version__)
_is_v5 = _transformers_version >= version.parse("5.0.0")

if _is_v5:
    # transformers 5.x: Qwen2TokenizerFast is available from top-level
    from transformers import Qwen2Tokenizer, Qwen2TokenizerFast
else:
    # transformers 4.x: Qwen2TokenizerFast is in the submodule
    from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer
    from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast

__all__ = ["Qwen2Tokenizer", "Qwen2TokenizerFast"]
