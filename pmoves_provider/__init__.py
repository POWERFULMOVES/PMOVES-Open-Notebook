"""PMOVES provider mode for Open Notebook.

Supports three modes:
- tensorzero: Routes all LLM calls via TensorZero gateway
- hybrid: Seeds TensorZero + local providers from PMOVES Model Registry
- native: No managed providers (user configures manually)
"""

from pmoves_provider.bootstrap import bootstrap_hybrid, bootstrap_tensorzero
from pmoves_provider.config import ProviderMode, get_provider_mode

__all__ = [
    "ProviderMode",
    "get_provider_mode",
    "bootstrap_tensorzero",
    "bootstrap_hybrid",
]
