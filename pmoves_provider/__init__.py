"""PMOVES TensorZero provider mode for Open Notebook.

When NOTEBOOK_PROVIDER_MODE=tensorzero (default in PMOVES), bootstraps a
managed openai_compatible Credential pointing at the TensorZero gateway,
discovers available models, and registers them in the Model table.
"""

from pmoves_provider.bootstrap import bootstrap_tensorzero
from pmoves_provider.config import ProviderMode, get_provider_mode

__all__ = ["ProviderMode", "get_provider_mode", "bootstrap_tensorzero"]
