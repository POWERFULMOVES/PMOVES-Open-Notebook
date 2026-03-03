"""
Pytest configuration file.

This file ensures that the project root is in the Python path,
allowing tests to import from the api and open_notebook modules.
"""

import os
import sys
from pathlib import Path

# Ensure fail-closed password auth remains enabled in tests with a stable token.
# API tests should send this token via Authorization header.
os.environ["OPEN_NOTEBOOK_PASSWORD"] = "test-password"

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
