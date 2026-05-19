"""
train_core_model.py

DEPRECATED: Use train.py --schema core instead.
This wrapper redirects to the unified training script.
"""

import sys

from train import main


if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--schema", "core", *sys.argv[1:]]
    main()
