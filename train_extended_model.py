"""
train_extended_model.py

DEPRECATED: Use train.py --schema extended instead.
This wrapper redirects to the unified training script.
"""

import sys

from train import main


if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--schema", "extended", *sys.argv[1:]]
    main()
