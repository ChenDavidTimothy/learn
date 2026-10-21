"""Process-local official comparator imports; canonical Gymnasium is preserved."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
COMPAT = ROOT.parent/'.venv'/'amago_compat'
UPSTREAM = ROOT/'third_party'/'amago'


def activate():
    if 'gymnasium' in sys.modules:
        raise RuntimeError('activate AMAGO compatibility packages before importing Gymnasium')
    if not (COMPAT/'gymnasium').exists() or not (UPSTREAM/'amago').exists():
        raise RuntimeError('prepare pinned upstream and .venv/amago_compat Gymnasium0.29.1 first')
    sys.path[:0] = [str(COMPAT), str(UPSTREAM)]
    os.environ['WANDB_MODE'] = 'disabled'
    os.environ['WANDB_DISABLE_CODE'] = 'true'
    os.environ['WANDB_SILENT'] = 'true'
    os.environ['TORCHDYNAMO_DISABLE'] = '1'  # Same official operations, eager runtime.
    import gymnasium
    assert gymnasium.__version__ == '0.29.1'
