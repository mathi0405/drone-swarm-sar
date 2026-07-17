"""Policy networks for MAPPO (CTDE). Torch is an *optional* dependency; importing
this subpackage without torch is fine until you actually build a network."""
from swarm_sar.policies.base import build_policy, HAS_TORCH  # noqa: F401


def load_policy(checkpoint: str):
    """Load a trained policy from a checkpoint (torch required)."""
    from swarm_sar.policies.base import load_checkpoint
    return load_checkpoint(checkpoint)


__all__ = ["build_policy", "load_policy", "HAS_TORCH"]
