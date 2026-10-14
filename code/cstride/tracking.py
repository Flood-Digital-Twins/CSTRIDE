"""Opt-in W&B logging; ordinary runs require no account or network access."""
import os

class _DisabledTracking:
    def init(self, **kwargs):
        return None
    def watch(self, *args, **kwargs):
        pass
    def log(self, *args, **kwargs):
        pass

if os.environ.get("CSTRIDE_WANDB") == "1":
    import wandb
else:
    wandb = _DisabledTracking()
