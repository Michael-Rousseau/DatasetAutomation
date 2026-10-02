# pyright: reportMissingImports=false
# torch comes from the optional `cleaning` extra, which CI does not install.

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def pick_device(preferred: str | None = None) -> str:
    """CUDA if present, else Apple's MPS, else CPU, so the same code runs on a laptop or a server."""
    import torch

    if preferred:
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# The bound of a type parameter is evaluated lazily, so torch is only needed by type checkers.
def move_to[Model: torch.nn.Module](model: Model, device: str) -> Model:
    """`model.to(device)`, keeping the model's type.

    transformers wraps `.to` in a decorator that type checkers read as an unbound function;
    `torch.nn.Module.to` is the same method, correctly typed. It moves the model in place.
    """
    import torch

    torch.nn.Module.to(model, device)
    return model
