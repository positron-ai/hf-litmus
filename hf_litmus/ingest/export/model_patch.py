import types
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from torch.nn import Module


@dataclass
class Patch:
    """
    A strategy for altering a PyTorch Module for ingestion.
    """

    def name(self) -> str:
        return "no-op"

    def apply(self, module: Module) -> bool:
        """
        Apply the patch, if applicable, to the given Module via mutation,
        returning True if the patch fired.
        """
        return False


@dataclass
class PatchMethod(Patch):
    """
    Replace a method of all modules which are subtypes of the given
    class.
    """

    description: str
    cls: type
    method: str
    func: Callable[..., Any]

    def name(self) -> str:
        return f"{self.cls.__name__} ({self.description})"

    def apply(self, module: Module) -> bool:
        if isinstance(module, self.cls):
            assert getattr(module, self.method, None) is not None
            setattr(module, self.method, types.MethodType(self.func, module))
            return True
        else:
            return False


_patches = []


def register_patch(patch: Patch):
    _patches.append(patch)


def patch_model(model):
    """Patch a transformers model"""
    patch_counts = defaultdict(lambda: 0)
    for module in model.modules():
        for patch in _patches:
            if patch.apply(module):
                patch_counts[patch.name()] += 1

    for name, count in patch_counts.items():
        if count > 0:
            print(f"Patch {name} rewrote {count} modules")
