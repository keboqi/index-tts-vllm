"""CPU fixtures for production code whose module imports require GPU packages."""

import ast
import copy
from functools import lru_cache
from pathlib import Path

from indextts_web.gpu_profiles import GIB, GpuInfo

ROOT = Path(__file__).resolve().parents[1]


def gpu(gib=24, free=None):
    return GpuInfo("test GPU", int(gib * GIB), int((gib if free is None else free) * GIB), "8.9")


@lru_cache
def source_tree(path):
    return ast.parse(path.read_text(encoding="utf-8-sig"))


def load_definition(path, name, namespace):
    """Execute an actual function/class with injected optional dependencies.

    This avoids importing CUDA, loading checkpoints, or starting Modal during
    CPU regression tests. It deliberately preserves the production body.
    """
    tree = source_tree(path)
    for part in name.split("."):
        tree = next(node for node in tree.body
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == part)
    definition = copy.deepcopy(tree)
    definition.decorator_list = []
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[definition.name]
