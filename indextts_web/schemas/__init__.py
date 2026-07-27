"""Public request-schema inventory.

The compatibility models still live beside their endpoint implementations.
This inventory gives modular callers one stable import boundary while those
large models migrate independently.
"""

from .compatibility import SchemaInventory, from_legacy

__all__ = ["SchemaInventory", "from_legacy"]

