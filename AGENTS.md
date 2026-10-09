# Python Project Instructions

## Formatting

* Prefer lines of roughly 80-100 characters.
* Use up to 120 characters when it improves readability.
* Exceed 120 only when breaking the line would make the code materially less clear.
* Docstrings should never exceed 100 characters.

## Documentation and Comments

Every package, module, class, and function must have a reStructuredText (RST) docstring.

Use comments where they explain reasoning that the code itself cannot express clearly and explain why code and algorithms are chosen or done in a certain way.

Also use comments to explain how non-trivial code works, do not comment obvious code.

## Type Safety

All code must be statically type-safe.

* Annotate parameters, return values, and important attributes or collections.
* Avoid unnecessary `Any`.
* Resolve type-checker errors as much as possible.
* Use `# type: ignore[...]` only when type errors cannot be resolved.

Custom types should be created in `src/proxycardtools/general/custom_types.py`