def deep_copy(value):
    """Create a deep copy of a JSON-like object."""
    if isinstance(value, dict):
        return {k: deep_copy(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [deep_copy(v) for v in value]
    else:
        return value
