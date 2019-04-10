import pyperclip
import numpy as np


def clean(json):
    """
    Dict -> Dict
    Clean a JSON object, extracting numpy objects to their python counterparts
    and throwing an error if this is not possible.
    """
    if isinstance(json, dict):
        return {k: clean(v) for k, v in json.items()}
    elif isinstance(json, list):
        return [clean(v) for v in json]
    elif hasattr(json, "tolist"):
        return clean(json.tolist())
    elif hasattr(json, "item"):
        return clean(json.item())
    else:
        if any([isinstance(json, t) for t in [int, float, bool, str]]):
            return json
        else:
            raise Exception("cannot clean JSON value of type {}".format(type(json)))


def stringify(json, decimal_places=None):
    """
    Dict -> Int? -> String
    Convert a JSON-like object into a string representation.
    """
    if isinstance(json, dict):
        return (
            "{"
            + ", ".join(
                [
                    stringify(k) + ": " + stringify(v, decimal_places=decimal_places)
                    for k, v in json.items()
                ]
            )
            + "}"
        )
    elif isinstance(json, list):
        return (
            "["
            + ", ".join(stringify(v, decimal_places=decimal_places) for v in json)
            + "]"
        )
    elif isinstance(json, str):
        return '"' + str(json) + '"'
    elif isinstance(json, float):
        return (
            "{}" if decimal_places is None else "{:." + str(decimal_places) + "f}"
        ).format(json)
    elif isinstance(json, int):
        return str(json)
    else:
        try:
            return stringify(json.item(), decimal_places=decimal_places)
        except:
            try:
                return stringify(json.tolist(), decimal_places=decimal_places)
            except:
                raise Exception(
                    "unrecognised type {} of {}".format(type(json), str(json))
                )


def copy_to_clipboard(json, decimal_places=None):
    """
    Dict -> String
    Convert a JSON-like object into a string representation and copy it
    to the clipboard.
    """
    string = stringify(json, decimal_places=decimal_places)
    pyperclip.copy(string)
    print("Copied string of length {} to clipboard.".format(len(string)))
