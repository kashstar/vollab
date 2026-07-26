from typing import Any


def coerce_to_list(value: object) -> list[Any]:
    """Normalize Tradier's list-collapsing quirk into a plain list.

    Tradier collapses a single-item JSON array to a bare scalar, and
    returns null for an empty collection.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def extract_collection(payload: dict[str, Any], outer_key: str, inner_key: str) -> list[Any]:
    """Drill into a two-level Tradier envelope and normalize the quirk.

    Handles both levels of nullability Tradier exhibits: the outer key
    itself can be null (e.g. {"expirations": null}), and/or the inner key
    can be null/scalar/list (e.g. {"expirations": {"date": null}}).
    """
    outer = payload.get(outer_key)
    if outer is None:
        return []
    return coerce_to_list(outer.get(inner_key))
