"""Data-blind plan input view: names/shapes, not numerical or cell values."""

from ..external_computers.values import packed


def is_table(value):
    if type(value) is not dict or set(value) != {"columns", "rows"}:
        return False
    columns, rows = value["columns"], value["rows"]
    return (
        type(columns) is list
        and 1 <= len(columns) <= 32
        and all(type(c) is str for c in columns)
        and type(rows) is list
        and all(type(row) is list and len(row) == len(columns) for row in rows)
    )


def has_tables(value):
    """Structural capability gate; does not inspect cell values or classify prose."""
    if is_table(value):
        return True
    if type(value) is dict:
        return any(has_tables(item) for item in value.values())
    if type(value) is list:
        return any(has_tables(item) for item in value)
    return False


def describe(value):
    kind = type(value)
    if kind is dict:
        if is_table(value):
            columns, rows = value["columns"], value["rows"]
            return {
                "kind": "sqlite_table",
                "columns": columns,
                "row_count": len(rows),
                "cell_kinds": [
                    sorted({type(row[i]).__name__ for row in rows})
                    for i in range(len(columns))
                ],
                "original_shape": "{columns:[names],rows:[[cells]]}",
            }
        return {"kind": "object", "fields": {k: describe(v) for k, v in value.items()}}
    if kind is list:
        shapes = {}
        for item in value:
            shape = describe(item)
            shapes[packed(shape)] = shape
            if len(shapes) > 8:
                return {
                    "kind": "array",
                    "length": len(value),
                    "items": {"kind": "mixed_json"},
                }
        return {
            "kind": "array",
            "length": len(value),
            "item_shapes": list(shapes.values()),
        }
    return {"kind": "null" if value is None else kind.__name__}


def input_view(inputs):
    result = {key: describe(value) for key, value in inputs.items()}
    packed(result)
    return result
