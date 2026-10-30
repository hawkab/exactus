"""Finite typed action contract. Does not grade unrestricted prose semantics."""

ACTIONS = frozenset(
    {
        "retain_pending",
        "maintain_reserve",
        "hold_rollout",
        "proceed_rollout",
        "reconcile_deliveries",
        "verify_capacity",
        "deduplicate_by_root",
        "replay_stored_answers",
        "admission_control",
    }
)
ACTION_HINT = (
    'actions must be JSON objects {"kind":"..."}, with no extra fields. Allowed kinds: '
    + ", ".join(sorted(ACTIONS))
    + ". Never clear pending work or disable N+1 reserve. Actions are recommendations, not executed commands. "
)


def action_contract(actions, required=()):
    if type(actions) is not list or not 1 <= len(actions) <= len(ACTIONS):
        return False
    kinds = []
    for action in actions:
        if type(action) is not dict or set(action) != {"kind"}:
            return False
        kind = action["kind"]
        if type(kind) is not str or kind not in ACTIONS:
            return False
        kinds.append(kind)
    selected = set(kinds)
    return (
        len(selected) == len(kinds)
        and not {"hold_rollout", "proceed_rollout"} <= selected
        and set(required) <= selected
    )
