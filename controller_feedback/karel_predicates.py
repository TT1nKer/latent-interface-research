"""Karel DSL observations. STOP is a decision, not an observation bit."""

from karel_trace_control import DELTAS


def observe(world):
    """Return (front_clear, left_clear, right_clear, marker_present)."""
    row, col = world.hero
    clear = []
    for offset in (0, -1, 1):
        dr, dc = DELTAS[(world.direction + offset) % 4]
        r, c = row + dr, col + dc
        clear.append(0 <= r < world.rows and 0 <= c < world.cols and (r, c) not in world.blocked)
    return (*clear, bool(world.markers.get(world.hero, 0)))
