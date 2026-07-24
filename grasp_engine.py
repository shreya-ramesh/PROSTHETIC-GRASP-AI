from dataclasses import dataclass
from typing import Callable, List, Optional

NO_RECOMMENDATION = "No Recommendation"

POWER_GRIP = "Power Grip"
PRECISION_GRIP = "Precision Grip"
TRIPOD_GRIP = "Tripod Grip"
HOOK_GRIP = "Hook Grip"
LATERAL_GRIP = "Lateral Grip"
SPHERICAL_GRIP = "Spherical Grip"


@dataclass(frozen=True)
class GraspRule:
    """A single, self-contained rule mapping detected labels to a grasp type.

    Rules are intentionally small and composable so the engine can later be
    extended with affordance-based reasoning (shape, size, material) without
    touching the rest of the application.
    """

    name: str
    matcher: Callable[[str], bool]
    grasp: str

    def matches(self, object_label: str) -> bool:
        return self.matcher(object_label.lower())


def label_in(*labels: str) -> Callable[[str], bool]:
    normalized_labels = {label.lower() for label in labels}
    return lambda candidate: candidate in normalized_labels


DEFAULT_RULES: List[GraspRule] = [
    GraspRule(
        name="cylindrical_containers",
        matcher=label_in("water bottle", "glass", "banana"),
        grasp=POWER_GRIP,
    ),
    GraspRule(
        name="handled_objects",
        matcher=label_in("cup"),
        grasp=HOOK_GRIP,
    ),
    GraspRule(
        name="small_thin_objects",
        matcher=label_in("medicine"),
        grasp=PRECISION_GRIP,
    ),
    GraspRule(
        name="medium_utility_objects",
        matcher=label_in("remote"),
        grasp=TRIPOD_GRIP,
    ),
    GraspRule(
        name="round_objects",
        matcher=label_in("apple", "orange"),
        grasp=SPHERICAL_GRIP,
    ),
]


class GraspEngine:
    def __init__(self, rules: Optional[List[GraspRule]] = None):
        self._rules = list(rules) if rules is not None else list(DEFAULT_RULES)

    def recommend(self, object_label: str) -> str:
        for rule in self._rules:
            if rule.matches(object_label):
                return rule.grasp
        return NO_RECOMMENDATION

    def register_rule(self, rule: GraspRule, priority: int = 0) -> None:
        """Insert a new rule ahead of existing ones at the given priority index."""
        self._rules.insert(priority, rule)
