"""
LISA Traffic Sign Dataset — class definitions.

This reflects the 7 annotation tags actually present in the Kaggle LISA release:
  go, goForward, goLeft, stop, stopLeft, warning, warningLeft

These correspond to traffic-light states and basic sign categories used across
the dayTrain, nightTrain, daySequence, and nightSequence splits.
"""

CLASS_NAMES: list[str] = [
    "go",
    "goForward",
    "goLeft",
    "stop",
    "stopLeft",
    "warning",
    "warningLeft",
]

# Reverse lookup: annotation tag → class index
TAG_TO_IDX: dict[str, int] = {
    name: idx for idx, name in enumerate(CLASS_NAMES)
}
