"""Area selector registry and factory for Bermuda."""

from __future__ import annotations

from .base import AreaSelectionResult, AreaSelectorBase, AreaSelectorConfig
from .min_distance import MinDistanceSelector
from .weighted_area import WeightedAreaSelector

__all__ = [
    "AREA_SELECTORS",
    "DEFAULT_SELECTOR_ID",
    "AreaSelectionResult",
    "AreaSelectorBase",
    "AreaSelectorConfig",
    "create_selector",
    "get_available_selectors",
]

# Registry mapping selector ID to selector class
AREA_SELECTORS: dict[str, type[AreaSelectorBase]] = {
    MinDistanceSelector.SELECTOR_ID: MinDistanceSelector,
    WeightedAreaSelector.SELECTOR_ID: WeightedAreaSelector,
}

DEFAULT_SELECTOR_ID = "min_distance"


def create_selector(
    selector_id: str | None = None,
    config: AreaSelectorConfig | None = None,
) -> AreaSelectorBase:
    """
    Factory function to create an area selector instance.

    Args:
        selector_id: The ID of the selector to create. If None, uses default.
        config: Configuration for the selector. If None, uses defaults.

    Returns:
        An instance of the requested area selector.

    Raises:
        ValueError: If the selector_id is not recognized.

    """
    if selector_id is None:
        selector_id = DEFAULT_SELECTOR_ID

    if config is None:
        config = AreaSelectorConfig()

    if selector_id not in AREA_SELECTORS:
        msg = f"Unknown area selector: {selector_id}. Available: {list(AREA_SELECTORS.keys())}"
        raise ValueError(msg)

    selector_class = AREA_SELECTORS[selector_id]
    return selector_class(config)


def get_available_selectors() -> list[dict[str, str]]:
    """
    Get a list of available area selectors for config flow UI.

    Returns:
        List of dicts with 'value' (selector ID) and 'label' (display name).

    """
    return [
        {"value": selector_class.SELECTOR_ID, "label": selector_class.SELECTOR_NAME}
        for selector_class in AREA_SELECTORS.values()
    ]
