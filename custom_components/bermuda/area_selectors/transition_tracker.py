"""
Hidden Markov Model transition tracker for area selection.

Learns which area transitions are common based on observed device movements,
and provides transition probabilities to improve area selection accuracy.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)

# Storage constants
STORAGE_DIR = ".storage"
STORAGE_FILE = "bermuda.transition_tracker"
STORAGE_VERSION = 1


class TransitionTracker:
    """
    Tracks area transitions and builds a probability matrix.

    Uses a simple counting approach to learn transition probabilities:
    - Count how many times we transition from area A to area B
    - P(B|A) = count(A→B) / total_transitions_from_A

    The tracker also maintains a "self-transition" count for staying
    in the same area, which helps model the probability of remaining
    in place vs. moving to another area.
    """

    # Configuration defaults
    DEFAULT_SMOOTHING = 1.0  # Laplace smoothing to avoid zero probabilities
    DEFAULT_MIN_TRANSITIONS = 5  # Minimum transitions before using learned probs
    DEFAULT_DECAY_FACTOR = 0.995  # Slight decay to adapt to changes over time
    DEFAULT_SELF_TRANSITION_WEIGHT = 10  # Initial weight for staying in same area

    def __init__(
        self,
        hass: HomeAssistant | None = None,
        smoothing: float = DEFAULT_SMOOTHING,
        min_transitions: int = DEFAULT_MIN_TRANSITIONS,
        decay_factor: float = DEFAULT_DECAY_FACTOR,
        self_transition_weight: float = DEFAULT_SELF_TRANSITION_WEIGHT,
    ) -> None:
        """
        Initialize the transition tracker.

        Args:
            hass: Home Assistant instance (for storage path). If None, persistence disabled.
            smoothing: Laplace smoothing factor to avoid zero probabilities.
            min_transitions: Minimum transitions needed before using learned probabilities.
            decay_factor: Factor to decay old counts (0.995 = 0.5% decay per update).
            self_transition_weight: Initial weight for self-transitions (staying in place).

        """
        self.hass = hass
        self.smoothing = smoothing
        self.min_transitions = min_transitions
        self.decay_factor = decay_factor
        self.self_transition_weight = self_transition_weight

        # Transition counts: {from_area: {to_area: count}}
        self._transitions: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

        # Track total transitions from each area for normalization
        self._total_from: dict[str, float] = defaultdict(float)

        # Track known areas (for smoothing across all areas)
        self._known_areas: set[str] = set()

        # Track last area per device for transition detection
        self._last_area: dict[str, str] = {}

        # Flag to track if we have unsaved changes
        self._dirty = False

        # Load persisted data if available
        if hass is not None:
            self._load()

    def record_area(self, device_address: str, area_id: str) -> None:
        """
        Record that a device is in a given area.

        If the device was previously in a different area, this records
        a transition. If it's the same area, records a self-transition.

        Args:
            device_address: Unique identifier for the device.
            area_id: The area the device is currently in.

        """
        if area_id is None:
            return

        # Add to known areas
        self._known_areas.add(area_id)

        # Check for transition
        last_area = self._last_area.get(device_address)

        if last_area is not None:
            # Record the transition (including self-transitions)
            self._record_transition(last_area, area_id)

        # Update last known area
        self._last_area[device_address] = area_id

    def _record_transition(self, from_area: str, to_area: str) -> None:
        """Record a transition from one area to another."""
        # Apply decay to all existing counts (helps adapt to changes)
        if self.decay_factor < 1.0:
            self._apply_decay()

        # Increment transition count
        self._transitions[from_area][to_area] += 1.0
        self._total_from[from_area] += 1.0

        self._dirty = True

    def _apply_decay(self) -> None:
        """Apply decay factor to all transition counts."""
        for from_area in self._transitions:
            for to_area in self._transitions[from_area]:
                self._transitions[from_area][to_area] *= self.decay_factor
            self._total_from[from_area] *= self.decay_factor

    def get_transition_probability(self, from_area: str | None, to_area: str) -> float:
        """
        Get the probability of transitioning from one area to another.

        Uses Laplace smoothing to ensure no probability is exactly zero.

        Args:
            from_area: The area the device is currently in (or None if unknown).
            to_area: The area we're considering transitioning to.

        Returns:
            Probability between 0 and 1. Returns 1.0 if from_area is None
            or if we don't have enough data yet.

        """
        # If no current area, return neutral probability
        if from_area is None:
            return 1.0

        # If we don't have enough transitions, return neutral probability
        total = self._total_from.get(from_area, 0)
        if total < self.min_transitions:
            return 1.0

        # Calculate probability with Laplace smoothing
        # P(to|from) = (count(from→to) + smoothing) / (total_from + smoothing * num_areas)
        count = self._transitions[from_area].get(to_area, 0)
        num_areas = max(len(self._known_areas), 1)

        return (count + self.smoothing) / (total + self.smoothing * num_areas)

    def get_all_probabilities(self, from_area: str | None) -> dict[str, float]:
        """
        Get transition probabilities to all known areas from a given area.

        Args:
            from_area: The area to get transition probabilities from.

        Returns:
            Dict mapping area_id to probability.

        """
        return {area: self.get_transition_probability(from_area, area) for area in self._known_areas}

    def initialize_area(self, area_id: str) -> None:
        """
        Initialize an area with default self-transition weight.

        Call this when a new area is discovered to give it a reasonable
        starting probability for self-transitions.

        Args:
            area_id: The area to initialize.

        """
        if area_id not in self._known_areas:
            self._known_areas.add(area_id)
            # Add initial self-transition weight
            self._transitions[area_id][area_id] = self.self_transition_weight
            self._total_from[area_id] = self.self_transition_weight
            self._dirty = True

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about learned transitions for debugging."""
        stats = {
            "known_areas": len(self._known_areas),
            "total_transitions": sum(self._total_from.values()),
            "areas": list(self._known_areas),
            "transition_matrix": {},
        }

        for from_area in self._known_areas:
            stats["transition_matrix"][from_area] = {
                to_area: round(self.get_transition_probability(from_area, to_area), 3) for to_area in self._known_areas
            }

        return stats

    # --- Persistence methods ---

    def _get_storage_path(self) -> Path | None:
        """Get the path to the storage file."""
        if self.hass is None:
            return None
        return Path(self.hass.config.path(STORAGE_DIR)) / STORAGE_FILE

    def _load(self) -> None:
        """Load transition data from persistent storage."""
        path = self._get_storage_path()
        if path is None or not path.exists():
            _LOGGER.debug("No transition data to load")
            return

        try:
            with path.open() as f:
                data = json.load(f)

            if data.get("version") != STORAGE_VERSION:
                _LOGGER.warning("Transition data version mismatch, starting fresh")
                return

            # Restore transitions
            transitions = data.get("transitions", {})
            for from_area, to_areas in transitions.items():
                for to_area, count in to_areas.items():
                    self._transitions[from_area][to_area] = count

            # Restore totals
            self._total_from = defaultdict(float, data.get("total_from", {}))

            # Restore known areas
            self._known_areas = set(data.get("known_areas", []))

            _LOGGER.info(
                "Loaded transition data: %d areas, %.0f total transitions",
                len(self._known_areas),
                sum(self._total_from.values()),
            )

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            _LOGGER.warning("Failed to load transition data: %s", e)

    def save(self) -> None:
        """Save transition data to persistent storage."""
        if not self._dirty:
            return

        path = self._get_storage_path()
        if path is None:
            return

        try:
            # Ensure storage directory exists
            path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "version": STORAGE_VERSION,
                "transitions": {from_area: dict(to_areas) for from_area, to_areas in self._transitions.items()},
                "total_from": dict(self._total_from),
                "known_areas": list(self._known_areas),
            }

            with path.open("w") as f:
                json.dump(data, f, indent=2)

            self._dirty = False
            _LOGGER.debug("Saved transition data")

        except OSError:
            _LOGGER.exception("Failed to save transition data")

    def clear(self) -> None:
        """Clear all learned transitions."""
        self._transitions.clear()
        self._total_from.clear()
        self._known_areas.clear()
        self._last_area.clear()
        self._dirty = True
        _LOGGER.info("Cleared all transition data")
