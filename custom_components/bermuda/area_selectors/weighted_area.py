"""Weighted area voting area selection algorithm."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import TYPE_CHECKING

from .base import AreaSelectionResult, AreaSelectorBase, AreaSelectorConfig
from .fingerprint_store import FingerprintStore
from .transition_tracker import TransitionTracker

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

    from custom_components.bermuda.bermuda_advert import BermudaAdvert
    from custom_components.bermuda.bermuda_device import BermudaDevice


class WeightedAreaSelector(AreaSelectorBase):
    """
    Area selector that uses weighted voting from multiple scanners.

    Instead of just picking the closest scanner's area, this algorithm
    considers distances from ALL visible scanners and uses inverse-distance
    weighting to vote for areas. The area with the highest weighted vote wins.

    This provides more stable area detection when a device is between
    multiple scanners or near area boundaries.

    Optionally uses signal variance weighting to reduce the influence of
    noisy/unstable signals (e.g., from multipath interference).
    """

    SELECTOR_ID = "weighted_area"
    SELECTOR_NAME = "Weighted Area Voting"

    # Configuration defaults
    DEFAULT_WEIGHT_POWER = 2.0  # Inverse square weighting (1/d²)
    DEFAULT_MIN_SCANNERS = 2  # Minimum scanners for weighted voting
    DEFAULT_FALLBACK_TO_CLOSEST = True  # Fall back to closest if < min_scanners
    DEFAULT_USE_VARIANCE_WEIGHTING = True  # Enable signal variance weighting
    DEFAULT_VARIANCE_SCALE = 10.0  # Scale factor for variance normalization
    DEFAULT_MIN_VARIANCE_SAMPLES = 3  # Minimum RSSI samples needed for variance calc
    DEFAULT_USE_HMM = True  # Enable Hidden Markov Model transition learning
    DEFAULT_HMM_WEIGHT = 1.0  # How strongly HMM affects the final weight (0-1)
    DEFAULT_USE_RECENCY_WEIGHTING = True  # Enable recency weighting
    DEFAULT_RECENCY_DECAY_SECS = 5.0  # Time constant for recency decay (seconds)
    DEFAULT_USE_PASSIVE_LEARNING = True  # Enable passive fingerprint learning
    DEFAULT_PASSIVE_LEARNING_THRESHOLD = 0.65  # Confidence threshold for passive learning
    DEFAULT_PASSIVE_LEARNING_INTERVAL = 30  # Minimum seconds between passive recordings
    DEFAULT_USE_FINGERPRINT_MATCHING = True  # Enable fingerprint matching in area selection
    DEFAULT_FINGERPRINT_WEIGHT = 0.5  # How strongly fingerprints affect the final weight (0-1)

    def __init__(self, config: AreaSelectorConfig) -> None:
        """Initialize the weighted area selector."""
        super().__init__(config)

        # Get algorithm-specific config with defaults
        self.weight_power = config.extra.get("weight_power", self.DEFAULT_WEIGHT_POWER)
        self.min_scanners = config.extra.get("min_scanners", self.DEFAULT_MIN_SCANNERS)
        self.fallback_to_closest = config.extra.get("fallback_to_closest", self.DEFAULT_FALLBACK_TO_CLOSEST)
        self.use_variance_weighting = config.extra.get("use_variance_weighting", self.DEFAULT_USE_VARIANCE_WEIGHTING)
        self.variance_scale = config.extra.get("variance_scale", self.DEFAULT_VARIANCE_SCALE)
        self.min_variance_samples = config.extra.get("min_variance_samples", self.DEFAULT_MIN_VARIANCE_SAMPLES)
        self.use_hmm = config.extra.get("use_hmm", self.DEFAULT_USE_HMM)
        self.hmm_weight = config.extra.get("hmm_weight", self.DEFAULT_HMM_WEIGHT)
        self.use_recency_weighting = config.extra.get("use_recency_weighting", self.DEFAULT_USE_RECENCY_WEIGHTING)
        self.recency_decay_secs = config.extra.get("recency_decay_secs", self.DEFAULT_RECENCY_DECAY_SECS)
        self.use_passive_learning = config.extra.get("use_passive_learning", self.DEFAULT_USE_PASSIVE_LEARNING)
        self.passive_learning_threshold = config.extra.get(
            "passive_learning_threshold", self.DEFAULT_PASSIVE_LEARNING_THRESHOLD
        )
        self.passive_learning_interval = config.extra.get(
            "passive_learning_interval", self.DEFAULT_PASSIVE_LEARNING_INTERVAL
        )
        self.use_fingerprint_matching = config.extra.get(
            "use_fingerprint_matching", self.DEFAULT_USE_FINGERPRINT_MATCHING
        )
        self.fingerprint_weight = config.extra.get("fingerprint_weight", self.DEFAULT_FINGERPRINT_WEIGHT)

        # Initialize transition tracker for HMM if enabled
        self._transition_tracker: TransitionTracker | None = None
        hass: HomeAssistant | None = config.extra.get("hass")
        if self.use_hmm:
            self._transition_tracker = TransitionTracker(hass=hass)

        # Initialize fingerprint store for manual training
        self._fingerprint_store = FingerprintStore(hass=hass)

        # Track save counter to periodically persist HMM data
        self._save_counter = 0
        self._save_interval = 100  # Save every N area selections

        # Track last passive learning time per device
        self._last_passive_learning: dict[str, float] = {}
        self._fingerprint_save_counter = 0

    def select_area(self, device: BermudaDevice, current_stamp: float) -> AreaSelectionResult:
        """
        Select area using weighted voting from multiple scanners.

        Each scanner votes for its area with weight = 1 / distance^power.
        The area with the highest total weight wins.

        If HMM is enabled, weights are multiplied by transition probabilities
        to favor areas that are reachable from the current area.
        """
        result = AreaSelectionResult(winning_advert=None)
        result.device = device.name

        # Get current area for HMM transition probability lookup
        current_area_id: str | None = None
        if device.area_advert is not None:
            current_area_id = device.area_advert.area_id

        # Collect all valid adverts
        valid_adverts: list[BermudaAdvert] = [
            advert for advert in device.adverts.values() if self.validate_advert(advert, current_stamp)
        ]

        if not valid_adverts:
            result.reason = "No valid adverts available"
            return result

        # If we don't have enough scanners, optionally fall back to closest
        if len(valid_adverts) < self.min_scanners:
            if self.fallback_to_closest:
                return self._select_closest(valid_adverts, result, device.address)
            result.reason = f"Only {len(valid_adverts)} scanner(s), need {self.min_scanners}"
            return result

        # Calculate weights for each area
        area_weights: dict[str, float] = defaultdict(float)
        area_adverts: dict[str, BermudaAdvert] = {}  # Track best advert per area
        area_names: dict[str, str] = {}  # Track area names

        for advert in valid_adverts:
            if advert.area_id is None or advert.rssi_distance is None:
                continue

            # Initialize area in transition tracker if using HMM
            if self._transition_tracker is not None:
                self._transition_tracker.initialize_area(advert.area_id)

            # Calculate weight using inverse distance
            # Add small epsilon to avoid division by zero for very close devices
            distance = max(advert.rssi_distance, 0.1)
            weight = 1.0 / (distance**self.weight_power)

            # Apply signal variance weighting if enabled
            if self.use_variance_weighting:
                confidence = self._calculate_signal_confidence(advert)
                weight *= confidence

            # Apply recency weighting if enabled
            if self.use_recency_weighting:
                recency_factor = self._calculate_recency_factor(advert, current_stamp)
                weight *= recency_factor

            # Apply HMM transition probability if enabled
            if self._transition_tracker is not None and self.hmm_weight > 0:
                transition_prob = self._transition_tracker.get_transition_probability(current_area_id, advert.area_id)
                # Blend transition probability with weight
                # When hmm_weight=1.0, full multiplication
                # When hmm_weight=0.5, sqrt of transition_prob is used
                weight *= transition_prob**self.hmm_weight

            area_weights[advert.area_id] += weight
            area_names[advert.area_id] = advert.area_name or advert.area_id

            # Track the closest advert for each area (for diagnostic output)
            if advert.area_id not in area_adverts or (
                area_adverts[advert.area_id].rssi_distance is not None
                and advert.rssi_distance < area_adverts[advert.area_id].rssi_distance  # type: ignore[union-attr]
            ):
                area_adverts[advert.area_id] = advert

        if not area_weights:
            result.reason = "No areas with valid weights"
            return result

        # Apply fingerprint matching if enabled and we have fingerprint data
        if self.use_fingerprint_matching and self.fingerprint_weight > 0:
            self._apply_fingerprint_matching(area_weights, valid_adverts)

        # Find the area with the highest weight
        winning_area_id = max(area_weights.keys(), key=lambda x: area_weights[x])
        winning_advert = area_adverts[winning_area_id]

        # Build diagnostic info
        sorted_areas = sorted(area_weights.items(), key=lambda x: x[1], reverse=True)
        top_areas = sorted_areas[:2] if len(sorted_areas) >= 2 else sorted_areas

        if len(top_areas) >= 2:
            result.areas = (
                area_names.get(top_areas[0][0], ""),
                area_names.get(top_areas[1][0], ""),
            )
            result.distance = (
                area_adverts[top_areas[0][0]].rssi_distance or 0,
                area_adverts[top_areas[1][0]].rssi_distance or 0,
            )
            # Calculate weight ratio as a proxy for confidence
            total_weight = sum(area_weights.values())
            if total_weight > 0:
                result.pcnt_diff = top_areas[0][1] / total_weight
        else:
            result.areas = (area_names.get(winning_area_id, ""), "")
            result.distance = (winning_advert.rssi_distance or 0, 0)
            result.pcnt_diff = 1.0

        result.scannername = (
            winning_advert.name,
            area_adverts[top_areas[1][0]].name if len(top_areas) >= 2 else "",
        )

        # Build reason string with weight breakdown
        weight_str = ", ".join(
            f"{area_names.get(area_id, area_id)}: {weight:.2f}" for area_id, weight in sorted_areas[:3]
        )
        result.reason = f"Weighted vote ({len(valid_adverts)} scanners): {weight_str}"

        result.winning_advert = winning_advert

        # Record area for HMM learning
        self._record_area_for_hmm(device.address, winning_area_id)

        # Passive fingerprint learning - record when confidence is high
        if self.use_passive_learning and result.pcnt_diff is not None:
            self._maybe_record_passive_fingerprint(
                device=device,
                area_id=winning_area_id,
                area_name=area_names.get(winning_area_id, winning_area_id),
                confidence=result.pcnt_diff,
                current_stamp=current_stamp,
                valid_adverts=valid_adverts,
            )

        return result

    def _select_closest(
        self, valid_adverts: list[BermudaAdvert], result: AreaSelectionResult, device_address: str
    ) -> AreaSelectionResult:
        """
        Fallback to selecting the closest scanner when not enough scanners are available.

        This provides a simpler selection method similar to min_distance.
        """
        closest: BermudaAdvert | None = None

        for advert in valid_adverts:
            if advert.rssi_distance is None:
                continue
            if closest is None or (closest.rssi_distance is not None and advert.rssi_distance < closest.rssi_distance):
                closest = advert

        if closest is not None:
            result.winning_advert = closest
            result.areas = (closest.area_name or "", "")
            result.scannername = (closest.name, "")
            result.distance = (closest.rssi_distance or 0, 0)
            result.reason = f"Fallback to closest (only {len(valid_adverts)} scanner(s))"

            # Record area for HMM learning
            if closest.area_id is not None:
                self._record_area_for_hmm(device_address, closest.area_id)

        return result

    def _record_area_for_hmm(self, device_address: str, area_id: str) -> None:
        """Record an area assignment for HMM transition learning."""
        if self._transition_tracker is None:
            return

        self._transition_tracker.record_area(device_address, area_id)

        # Periodically save the transition data
        self._save_counter += 1
        if self._save_counter >= self._save_interval:
            self._transition_tracker.save()
            self._save_counter = 0

    def save_hmm_data(self) -> None:
        """Save HMM transition data to persistent storage."""
        if self._transition_tracker is not None:
            self._transition_tracker.save()

    def get_hmm_statistics(self) -> dict | None:
        """Get HMM statistics for debugging/diagnostics."""
        if self._transition_tracker is not None:
            return self._transition_tracker.get_statistics()
        return None

    def clear_hmm_data(self) -> None:
        """Clear all learned HMM transition data."""
        if self._transition_tracker is not None:
            self._transition_tracker.clear()

    def train_location(
        self,
        device: BermudaDevice,
        area_id: str,
        area_name: str,
        current_stamp: float,
    ) -> dict[str, int]:
        """
        Train the model with a known device location.

        Records the current RSSI readings as a fingerprint for the specified area,
        and updates the HMM transition tracker with this area assignment.

        Args:
            device: The BermudaDevice to train with.
            area_id: The ID of the area the device is in.
            area_name: Human-readable name of the area.
            current_stamp: Current monotonic timestamp.

        Returns:
            Dict with counts of what was trained: {"fingerprints": N, "transitions": 0|1}

        """
        result = {"fingerprints": 0, "transitions": 0}

        # Collect current RSSI readings from all valid scanners
        scanner_rssi: dict[str, float] = {}
        for advert in device.adverts.values():
            if self.validate_advert(advert, current_stamp) and advert.rssi is not None:
                scanner_rssi[advert.scanner_address] = advert.rssi

        # Record fingerprint if we have RSSI data
        if scanner_rssi:
            self._fingerprint_store.record_fingerprint(
                area_id=area_id,
                area_name=area_name,
                scanner_rssi=scanner_rssi,
                timestamp=current_stamp,
            )
            result["fingerprints"] = len(scanner_rssi)
            self._fingerprint_store.save()

        # Record area for HMM transition learning
        if self._transition_tracker is not None:
            self._transition_tracker.record_area(device.address, area_id)
            self._transition_tracker.save()
            result["transitions"] = 1

        return result

    def get_fingerprint_statistics(self) -> dict:
        """Get fingerprint statistics for debugging/diagnostics."""
        return self._fingerprint_store.get_statistics()

    def clear_fingerprint_data(self) -> None:
        """Clear all learned fingerprint data."""
        self._fingerprint_store.clear()
        self._fingerprint_store.save()

    def _calculate_signal_confidence(self, advert: BermudaAdvert) -> float:
        """
        Calculate a confidence factor based on RSSI signal variance.

        High variance indicates noisy/unstable signal (multipath, interference),
        which should be weighted less. Low variance indicates stable signal,
        which should be weighted more.

        Returns a value between 0 and 1, where:
        - 1.0 = perfect confidence (no variance)
        - 0.0 = no confidence (extremely high variance)

        The formula used is: confidence = 1 / (1 + variance / scale)

        This provides a smooth decay from 1.0 as variance increases.
        With default scale=10:
        - variance=0 → confidence=1.0
        - variance=5 → confidence=0.67
        - variance=10 → confidence=0.5
        - variance=20 → confidence=0.33
        - variance=50 → confidence=0.17
        """
        # Get recent RSSI readings from history
        rssi_history = advert.hist_rssi

        # Need minimum samples to calculate meaningful variance
        if len(rssi_history) < self.min_variance_samples:
            # Not enough samples, assume moderate confidence
            return 0.75

        # Calculate standard deviation of recent RSSI values
        # Use only recent readings (last 5-10) for responsiveness
        recent_rssi = rssi_history[: min(10, len(rssi_history))]

        try:
            variance = statistics.stdev(recent_rssi)
        except statistics.StatisticsError:
            # Can happen if all values are identical (stdev needs at least 2 distinct values)
            # Identical values = zero variance = maximum confidence
            return 1.0

        # Calculate confidence using inverse relationship with variance
        # confidence = 1 / (1 + variance / scale)
        return 1.0 / (1.0 + variance / self.variance_scale)

    def _calculate_recency_factor(self, advert: BermudaAdvert, current_stamp: float) -> float:
        """
        Calculate a weight factor based on how recent the reading is.

        Fresh readings get higher weight, stale readings get lower weight.
        Uses exponential decay: factor = exp(-age / decay_constant)

        Returns a value between 0 and 1, where:
        - 1.0 = just received (age=0)
        - 0.37 = age equals decay_constant
        - 0.14 = age equals 2x decay_constant
        - 0.05 = age equals 3x decay_constant

        With default decay_constant=5 seconds:
        - age=0s → factor=1.0
        - age=2.5s → factor=0.61
        - age=5s → factor=0.37
        - age=10s → factor=0.14

        """
        # Get the stamp of the last reading
        if advert.stamp is None:
            # No timestamp, assume moderately recent
            return 0.5

        age = current_stamp - advert.stamp

        # Clamp age to non-negative (shouldn't happen but just in case)
        age = max(0.0, age)

        # Calculate exponential decay
        # factor = e^(-age / decay_constant)
        return math.exp(-age / self.recency_decay_secs)

    def _maybe_record_passive_fingerprint(
        self,
        device: BermudaDevice,
        area_id: str,
        area_name: str,
        confidence: float,
        current_stamp: float,
        valid_adverts: list[BermudaAdvert],
    ) -> None:
        """
        Passively record a fingerprint when area selection confidence is high.

        Only records if:
        1. Confidence exceeds the threshold
        2. Enough time has passed since last recording for this device

        """
        # Check confidence threshold
        if confidence < self.passive_learning_threshold:
            return

        # Check time since last passive learning for this device
        last_time = self._last_passive_learning.get(device.address, 0.0)
        if current_stamp - last_time < self.passive_learning_interval:
            return

        # Collect RSSI readings from valid adverts
        scanner_rssi: dict[str, float] = {}
        for advert in valid_adverts:
            if advert.rssi is not None:
                scanner_rssi[advert.scanner_address] = advert.rssi

        if not scanner_rssi:
            return

        # Record the fingerprint
        self._fingerprint_store.record_fingerprint(
            area_id=area_id,
            area_name=area_name,
            scanner_rssi=scanner_rssi,
            timestamp=current_stamp,
        )

        # Update last passive learning time
        self._last_passive_learning[device.address] = current_stamp

        # Periodically save fingerprint data
        self._fingerprint_save_counter += 1
        if self._fingerprint_save_counter >= self._save_interval:
            self._fingerprint_store.save()
            self._fingerprint_save_counter = 0

    def _apply_fingerprint_matching(
        self,
        area_weights: dict[str, float],
        valid_adverts: list[BermudaAdvert],
    ) -> None:
        """
        Apply fingerprint matching to boost weights for areas with matching RSSI patterns.

        Collects current RSSI readings and compares against stored fingerprints.
        Areas with high fingerprint similarity get their weights boosted.

        Args:
            area_weights: Dict of area_id to current weight (modified in place).
            valid_adverts: List of valid adverts with current RSSI readings.

        """
        # Collect current RSSI readings
        scanner_rssi: dict[str, float] = {}
        for advert in valid_adverts:
            if advert.rssi is not None:
                scanner_rssi[advert.scanner_address] = advert.rssi

        if not scanner_rssi:
            return

        # Get fingerprint similarity scores
        similarity_scores = self._fingerprint_store.match_fingerprint(scanner_rssi)
        if not similarity_scores:
            return

        # Apply similarity scores to area weights
        # Formula: new_weight = weight * (1 + fingerprint_weight * similarity)
        # This boosts weights proportionally to fingerprint match quality
        # With fingerprint_weight=0.5 and similarity=1.0, weight increases by 50%
        # With fingerprint_weight=0.5 and similarity=0.5, weight increases by 25%
        for area_id, similarity in similarity_scores.items():
            if area_id in area_weights:
                boost_factor = 1.0 + self.fingerprint_weight * similarity
                area_weights[area_id] *= boost_factor
