"""
RSSI Fingerprint storage for area selection.

Stores RSSI readings associated with specific areas, enabling
fingerprint-based area detection in the future.
"""

from __future__ import annotations

import json
import logging
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)

# Storage constants
STORAGE_DIR = ".storage"
STORAGE_FILE = "bermuda.fingerprints"
STORAGE_VERSION = 1

# Fingerprint constants
DEFAULT_MAX_SAMPLES_PER_AREA = 100  # Maximum fingerprint samples to keep per area
DEFAULT_MIN_SAMPLES_FOR_MATCHING = 5  # Minimum samples needed for fingerprint matching


@dataclass
class RSSIFingerprint:
    """A single RSSI fingerprint sample."""

    scanner_rssi: dict[str, float]  # {scanner_address: rssi}
    timestamp: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to serializable dict."""
        return {"scanner_rssi": self.scanner_rssi, "timestamp": self.timestamp}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RSSIFingerprint:
        """Create from dict."""
        return cls(
            scanner_rssi=data.get("scanner_rssi", {}),
            timestamp=data.get("timestamp", 0.0),
        )


@dataclass
class AreaFingerprints:
    """Collection of fingerprints for a single area."""

    area_id: str
    area_name: str = ""
    samples: list[RSSIFingerprint] = field(default_factory=list)

    def add_sample(self, fingerprint: RSSIFingerprint, max_samples: int = DEFAULT_MAX_SAMPLES_PER_AREA) -> None:
        """Add a fingerprint sample, maintaining max size."""
        self.samples.append(fingerprint)
        # Keep only the most recent samples
        if len(self.samples) > max_samples:
            self.samples = self.samples[-max_samples:]

    def get_mean_rssi(self) -> dict[str, float]:
        """Get mean RSSI per scanner across all samples."""
        if not self.samples:
            return {}

        scanner_values: dict[str, list[float]] = {}
        for sample in self.samples:
            for scanner, rssi in sample.scanner_rssi.items():
                if scanner not in scanner_values:
                    scanner_values[scanner] = []
                scanner_values[scanner].append(rssi)

        return {scanner: statistics.mean(values) for scanner, values in scanner_values.items()}

    def get_rssi_variance(self) -> dict[str, float]:
        """Get RSSI variance per scanner across all samples."""
        if len(self.samples) < 2:
            return {}

        scanner_values: dict[str, list[float]] = {}
        for sample in self.samples:
            for scanner, rssi in sample.scanner_rssi.items():
                if scanner not in scanner_values:
                    scanner_values[scanner] = []
                scanner_values[scanner].append(rssi)

        result = {}
        for scanner, values in scanner_values.items():
            if len(values) >= 2:
                result[scanner] = statistics.stdev(values)
        return result

    def to_dict(self) -> dict[str, Any]:
        """Convert to serializable dict."""
        return {
            "area_id": self.area_id,
            "area_name": self.area_name,
            "samples": [s.to_dict() for s in self.samples],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AreaFingerprints:
        """Create from dict."""
        return cls(
            area_id=data.get("area_id", ""),
            area_name=data.get("area_name", ""),
            samples=[RSSIFingerprint.from_dict(s) for s in data.get("samples", [])],
        )


class FingerprintStore:
    """
    Stores and manages RSSI fingerprints for areas.

    Fingerprints are RSSI readings from all visible scanners at a known location.
    Multiple samples per area allow for statistical analysis and better matching.
    """

    def __init__(
        self,
        hass: HomeAssistant | None = None,
        max_samples_per_area: int = DEFAULT_MAX_SAMPLES_PER_AREA,
        min_samples_for_matching: int = DEFAULT_MIN_SAMPLES_FOR_MATCHING,
    ) -> None:
        """
        Initialize the fingerprint store.

        Args:
            hass: Home Assistant instance for storage path. If None, persistence disabled.
            max_samples_per_area: Maximum fingerprint samples to keep per area.
            min_samples_for_matching: Minimum samples needed for fingerprint matching.

        """
        self.hass = hass
        self.max_samples_per_area = max_samples_per_area
        self.min_samples_for_matching = min_samples_for_matching

        # Fingerprints by area: {area_id: AreaFingerprints}
        self._fingerprints: dict[str, AreaFingerprints] = {}

        # Flag for unsaved changes
        self._dirty = False

        # Load persisted data
        if hass is not None:
            self._load()

    def record_fingerprint(
        self,
        area_id: str,
        area_name: str,
        scanner_rssi: dict[str, float],
        timestamp: float = 0.0,
    ) -> None:
        """
        Record an RSSI fingerprint for an area.

        Args:
            area_id: The area identifier.
            area_name: Human-readable area name.
            scanner_rssi: Dict mapping scanner address to RSSI value.
            timestamp: When the reading was taken (monotonic time).

        """
        if not scanner_rssi:
            return

        # Create area entry if needed
        if area_id not in self._fingerprints:
            self._fingerprints[area_id] = AreaFingerprints(area_id=area_id, area_name=area_name)

        # Update area name if provided
        if area_name:
            self._fingerprints[area_id].area_name = area_name

        # Add the fingerprint
        fingerprint = RSSIFingerprint(scanner_rssi=scanner_rssi, timestamp=timestamp)
        self._fingerprints[area_id].add_sample(fingerprint, self.max_samples_per_area)

        self._dirty = True
        _LOGGER.debug(
            "Recorded fingerprint for %s: %d scanners, %d total samples",
            area_name or area_id,
            len(scanner_rssi),
            len(self._fingerprints[area_id].samples),
        )

    def get_area_fingerprint(self, area_id: str) -> AreaFingerprints | None:
        """Get fingerprint data for an area."""
        return self._fingerprints.get(area_id)

    def get_all_areas(self) -> list[str]:
        """Get list of all areas with fingerprints."""
        return list(self._fingerprints.keys())

    def has_sufficient_data(self, area_id: str) -> bool:
        """Check if an area has enough fingerprint samples for matching."""
        fp = self._fingerprints.get(area_id)
        if fp is None:
            return False
        return len(fp.samples) >= self.min_samples_for_matching

    def match_fingerprint(self, scanner_rssi: dict[str, float]) -> dict[str, float]:
        """
        Match a current RSSI reading against stored fingerprints.

        Returns a dict mapping area_id to similarity score (0-1, higher is better).
        Only includes areas with sufficient fingerprint data.

        Uses Euclidean distance in RSSI space, normalized to a 0-1 score.

        Args:
            scanner_rssi: Current RSSI readings {scanner_address: rssi}.

        Returns:
            Dict mapping area_id to similarity score.

        """
        if not scanner_rssi:
            return {}

        scores: dict[str, float] = {}

        for area_id, area_fp in self._fingerprints.items():
            if not self.has_sufficient_data(area_id):
                continue

            mean_rssi = area_fp.get_mean_rssi()
            if not mean_rssi:
                continue

            # Calculate Euclidean distance for common scanners
            common_scanners = set(scanner_rssi.keys()) & set(mean_rssi.keys())
            if not common_scanners:
                continue

            sum_sq_diff = 0.0
            for scanner in common_scanners:
                diff = scanner_rssi[scanner] - mean_rssi[scanner]
                sum_sq_diff += diff * diff

            distance = (sum_sq_diff / len(common_scanners)) ** 0.5

            # Convert distance to similarity score (0-1)
            # Using exponential decay: score = exp(-distance / scale)
            # With scale=10, distance of 10 dBm gives score ~0.37
            scale = 10.0
            score = 2.718281828 ** (-distance / scale)

            scores[area_id] = score

        return scores

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about stored fingerprints."""
        stats: dict[str, Any] = {
            "total_areas": len(self._fingerprints),
            "total_samples": sum(len(fp.samples) for fp in self._fingerprints.values()),
            "areas": {},
        }

        for area_id, area_fp in self._fingerprints.items():
            stats["areas"][area_id] = {
                "name": area_fp.area_name,
                "samples": len(area_fp.samples),
                "scanners": list(area_fp.get_mean_rssi().keys()),
                "mean_rssi": {k: round(v, 1) for k, v in area_fp.get_mean_rssi().items()},
            }

        return stats

    # --- Persistence methods ---

    def _get_storage_path(self) -> Path | None:
        """Get the path to the storage file."""
        if self.hass is None:
            return None
        return Path(self.hass.config.path(STORAGE_DIR)) / STORAGE_FILE

    def _load(self) -> None:
        """Load fingerprint data from persistent storage."""
        path = self._get_storage_path()
        if path is None or not path.exists():
            _LOGGER.debug("No fingerprint data to load")
            return

        try:
            with path.open() as f:
                data = json.load(f)

            if data.get("version") != STORAGE_VERSION:
                _LOGGER.warning("Fingerprint data version mismatch, starting fresh")
                return

            # Restore fingerprints
            for area_id, area_data in data.get("fingerprints", {}).items():
                self._fingerprints[area_id] = AreaFingerprints.from_dict(area_data)

            total_samples = sum(len(fp.samples) for fp in self._fingerprints.values())
            _LOGGER.info(
                "Loaded fingerprint data: %d areas, %d total samples",
                len(self._fingerprints),
                total_samples,
            )

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            _LOGGER.warning("Failed to load fingerprint data: %s", e)

    def save(self) -> None:
        """Save fingerprint data to persistent storage."""
        if not self._dirty:
            return

        path = self._get_storage_path()
        if path is None:
            return

        try:
            path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "version": STORAGE_VERSION,
                "fingerprints": {area_id: fp.to_dict() for area_id, fp in self._fingerprints.items()},
            }

            with path.open("w") as f:
                json.dump(data, f, indent=2)

            self._dirty = False
            _LOGGER.debug("Saved fingerprint data")

        except OSError:
            _LOGGER.exception("Failed to save fingerprint data")

    def clear(self) -> None:
        """Clear all stored fingerprints."""
        self._fingerprints.clear()
        self._dirty = True
        _LOGGER.info("Cleared all fingerprint data")

    def clear_area(self, area_id: str) -> None:
        """Clear fingerprints for a specific area."""
        if area_id in self._fingerprints:
            del self._fingerprints[area_id]
            self._dirty = True
            _LOGGER.info("Cleared fingerprint data for area %s", area_id)
