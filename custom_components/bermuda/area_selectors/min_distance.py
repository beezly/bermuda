"""Minimum distance area selection algorithm."""

from __future__ import annotations

from typing import TYPE_CHECKING

from custom_components.bermuda.const import _LOGGER

from .base import AreaSelectionResult, AreaSelectorBase, AreaSelectorConfig

if TYPE_CHECKING:
    from custom_components.bermuda.bermuda_advert import BermudaAdvert
    from custom_components.bermuda.bermuda_device import BermudaDevice


class MinDistanceSelector(AreaSelectorBase):
    """
    Area selector that chooses the area with the minimum distance.

    This is the original Bermuda algorithm that uses hysteresis and
    historical comparison to prevent bouncing between areas.
    """

    SELECTOR_ID = "min_distance"
    SELECTOR_NAME = "Minimum Distance"

    def __init__(self, config: AreaSelectorConfig) -> None:
        """Initialize the minimum distance selector."""
        super().__init__(config)
        # Enable verbose logging for specific devices (for debugging)
        self._superchatty_devices: set[str] = set()

    def select_area(self, device: BermudaDevice, current_stamp: float) -> AreaSelectionResult:
        """
        Select area for a device based on closest scanner/proxy.

        Uses hysteresis to prevent bouncing between areas by requiring
        significant distance differences and considering historical readings.
        """
        # The current area_advert (which might be None) is the one to beat
        incumbent: BermudaAdvert | None = device.area_advert

        result = AreaSelectionResult(winning_advert=incumbent)
        result.device = device.name

        superchatty = device.name in self._superchatty_devices

        for challenger in device.adverts.values():
            # Check each scanner and any time one is found to be closer/better
            # than the existing incumbent, replace it.

            # Skip self-comparison
            if incumbent is challenger:
                continue

            # Validate the challenger
            if not self.validate_advert(challenger, current_stamp):
                continue

            # If incumbent is invalid, challenger wins by default
            if not self._is_valid_incumbent(incumbent):
                incumbent = challenger
                if superchatty:
                    _LOGGER.debug(
                        "%s IS closest to %s: Incumbent is invalid",
                        device.name,
                        challenger.name,
                    )
                continue

            # From here, both incumbent and challenger are valid
            # The challenger must be closer to even be considered
            if incumbent.rssi_distance < challenger.rssi_distance:  # type: ignore[union-attr]
                continue

            # Build test data for this comparison
            result.reason = None
            result.same_area = incumbent.area_id == challenger.area_id
            result.areas = (incumbent.area_name or "", challenger.area_name or "")
            result.scannername = (incumbent.name, challenger.name)
            result.distance = (incumbent.rssi_distance, challenger.rssi_distance)  # type: ignore[assignment]

            # How recently have we heard from the scanners?
            result.last_ad_age = (
                current_stamp - incumbent.scanner_device.last_seen,
                current_stamp - challenger.scanner_device.last_seen,
            )

            # How old are the ads?
            result.this_ad_age = (
                current_stamp - incumbent.stamp,
                current_stamp - challenger.stamp,
            )

            # Calculate percentage difference between distances
            result.pcnt_diff = self._calculate_percentage_diff(
                challenger.rssi_distance,
                incumbent.rssi_distance,  # type: ignore[arg-type]
            )

            # Check same-area win condition
            if self._check_same_area_win(result):
                result.reason = "WIN awarded for same area, newer, closer advert"
                incumbent = challenger
                continue

            # Check historical win condition
            if self._check_historical_win(challenger, incumbent, result):
                result.reason = "WIN on historical min/max"
                incumbent = challenger
                continue

            # Check outright percentage difference win
            if result.pcnt_diff < self.config.pdiff_outright:
                result.reason = "LOSS - failed on percentage_difference"
                continue

            # If we made it through all checks, challenger wins
            result.reason = "WIN by not losing!"
            incumbent = challenger

        if superchatty and result.reason is not None:
            _LOGGER.info(
                "***************\n**************** %s *******************\n%s",
                result.reason,
                result,
            )

        result.winning_advert = incumbent
        return result

    def _is_valid_incumbent(self, incumbent: BermudaAdvert | None) -> bool:
        """Check if the incumbent advert has valid data for comparison."""
        if incumbent is None:
            return False
        if incumbent.rssi_distance is None:
            return False
        return incumbent.area_id is not None

    def _calculate_percentage_diff(self, distance_a: float, distance_b: float) -> float:
        """Calculate the percentage difference between two distances."""
        return abs(distance_a - distance_b) / ((distance_a + distance_b) / 2)

    def _check_same_area_win(self, result: AreaSelectionResult) -> bool:
        """
        Check if challenger wins based on same-area, newer, closer criteria.

        Returns True if challenger should win.
        """
        return (
            result.same_area
            and (result.this_ad_age[0] > result.this_ad_age[1] + 1)
            and result.distance[0] >= result.distance[1]
        )

    def _check_historical_win(
        self,
        challenger: BermudaAdvert,
        incumbent: BermudaAdvert,
        result: AreaSelectionResult,
    ) -> bool:
        """
        Check if challenger wins based on historical distance comparison.

        If the challenger's worst reading in the history window is still closer
        than the incumbent's best reading in that time, and the percentage
        difference exceeds the threshold, the challenger wins.

        Returns True if challenger should win.
        """
        if len(challenger.hist_distance_by_interval) <= self.config.min_history:
            return False

        # Get historical min/max
        incumbent_min = min(incumbent.hist_distance_by_interval[: self.config.history_window])
        challenger_max = max(challenger.hist_distance_by_interval[: self.config.history_window])

        result.hist_min_max = (incumbent_min, challenger_max)

        # Challenger's worst must be better than incumbent's best,
        # and percentage difference must exceed threshold
        return challenger_max < incumbent_min and result.pcnt_diff > self.config.pdiff_historical

    def enable_verbose_logging(self, device_name: str) -> None:
        """Enable verbose logging for a specific device name."""
        self._superchatty_devices.add(device_name)

    def disable_verbose_logging(self, device_name: str) -> None:
        """Disable verbose logging for a specific device name."""
        self._superchatty_devices.discard(device_name)
