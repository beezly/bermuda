"""Abstract base class and shared types for area selection algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from bluetooth_data_tools import monotonic_time_coarse

from custom_components.bermuda.const import AREA_MAX_AD_AGE, CONF_MAX_RADIUS, DEFAULT_MAX_RADIUS

if TYPE_CHECKING:
    from custom_components.bermuda.bermuda_advert import BermudaAdvert
    from custom_components.bermuda.bermuda_device import BermudaDevice


@dataclass
class AreaSelectionResult:
    """Result from an area selection algorithm."""

    winning_advert: BermudaAdvert | None
    reason: str | None = None

    # Diagnostic fields for sensortext()
    device: str = ""
    scannername: tuple[str, str] = ("", "")
    areas: tuple[str, str] = ("", "")
    pcnt_diff: float = 0
    same_area: bool = False
    last_ad_age: tuple[float, float] = (0, 0)
    this_ad_age: tuple[float, float] = (0, 0)
    distance: tuple[float, float] = (0, 0)
    hist_min_max: tuple[float, float] = (0, 0)

    def to_diagnostic_text(self) -> str:
        """Return a text summary suitable for use in a sensor entity."""
        out = ""
        for var, val in vars(self).items():
            if var == "winning_advert":
                # Skip the winning_advert object itself
                continue
            out += f"{var}|"
            if isinstance(val, tuple):
                for v in val:
                    if isinstance(v, float):
                        out += f"{v:.2f}|"
                    else:
                        out += f"{v}"
            elif var == "pcnt_diff":
                out += f"{val:.3f}"
            else:
                out += f"{val}"
            out += "\n"
        return out[:255]

    def __str__(self) -> str:
        """Create string representation for debug logging/dumping."""
        out = ""
        for var, val in vars(self).items():
            if var == "winning_advert":
                continue
            out += f"** {var:20} "
            if isinstance(val, tuple):
                for v in val:
                    if isinstance(v, float):
                        out += f"{v:.2f} "
                    else:
                        out += f"{v} "
                out += "\n"
            elif var == "pcnt_diff":
                out += f"{val:.3f}\n"
            else:
                out += f"{val}\n"
        return out


@dataclass
class AreaSelectorConfig:
    """Configuration for area selection algorithms."""

    max_radius: float = DEFAULT_MAX_RADIUS
    max_ad_age: float = AREA_MAX_AD_AGE

    # Algorithm-specific parameters with defaults
    # These are used by MinDistanceSelector but defined here for compatibility
    pdiff_outright: float = 0.30  # Percentage difference to win outright / instantly
    pdiff_historical: float = 0.15  # Percentage difference required to win on historical test
    min_history: int = 3  # Minimum history entries required for historical test
    history_window: int = 5  # Time period to compare history between incumbent and challenger

    # Additional fields that can be added by derived configs
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_options(cls, options: dict) -> AreaSelectorConfig:
        """Create config from coordinator options dict."""
        return cls(
            max_radius=options.get(CONF_MAX_RADIUS, DEFAULT_MAX_RADIUS),
        )


class AreaSelectorBase(ABC):
    """Abstract base class for area selection algorithms."""

    # Subclasses must define these
    SELECTOR_ID: str = ""
    SELECTOR_NAME: str = ""

    def __init__(self, config: AreaSelectorConfig) -> None:
        """Initialize the selector with configuration."""
        self.config = config

    @abstractmethod
    def select_area(self, device: BermudaDevice, current_stamp: float) -> AreaSelectionResult:
        """
        Select the best area for a device based on available adverts.

        Args:
            device: The BermudaDevice to find the area for
            current_stamp: Current monotonic timestamp

        Returns:
            AreaSelectionResult containing the winning advert and diagnostic info

        """

    def validate_advert(
        self,
        advert: BermudaAdvert,
        current_stamp: float,
    ) -> bool:
        """
        Check if an advert is valid for area selection.

        Shared validation logic that can be used by all selectors.

        Args:
            advert: The BermudaAdvert to validate
            current_stamp: Current monotonic timestamp

        Returns:
            True if the advert is valid for consideration

        """
        # Reject stale adverts
        if advert.stamp < current_stamp - self.config.max_ad_age:
            return False

        # Reject adverts without distance or area
        if advert.rssi_distance is None:
            return False

        if advert.rssi_distance > self.config.max_radius:
            return False

        return advert.area_id is not None

    def get_current_stamp(self) -> float:
        """Get the current monotonic timestamp."""
        return monotonic_time_coarse()
