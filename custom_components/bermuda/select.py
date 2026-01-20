"""Create Select entities for area training."""

from __future__ import annotations

from typing import TYPE_CHECKING

from homeassistant.components.select import SelectEntity
from homeassistant.const import EntityCategory
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import area_registry as ar
from homeassistant.helpers.dispatcher import async_dispatcher_connect

from .const import _LOGGER, SIGNAL_DEVICE_NEW
from .entity import BermudaEntity

if TYPE_CHECKING:
    from homeassistant.helpers.entity_platform import AddEntitiesCallback

    from . import BermudaConfigEntry
    from .coordinator import BermudaDataUpdateCoordinator


# Special option value for "none selected"
OPTION_NONE = "none"


async def async_setup_entry(
    hass: HomeAssistant,
    entry: BermudaConfigEntry,
    async_add_devices: AddEntitiesCallback,
) -> None:
    """Load Select entities for a config entry."""
    coordinator: BermudaDataUpdateCoordinator = entry.runtime_data.coordinator

    created_devices: list[str] = []

    @callback
    def device_new(address: str) -> None:
        """Create entities for newly-found device."""
        if address not in created_devices:
            entities = []
            entities.append(BermudaTrainAreaSelect(coordinator, entry, address))
            async_add_devices(entities, False)
            created_devices.append(address)
        coordinator.select_created(address)

    entry.async_on_unload(async_dispatcher_connect(hass, SIGNAL_DEVICE_NEW, device_new))


class BermudaTrainAreaSelect(BermudaEntity, SelectEntity):
    """A Select entity for training device area location."""

    _attr_should_poll = False
    _attr_has_entity_name = True
    _attr_name = "Train Location"
    _attr_translation_key = "train_location"
    _attr_entity_category = EntityCategory.CONFIG
    _attr_entity_registry_enabled_default = False  # Disabled by default
    _attr_icon = "mdi:map-marker-plus"

    def __init__(
        self,
        coordinator: BermudaDataUpdateCoordinator,
        entry: BermudaConfigEntry,
        address: str,
    ) -> None:
        """Initialize the select entity."""
        super().__init__(coordinator, entry, address)
        self._attr_current_option = OPTION_NONE
        self._area_registry = ar.async_get(coordinator.hass)

    @property
    def options(self) -> list[str]:
        """Return list of available areas plus 'none'."""
        return [OPTION_NONE, *[area.id for area in self._area_registry.async_list_areas()]]

    @property
    def current_option(self) -> str | None:
        """Return current selected option."""
        return self._attr_current_option

    async def async_select_option(self, option: str) -> None:
        """Handle option selection - train the model with this area."""
        if option == OPTION_NONE:
            self._attr_current_option = OPTION_NONE
            self.async_write_ha_state()
            return

        # Get the area info
        area = self._area_registry.async_get_area(option)
        if area is None:
            _LOGGER.warning("Area %s not found", option)
            return

        # Train the model using coordinator's method
        result = self.coordinator.train_device_location(
            device_address=self.address,
            area_id=area.id,
        )

        if not result.get("success", False):
            _LOGGER.warning(
                "Failed to train location for %s: %s",
                self.address,
                result.get("error", "Unknown error"),
            )
            return

        _LOGGER.info(
            "Trained location for %s in %s: %d fingerprint readings, %d transition recorded",
            result.get("device", self.address),
            area.name,
            result.get("fingerprints_recorded", 0),
            result.get("transition_recorded", 0),
        )

        # Reset to "none" after training
        self._attr_current_option = OPTION_NONE
        self.async_write_ha_state()

    @property
    def unique_id(self) -> str:
        """Return unique ID for this entity."""
        return f"{self._device.unique_id}_train_location"

    @property
    def extra_state_attributes(self) -> dict:
        """Return extra state attributes."""
        return {
            "help": "Select an area to train the location model with the device's current RSSI readings",
        }
