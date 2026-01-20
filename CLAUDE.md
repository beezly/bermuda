# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Bermuda is a Home Assistant custom integration for room-level Bluetooth device tracking. It determines which room/area a BLE device is in by analyzing signal strength (RSSI) from multiple Bluetooth proxy devices (ESPHome, Shelly) distributed around the home.

## Commands

### Testing
```bash
pytest                           # Run all tests
pytest tests/test_util.py        # Run a single test file
pytest tests/test_util.py::test_rssi_to_metres  # Run a single test
pytest -v                        # Verbose output
```

### Linting
```bash
ruff check .                     # Check for linting errors
ruff check . --fix               # Auto-fix linting errors
ruff format .                    # Format code
```

### Pre-commit
```bash
pre-commit run --all-files       # Run all pre-commit hooks
```

## Architecture

### Core Components (`custom_components/bermuda/`)

**Data Flow:** BLE advertisements → `coordinator.py` → `BermudaDevice`/`BermudaAdvert` objects → Sensor entities update

- **coordinator.py** - Central processing engine (`BermudaDataUpdateCoordinator`)
  - Receives BLE advertisements from HA's bluetooth manager
  - Implements area detection algorithm (`_refresh_area_by_min_distance`)
  - Manages device lifecycle and pruning
  - Runs processing loop every ~1 second (`UPDATE_INTERVAL`)

- **bermuda_device.py** - `BermudaDevice` class representing both:
  - Transmitting devices (phones, beacons, thermometers)
  - Receiving devices (scanners/proxies)
  - Tracks area assignments, RSSI calibration, device metadata

- **bermuda_advert.py** - `BermudaAdvert` class for scanner↔device relationships
  - Stores RSSI measurements and converts to distance estimates
  - Maintains measurement history for smoothing algorithms
  - Uses path-loss model: `distance = 10^((ref_power - rssi) / (10 * attenuation))`

- **bermuda_irk.py** - Identity Resolving Key management
  - Resolves randomized MAC addresses (iOS/Android privacy feature)
  - Integrates with Home Assistant's Private BLE Device component

### Entity Platforms

- **sensor.py** - Area, distance, RSSI, floor sensors
- **device_tracker.py** - Home/Away presence tracking
- **number.py** - Per-device RSSI reference power calibration

### Key Concepts

- **Metadevices**: Virtual devices created for tracking protocols with randomized MACs (iBeacon, Private BLE). The physical device (source) maps to a stable metadevice.

- **Area Detection Algorithm**: Uses hysteresis to prevent "bouncing" between areas. Considers signal freshness, distance thresholds (15-30% difference required), and same-area preferences.

- **Device Pruning**: Memory management removes stale device entries (tracked devices and scanners are never pruned). IRK addresses pruned at 4-16 minutes, regular devices at 1 day.

### Configuration Constants (`const.py`)

Key tunable values:
- `UPDATE_INTERVAL` (1.05s) - Internal BLE processing cycle
- `CONF_UPDATE_INTERVAL` (default 10s) - Sensor update frequency
- `DEFAULT_MAX_RADIUS` (20m) - Maximum area detection distance
- `DEFAULT_REF_POWER` (-55 dBm) - RSSI at 1 meter reference
- `DEFAULT_ATTENUATION` (3) - Environmental signal attenuation factor

## Testing Notes

Tests use `pytest-homeassistant-custom-component` which provides HA test fixtures. The `conftest.py` sets up automatic bluetooth mocking and custom integration loading.
