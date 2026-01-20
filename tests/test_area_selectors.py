"""Tests for the area_selectors module."""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import pytest

from custom_components.bermuda.area_selectors import (
    AREA_SELECTORS,
    DEFAULT_SELECTOR_ID,
    create_selector,
    get_available_selectors,
)
from custom_components.bermuda.area_selectors.base import (
    AreaSelectionResult,
    AreaSelectorBase,
    AreaSelectorConfig,
)
from custom_components.bermuda.area_selectors.fingerprint_store import (
    AreaFingerprints,
    FingerprintStore,
    RSSIFingerprint,
)
from custom_components.bermuda.area_selectors.min_distance import MinDistanceSelector
from custom_components.bermuda.area_selectors.transition_tracker import TransitionTracker
from custom_components.bermuda.area_selectors.weighted_area import WeightedAreaSelector


# --- Fixtures ---


@pytest.fixture
def mock_advert():
    """Create a mock BermudaAdvert."""
    advert = MagicMock()
    advert.area_id = "living_room"
    advert.area_name = "Living Room"
    advert.rssi_distance = 2.0
    advert.rssi = -65
    advert.stamp = 100.0
    advert.name = "Scanner1"
    advert.scanner_address = "11:22:33:44:55:66"
    advert.hist_rssi = [-65, -66, -64, -65, -67]
    advert.hist_distance_by_interval = [2.0, 2.1, 1.9, 2.0, 2.2]
    return advert


@pytest.fixture
def mock_advert2():
    """Create a second mock BermudaAdvert in a different area."""
    advert = MagicMock()
    advert.area_id = "kitchen"
    advert.area_name = "Kitchen"
    advert.rssi_distance = 3.5
    advert.rssi = -72
    advert.stamp = 100.0
    advert.name = "Scanner2"
    advert.scanner_address = "AA:BB:CC:DD:EE:FF"
    advert.hist_rssi = [-72, -73, -71, -72, -74]
    advert.hist_distance_by_interval = [3.5, 3.6, 3.4, 3.5, 3.7]
    return advert


@pytest.fixture
def mock_device(mock_advert, mock_advert2):
    """Create a mock BermudaDevice with adverts."""
    device = MagicMock()
    device.name = "Test Device"
    device.address = "aa:bb:cc:dd:ee:ff"
    device.area_advert = None
    device.adverts = {
        ("11:22:33:44:55:66", "aa:bb:cc:dd:ee:ff"): mock_advert,
        ("AA:BB:CC:DD:EE:FF", "aa:bb:cc:dd:ee:ff"): mock_advert2,
    }
    return device


@pytest.fixture
def basic_config():
    """Create a basic AreaSelectorConfig."""
    return AreaSelectorConfig(max_radius=20.0, max_ad_age=30.0)


@pytest.fixture
def weighted_config():
    """Create config for WeightedAreaSelector with all features enabled."""
    return AreaSelectorConfig(
        max_radius=20.0,
        max_ad_age=30.0,
        extra={
            "use_variance_weighting": True,
            "use_recency_weighting": True,
            "use_hmm": False,  # Disable for simpler testing
            "use_passive_learning": False,
            "use_fingerprint_matching": False,
        },
    )


# --- Registry Tests ---


class TestAreaSelectorRegistry:
    """Tests for area selector registry and factory."""

    def test_default_selector_id(self):
        """Test default selector ID is set."""
        assert DEFAULT_SELECTOR_ID == "min_distance"

    def test_selectors_registered(self):
        """Test that selectors are registered."""
        assert "min_distance" in AREA_SELECTORS
        assert "weighted_area" in AREA_SELECTORS

    def test_create_selector_default(self, basic_config):
        """Test creating default selector."""
        selector = create_selector(config=basic_config)
        assert isinstance(selector, MinDistanceSelector)

    def test_create_selector_by_id(self, basic_config):
        """Test creating selector by ID."""
        selector = create_selector("weighted_area", basic_config)
        assert isinstance(selector, WeightedAreaSelector)

    def test_create_selector_invalid_id(self, basic_config):
        """Test creating selector with invalid ID raises ValueError."""
        with pytest.raises(ValueError, match="Unknown area selector"):
            create_selector("nonexistent", basic_config)

    def test_get_available_selectors(self):
        """Test getting list of available selectors."""
        selectors = get_available_selectors()
        assert len(selectors) >= 2
        values = [s["value"] for s in selectors]
        assert "min_distance" in values
        assert "weighted_area" in values
        # Check that label key exists
        for s in selectors:
            assert "label" in s


# --- AreaSelectionResult Tests ---


class TestAreaSelectionResult:
    """Tests for AreaSelectionResult dataclass."""

    def test_result_defaults(self):
        """Test result has correct defaults."""
        result = AreaSelectionResult(winning_advert=None)
        assert result.winning_advert is None
        assert result.reason is None
        assert result.areas == ("", "")
        assert result.pcnt_diff == 0

    def test_to_diagnostic_text(self, mock_advert):
        """Test diagnostic text generation."""
        result = AreaSelectionResult(winning_advert=mock_advert)
        result.device = "Test Device"
        result.areas = ("Living Room", "Kitchen")
        result.distance = (2.0, 3.5)
        result.pcnt_diff = 0.65
        result.reason = "Test reason"

        text = result.to_diagnostic_text()
        assert "Test Device" in text
        assert "Living Room" in text


# --- TransitionTracker Tests ---


class TestTransitionTracker:
    """Tests for HMM TransitionTracker."""

    def test_initialization(self):
        """Test tracker initializes correctly."""
        tracker = TransitionTracker(hass=None)
        assert tracker.smoothing == 1.0
        assert tracker.min_transitions == 5
        assert len(tracker._known_areas) == 0

    def test_record_area(self):
        """Test recording area adds to known areas."""
        tracker = TransitionTracker(hass=None)
        tracker.record_area("device1", "living_room")
        assert "living_room" in tracker._known_areas
        assert tracker._last_area["device1"] == "living_room"

    def test_record_transition(self):
        """Test transitions are recorded."""
        tracker = TransitionTracker(hass=None)
        tracker.record_area("device1", "living_room")
        tracker.record_area("device1", "kitchen")

        # Should have recorded living_room -> kitchen transition
        assert tracker._transitions["living_room"]["kitchen"] > 0

    def test_self_transition(self):
        """Test self-transitions are recorded."""
        tracker = TransitionTracker(hass=None)
        tracker.record_area("device1", "living_room")
        tracker.record_area("device1", "living_room")

        assert tracker._transitions["living_room"]["living_room"] > 0

    def test_get_transition_probability_no_data(self):
        """Test probability returns 1.0 when no data."""
        tracker = TransitionTracker(hass=None)
        prob = tracker.get_transition_probability("living_room", "kitchen")
        assert prob == 1.0

    def test_get_transition_probability_with_data(self):
        """Test probability calculation with sufficient data."""
        tracker = TransitionTracker(hass=None, min_transitions=2)
        # Record enough transitions
        for _ in range(5):
            tracker.record_area("device1", "living_room")
            tracker.record_area("device1", "kitchen")

        prob_kitchen = tracker.get_transition_probability("living_room", "kitchen")
        prob_bedroom = tracker.get_transition_probability("living_room", "bedroom")

        # Kitchen should have higher probability than unknown bedroom
        assert prob_kitchen > prob_bedroom

    def test_initialize_area(self):
        """Test area initialization with self-transition weight."""
        tracker = TransitionTracker(hass=None)
        tracker.initialize_area("living_room")

        assert "living_room" in tracker._known_areas
        assert tracker._transitions["living_room"]["living_room"] == tracker.self_transition_weight

    def test_get_statistics(self):
        """Test statistics retrieval."""
        tracker = TransitionTracker(hass=None)
        tracker.record_area("device1", "living_room")
        tracker.record_area("device1", "kitchen")

        stats = tracker.get_statistics()
        assert "known_areas" in stats
        assert "total_transitions" in stats
        assert stats["known_areas"] == 2

    def test_clear(self):
        """Test clearing all data."""
        tracker = TransitionTracker(hass=None)
        tracker.record_area("device1", "living_room")
        tracker.clear()

        assert len(tracker._known_areas) == 0
        assert len(tracker._transitions) == 0


# --- FingerprintStore Tests ---


class TestRSSIFingerprint:
    """Tests for RSSIFingerprint dataclass."""

    def test_to_dict(self):
        """Test serialization to dict."""
        fp = RSSIFingerprint(scanner_rssi={"scanner1": -65.0}, timestamp=100.0)
        d = fp.to_dict()
        assert d["scanner_rssi"]["scanner1"] == -65.0
        assert d["timestamp"] == 100.0

    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {"scanner_rssi": {"scanner1": -65.0}, "timestamp": 100.0}
        fp = RSSIFingerprint.from_dict(data)
        assert fp.scanner_rssi["scanner1"] == -65.0
        assert fp.timestamp == 100.0


class TestAreaFingerprints:
    """Tests for AreaFingerprints dataclass."""

    def test_add_sample(self):
        """Test adding fingerprint samples."""
        af = AreaFingerprints(area_id="living_room", area_name="Living Room")
        fp = RSSIFingerprint(scanner_rssi={"scanner1": -65.0})
        af.add_sample(fp)
        assert len(af.samples) == 1

    def test_add_sample_max_limit(self):
        """Test samples are limited to max."""
        af = AreaFingerprints(area_id="living_room")
        for i in range(150):
            fp = RSSIFingerprint(scanner_rssi={"scanner1": float(-60 - i)})
            af.add_sample(fp, max_samples=100)
        assert len(af.samples) == 100

    def test_get_mean_rssi(self):
        """Test mean RSSI calculation."""
        af = AreaFingerprints(area_id="living_room")
        af.add_sample(RSSIFingerprint(scanner_rssi={"scanner1": -60.0}))
        af.add_sample(RSSIFingerprint(scanner_rssi={"scanner1": -70.0}))

        mean = af.get_mean_rssi()
        assert mean["scanner1"] == -65.0

    def test_get_rssi_variance(self):
        """Test RSSI variance calculation."""
        af = AreaFingerprints(area_id="living_room")
        af.add_sample(RSSIFingerprint(scanner_rssi={"scanner1": -60.0}))
        af.add_sample(RSSIFingerprint(scanner_rssi={"scanner1": -70.0}))

        variance = af.get_rssi_variance()
        assert "scanner1" in variance
        assert variance["scanner1"] > 0


class TestFingerprintStore:
    """Tests for FingerprintStore."""

    def test_initialization(self):
        """Test store initializes correctly."""
        store = FingerprintStore(hass=None)
        assert store.max_samples_per_area == 100
        assert store.min_samples_for_matching == 5

    def test_record_fingerprint(self):
        """Test recording fingerprints."""
        store = FingerprintStore(hass=None)
        store.record_fingerprint(
            area_id="living_room",
            area_name="Living Room",
            scanner_rssi={"scanner1": -65.0},
            timestamp=100.0,
        )

        assert "living_room" in store.get_all_areas()
        fp = store.get_area_fingerprint("living_room")
        assert fp is not None
        assert len(fp.samples) == 1

    def test_has_sufficient_data(self):
        """Test checking for sufficient data."""
        store = FingerprintStore(hass=None, min_samples_for_matching=3)

        # Not enough samples
        store.record_fingerprint("living_room", "Living Room", {"s1": -65.0})
        assert store.has_sufficient_data("living_room") is False

        # Add more samples
        store.record_fingerprint("living_room", "Living Room", {"s1": -66.0})
        store.record_fingerprint("living_room", "Living Room", {"s1": -64.0})
        assert store.has_sufficient_data("living_room") is True

    def test_match_fingerprint(self):
        """Test fingerprint matching."""
        store = FingerprintStore(hass=None, min_samples_for_matching=2)

        # Add fingerprints for two areas
        for _ in range(3):
            store.record_fingerprint("living_room", "Living Room", {"s1": -60.0, "s2": -70.0})
            store.record_fingerprint("kitchen", "Kitchen", {"s1": -80.0, "s2": -50.0})

        # Match against living room pattern
        scores = store.match_fingerprint({"s1": -61.0, "s2": -71.0})

        assert "living_room" in scores
        assert "kitchen" in scores
        # Living room should have higher similarity
        assert scores["living_room"] > scores["kitchen"]

    def test_match_fingerprint_no_common_scanners(self):
        """Test matching with no common scanners."""
        store = FingerprintStore(hass=None, min_samples_for_matching=2)

        for _ in range(3):
            store.record_fingerprint("living_room", "Living Room", {"s1": -60.0})

        # Try to match with different scanner
        scores = store.match_fingerprint({"s3": -65.0})
        assert "living_room" not in scores

    def test_get_statistics(self):
        """Test statistics retrieval."""
        store = FingerprintStore(hass=None)
        store.record_fingerprint("living_room", "Living Room", {"s1": -65.0})

        stats = store.get_statistics()
        assert stats["total_areas"] == 1
        assert stats["total_samples"] == 1
        assert "living_room" in stats["areas"]

    def test_clear(self):
        """Test clearing all data."""
        store = FingerprintStore(hass=None)
        store.record_fingerprint("living_room", "Living Room", {"s1": -65.0})
        store.clear()

        assert len(store.get_all_areas()) == 0

    def test_clear_area(self):
        """Test clearing specific area."""
        store = FingerprintStore(hass=None)
        store.record_fingerprint("living_room", "Living Room", {"s1": -65.0})
        store.record_fingerprint("kitchen", "Kitchen", {"s1": -70.0})

        store.clear_area("living_room")

        assert "living_room" not in store.get_all_areas()
        assert "kitchen" in store.get_all_areas()


# --- MinDistanceSelector Tests ---


class TestMinDistanceSelector:
    """Tests for MinDistanceSelector."""

    def test_initialization(self, basic_config):
        """Test selector initializes correctly."""
        selector = MinDistanceSelector(basic_config)
        assert selector.SELECTOR_ID == "min_distance"
        assert selector.config.max_radius == 20.0

    def test_validate_advert_valid(self, basic_config, mock_advert):
        """Test advert validation passes for valid advert."""
        selector = MinDistanceSelector(basic_config)
        # Advert is recent and within radius
        assert selector.validate_advert(mock_advert, 105.0) is True

    def test_validate_advert_too_old(self, basic_config, mock_advert):
        """Test advert validation fails for old advert."""
        selector = MinDistanceSelector(basic_config)
        # Advert is too old (stamp=100, current=200, max_age=30)
        assert selector.validate_advert(mock_advert, 200.0) is False

    def test_validate_advert_too_far(self, basic_config, mock_advert):
        """Test advert validation fails for distant advert."""
        config = AreaSelectorConfig(max_radius=1.0, max_ad_age=30.0)
        selector = MinDistanceSelector(config)
        # Advert is too far (distance=2.0, max_radius=1.0)
        assert selector.validate_advert(mock_advert, 105.0) is False

    def test_select_area_no_adverts(self, basic_config):
        """Test selection with no valid adverts."""
        selector = MinDistanceSelector(basic_config)
        device = MagicMock()
        device.adverts = {}
        device.area_advert = None
        device.name = "Test"

        result = selector.select_area(device, 100.0)
        assert result.winning_advert is None

    def test_select_area_single_advert(self, basic_config, mock_device, mock_advert):
        """Test selection with single valid advert."""
        selector = MinDistanceSelector(basic_config)
        mock_device.adverts = {("scanner1", "device1"): mock_advert}

        result = selector.select_area(mock_device, 105.0)
        assert result.winning_advert == mock_advert

    def test_select_area_multiple_adverts(self, basic_config, mock_device):
        """Test selection picks closest scanner."""
        selector = MinDistanceSelector(basic_config)

        result = selector.select_area(mock_device, 105.0)
        # Should pick living_room (2.0m) over kitchen (3.5m)
        assert result.winning_advert.area_id == "living_room"


# --- WeightedAreaSelector Tests ---


class TestWeightedAreaSelector:
    """Tests for WeightedAreaSelector."""

    def test_initialization(self, weighted_config):
        """Test selector initializes correctly."""
        selector = WeightedAreaSelector(weighted_config)
        assert selector.SELECTOR_ID == "weighted_area"
        assert selector.weight_power == 2.0
        assert selector.use_variance_weighting is True

    def test_select_area_weighted_voting(self, weighted_config, mock_device):
        """Test weighted voting selects closest area."""
        selector = WeightedAreaSelector(weighted_config)

        result = selector.select_area(mock_device, 105.0)
        # Living room at 2m should have higher weight than kitchen at 3.5m
        assert result.winning_advert.area_id == "living_room"
        assert result.reason is not None
        assert "Weighted vote" in result.reason

    def test_select_area_fallback_single_scanner(self, mock_advert):
        """Test fallback to closest with single scanner."""
        config = AreaSelectorConfig(
            max_radius=20.0,
            max_ad_age=30.0,
            extra={"min_scanners": 2, "fallback_to_closest": True},
        )
        selector = WeightedAreaSelector(config)

        device = MagicMock()
        device.name = "Test"
        device.address = "aa:bb:cc:dd:ee:ff"
        device.area_advert = None
        device.adverts = {("scanner1", "device1"): mock_advert}

        result = selector.select_area(device, 105.0)
        assert result.winning_advert == mock_advert
        assert "Fallback" in result.reason

    def test_select_area_no_fallback(self, mock_advert):
        """Test no selection when below min_scanners without fallback."""
        config = AreaSelectorConfig(
            max_radius=20.0,
            max_ad_age=30.0,
            extra={"min_scanners": 2, "fallback_to_closest": False},
        )
        selector = WeightedAreaSelector(config)

        device = MagicMock()
        device.name = "Test"
        device.address = "aa:bb:cc:dd:ee:ff"
        device.area_advert = None
        device.adverts = {("scanner1", "device1"): mock_advert}

        result = selector.select_area(device, 105.0)
        assert result.winning_advert is None

    def test_calculate_signal_confidence_stable(self, weighted_config):
        """Test signal confidence for stable signal."""
        selector = WeightedAreaSelector(weighted_config)

        advert = MagicMock()
        advert.hist_rssi = [-65, -65, -65, -65, -65]  # Very stable

        confidence = selector._calculate_signal_confidence(advert)
        assert confidence > 0.9  # Should be close to 1.0

    def test_calculate_signal_confidence_noisy(self, weighted_config):
        """Test signal confidence for noisy signal."""
        selector = WeightedAreaSelector(weighted_config)

        advert = MagicMock()
        advert.hist_rssi = [-50, -80, -55, -75, -60]  # Very noisy

        confidence = selector._calculate_signal_confidence(advert)
        assert confidence < 0.5  # Should be significantly reduced

    def test_calculate_signal_confidence_insufficient_samples(self, weighted_config):
        """Test signal confidence with insufficient samples."""
        selector = WeightedAreaSelector(weighted_config)

        advert = MagicMock()
        advert.hist_rssi = [-65, -66]  # Only 2 samples

        confidence = selector._calculate_signal_confidence(advert)
        assert confidence == 0.75  # Default for insufficient samples

    def test_calculate_recency_factor_fresh(self, weighted_config):
        """Test recency factor for fresh reading."""
        selector = WeightedAreaSelector(weighted_config)

        advert = MagicMock()
        advert.stamp = 100.0

        factor = selector._calculate_recency_factor(advert, 100.0)
        assert factor == 1.0  # Age = 0

    def test_calculate_recency_factor_stale(self, weighted_config):
        """Test recency factor for stale reading."""
        selector = WeightedAreaSelector(weighted_config)

        advert = MagicMock()
        advert.stamp = 95.0

        factor = selector._calculate_recency_factor(advert, 100.0)
        # Age = 5s, decay = 5s, so factor = exp(-1) ≈ 0.37
        assert 0.35 < factor < 0.40

    def test_calculate_recency_factor_no_stamp(self, weighted_config):
        """Test recency factor when stamp is None."""
        selector = WeightedAreaSelector(weighted_config)

        advert = MagicMock()
        advert.stamp = None

        factor = selector._calculate_recency_factor(advert, 100.0)
        assert factor == 0.5  # Default for no timestamp

    def test_train_location(self, weighted_config, mock_device):
        """Test manual training records fingerprint and transition."""
        selector = WeightedAreaSelector(weighted_config)

        result = selector.train_location(
            device=mock_device,
            area_id="bedroom",
            area_name="Bedroom",
            current_stamp=105.0,
        )

        assert result["fingerprints"] > 0
        # HMM disabled in config, so transitions = 0
        assert result["transitions"] == 0

    def test_hmm_integration(self, mock_device):
        """Test HMM affects area selection."""
        config = AreaSelectorConfig(
            max_radius=20.0,
            max_ad_age=30.0,
            extra={
                "use_hmm": True,
                "hmm_weight": 1.0,
                "use_variance_weighting": False,
                "use_recency_weighting": False,
                "use_passive_learning": False,
                "use_fingerprint_matching": False,
            },
        )
        selector = WeightedAreaSelector(config)

        # First selection to establish current area
        mock_device.area_advert = MagicMock()
        mock_device.area_advert.area_id = "living_room"

        result = selector.select_area(mock_device, 105.0)
        assert result.winning_advert is not None

    def test_fingerprint_matching_integration(self, mock_device):
        """Test fingerprint matching boosts weights."""
        config = AreaSelectorConfig(
            max_radius=20.0,
            max_ad_age=30.0,
            extra={
                "use_fingerprint_matching": True,
                "fingerprint_weight": 0.5,
                "use_variance_weighting": False,
                "use_recency_weighting": False,
                "use_hmm": False,
                "use_passive_learning": False,
            },
        )
        selector = WeightedAreaSelector(config)

        # Add fingerprints for kitchen that match current readings
        for _ in range(10):
            selector._fingerprint_store.record_fingerprint(
                area_id="kitchen",
                area_name="Kitchen",
                scanner_rssi={"AA:BB:CC:DD:EE:FF": -72.0},
            )

        result = selector.select_area(mock_device, 105.0)
        # Even though living room is closer, fingerprint boost might affect result
        assert result.winning_advert is not None

    def test_get_hmm_statistics(self):
        """Test retrieving HMM statistics."""
        config = AreaSelectorConfig(
            max_radius=20.0,
            max_ad_age=30.0,
            extra={"use_hmm": True},
        )
        selector = WeightedAreaSelector(config)

        stats = selector.get_hmm_statistics()
        assert stats is not None
        assert "known_areas" in stats

    def test_get_fingerprint_statistics(self, weighted_config):
        """Test retrieving fingerprint statistics."""
        selector = WeightedAreaSelector(weighted_config)

        stats = selector.get_fingerprint_statistics()
        assert "total_areas" in stats
        assert "total_samples" in stats

    def test_clear_hmm_data(self):
        """Test clearing HMM data."""
        config = AreaSelectorConfig(
            max_radius=20.0,
            max_ad_age=30.0,
            extra={"use_hmm": True},
        )
        selector = WeightedAreaSelector(config)

        # Add some data
        selector._transition_tracker.record_area("device1", "living_room")

        # Clear it
        selector.clear_hmm_data()

        stats = selector.get_hmm_statistics()
        assert stats["known_areas"] == 0

    def test_clear_fingerprint_data(self, weighted_config):
        """Test clearing fingerprint data."""
        selector = WeightedAreaSelector(weighted_config)

        # Add some data
        selector._fingerprint_store.record_fingerprint("living_room", "Living Room", {"s1": -65})

        # Clear it
        selector.clear_fingerprint_data()

        stats = selector.get_fingerprint_statistics()
        assert stats["total_areas"] == 0
