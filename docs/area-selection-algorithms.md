# Area Selection Algorithms

This document describes the area selection algorithms available in Bermuda and potential future improvements for indoor positioning without requiring explicit scanner coordinates.

## Current Implementations

### 1. Minimum Distance (`min_distance`)

The original Bermuda algorithm that selects the area of the closest scanner.

**How it works:**
- For each device, find the scanner with the smallest estimated distance
- Use hysteresis to prevent bouncing between areas:
  - Require 30% distance difference to switch areas outright
  - Require 15% difference with historical confirmation
  - Prefer staying in the same area when distances are close

**Pros:**
- Simple and intuitive
- Low computational overhead
- Works well when device is clearly closest to one scanner

**Cons:**
- Can bounce between areas when device is equidistant from multiple scanners
- Ignores information from non-closest scanners
- Single noisy reading can cause incorrect area switch

**Configuration:** Default algorithm, no additional configuration needed.

---

### 2. Weighted Area Voting (`weighted_area`)

Uses inverse-distance weighting from ALL visible scanners to vote for areas.

**How it works:**
1. Collect distance readings from all valid scanners
2. Calculate weight for each scanner: `weight = 1 / distance²`
3. Sum weights by area
4. Area with highest total weight wins

**Example:**
```
Scanner A (Living Room): 2m away → weight = 1/4 = 0.25
Scanner B (Kitchen): 3m away → weight = 1/9 = 0.11
Scanner C (Bedroom): 5m away → weight = 1/25 = 0.04

Living Room total: 0.25
Kitchen total: 0.11
Bedroom total: 0.04

Winner: Living Room
```

**Pros:**
- Considers information from multiple scanners
- More stable near area boundaries
- Naturally handles cases where multiple scanners are in the same area

**Cons:**
- May be slower to respond to actual room changes
- ~~All scanners weighted equally regardless of signal quality~~ (Fixed with variance weighting)

**Configuration:**
- `weight_power`: Exponent for distance weighting (default: 2.0)
- `min_scanners`: Minimum scanners for weighted voting (default: 2)
- `fallback_to_closest`: Fall back to closest if < min_scanners (default: true)
- `use_variance_weighting`: Enable signal variance weighting (default: true)
- `variance_scale`: Scale factor for variance normalization (default: 10.0)
- `min_variance_samples`: Minimum RSSI samples needed for variance calculation (default: 3)

---

## Planned Improvements

### Phase 1: Signal Quality Weighting

#### Signal Variance Weighting

**Status:** Implemented in `weighted_area` selector

Penalize readings from scanners with unstable/noisy signals.

**How it works:**
- Calculate RSSI standard deviation over recent readings (last 10 samples)
- High variance = noisy signal (multipath, interference) = lower confidence
- Low variance = stable signal = higher confidence
- Multiply weight by confidence factor

```python
variance = stdev(recent_rssi_readings)
confidence = 1 / (1 + variance / scale)
weight = (1 / distance²) * confidence
```

**Confidence values with default scale=10:**
| RSSI Std Dev | Confidence |
|--------------|------------|
| 0 (stable)   | 1.00       |
| 5            | 0.67       |
| 10           | 0.50       |
| 20           | 0.33       |
| 50 (noisy)   | 0.17       |

**Configuration:**
- `use_variance_weighting`: Enable/disable (default: true)
- `variance_scale`: Scale factor for normalization (default: 10.0)
- `min_variance_samples`: Minimum samples needed (default: 3)

**Expected improvement:** 10-15% reduction in incorrect area switches

#### Recency Weighting

**Status:** Implemented in `weighted_area` selector

Weight fresher readings higher than stale ones.

**How it works:**
- Calculate age of each scanner reading (time since last advertisement)
- Apply exponential decay to weight based on age
- Very recent readings get full weight
- Older readings get progressively less weight

```python
age = current_time - reading_timestamp
recency_factor = exp(-age / decay_constant)
weight = base_weight * recency_factor
```

**Recency values with default decay_constant=5 seconds:**
| Reading Age | Recency Factor |
|-------------|----------------|
| 0s (fresh)  | 1.00           |
| 2.5s        | 0.61           |
| 5s          | 0.37           |
| 10s         | 0.14           |
| 15s (stale) | 0.05           |

**Configuration:**
- `use_recency_weighting`: Enable/disable (default: true)
- `recency_decay_secs`: Time constant for decay in seconds (default: 5.0)

**Expected improvement:** 5-10% faster response to actual movement

---

### Phase 2: Learning-Based Approaches

#### Hidden Markov Model (HMM)

**Status:** Implemented in `weighted_area` selector

Learn which area transitions are likely based on floor plan topology.

**How it works:**
1. Track area transitions over time (e.g., "Living Room → Kitchen")
2. Build transition probability matrix with Laplace smoothing
3. When making area decision, multiply signal-based weight by transition probability
4. Persist learned transitions to `.storage/bermuda.transition_tracker`

**Key insight:** This implicitly learns floor plan topology without coordinates:
- Adjacent rooms will have high transition probabilities
- Non-adjacent rooms will have low transition probabilities
- System learns that "Kitchen → Garage" is unlikely without explicit configuration

**Configuration:**
- `use_hmm`: Enable/disable HMM (default: true)
- `hmm_weight`: How strongly HMM affects final weight, 0-1 (default: 1.0)

**TransitionTracker parameters:**
- `smoothing`: Laplace smoothing factor (default: 1.0)
- `min_transitions`: Minimum transitions before using learned probabilities (default: 5)
- `decay_factor`: Gradual decay to adapt to changes (default: 0.995)
- `self_transition_weight`: Initial bias for staying in place (default: 10)

**Persistence:**
- Transitions are saved to `~/.storage/bermuda.transition_tracker`
- Saves periodically (every 100 area selections) and on shutdown
- Survives Home Assistant restarts

**Example transition matrix after learning:**
```
From\To    | Kitchen | Living | Bedroom | Garage
-----------+---------+--------+---------+--------
Kitchen    |  0.70   |  0.25  |  0.04   |  0.01
Living     |  0.20   |  0.65  |  0.14   |  0.01
Bedroom    |  0.05   |  0.15  |  0.79   |  0.01
Garage     |  0.10   |  0.05  |  0.02   |  0.83
```

**Expected improvement:** 20-30% reduction in false area switches

#### RSSI Fingerprint Matching

**Status:** Implemented in `weighted_area` selector

Store and match RSSI fingerprints to improve area detection accuracy.

**How it works:**
1. Record RSSI readings from all visible scanners when in a known area (via manual training or passive learning)
2. Build per-area statistics: mean RSSI and variance per scanner
3. During area selection, match current RSSI readings against stored fingerprints
4. Boost weights for areas with high fingerprint similarity
5. Persist fingerprints to `.storage/bermuda.fingerprints`

**Matching algorithm:**
- Calculate Euclidean distance between current RSSI and stored mean RSSI
- Convert distance to similarity score using exponential decay: `score = exp(-distance / 10)`
- Boost area weight: `new_weight = weight * (1 + fingerprint_weight * similarity)`

**Similarity scores:**
| RSSI Distance | Similarity |
|---------------|------------|
| 0 dBm (exact) | 1.00       |
| 5 dBm         | 0.61       |
| 10 dBm        | 0.37       |
| 20 dBm        | 0.14       |

**Configuration:**
- `use_fingerprint_matching`: Enable/disable (default: true)
- `fingerprint_weight`: How strongly fingerprints boost weights (default: 0.5)

**FingerprintStore features:**
- Maximum 100 samples per area (configurable)
- Minimum 5 samples required for matching
- Calculates mean RSSI and variance per scanner

**Persistence:**
- Fingerprints saved to `~/.storage/bermuda.fingerprints`
- Survives Home Assistant restarts

---

### Manual Training

**Status:** Implemented

Allow users to manually train the area selector by indicating their current location.

**Service: `bermuda.train_location`**

Call this service to record training data for a device in a specific area.

**Parameters:**
- `device_address` (required): MAC address of the device to train
- `area_id` (required): Home Assistant area ID

**Example service call:**
```yaml
service: bermuda.train_location
data:
  device_address: "AA:BB:CC:DD:EE:FF"
  area_id: "living_room"
```

**What gets trained:**
1. **RSSI Fingerprint**: Records current RSSI readings from all visible scanners
2. **HMM Transition**: Records this area for transition probability learning

**Select Entity: "Train Location"**

Each tracked device has a "Train Location" select entity (disabled by default).

**To use:**
1. Enable the entity in Home Assistant
2. Go to the device's location in your home
3. Select the area from the dropdown
4. The model is trained with current RSSI readings

**Use cases:**
- Quickly train the model when area detection is wrong
- Bootstrap learning in a new installation
- Correct persistent misdetection in specific locations

---

#### Passive Fingerprint Learning

**Status:** Implemented in `weighted_area` selector

Automatically learn RSSI "signatures" for each area over time.

**How it works:**
1. When device is confidently in an area (weight ratio above threshold), record RSSI pattern
2. Rate-limit recordings to avoid flooding storage (default: once per 30 seconds per device)
3. Build per-area fingerprint statistics over time
4. Fingerprints persist to `.storage/bermuda.fingerprints`

**Configuration:**
- `use_passive_learning`: Enable/disable (default: true)
- `passive_learning_threshold`: Confidence threshold for recording (default: 0.65)
- `passive_learning_interval`: Minimum seconds between recordings per device (default: 30)

**Confidence threshold:**
The confidence value represents the winning area's weight as a fraction of total weight.
- 0.65 = winning area has 65% of total weight (moderate confidence)
- Higher thresholds = only record when very confident
- Lower thresholds = record more often but potentially less accurate data

**Pros:**
- No manual calibration required
- Adapts to environmental changes over time
- Builds database automatically in the background

**Cons:**
- Requires time to build reliable fingerprints
- May record inaccurate data if threshold is too low

**Note:** Active fingerprint matching in area selection is planned for future release. Currently fingerprints are collected but not yet used for area decisions.

#### Gaussian Mixture Models (GMM)

**Status:** Research/Planned

Model RSSI distributions per area as mixture of Gaussians.

**How it works:**
- RSSI readings in a room aren't single values - they form distributions
- GMM models these distributions more accurately than single mean/variance
- Handles multimodal distributions (e.g., different signal paths in same room)
- Assign new readings to area whose GMM has highest likelihood

**Expected improvement:** 15-25% accuracy improvement

**Implementation complexity:** ~100 lines of code (with scipy)

---

### Phase 3: Advanced Approaches

#### Bayesian Particle Filters

**Status:** Research Only

Full probabilistic model of device location.

**How it works:**
1. Maintain a set of "particles" representing possible device locations
2. Each particle has a weight based on how well it explains observed signals
3. Resample particles based on weights
4. Estimate location as weighted average of particles

**Pros:**
- Handles uncertainty explicitly
- Can model complex movement patterns
- Theoretically optimal given the model

**Cons:**
- Higher computational overhead
- Requires tuning particle count and resampling strategy
- More complex to implement and debug

**Expected improvement:** 15-30% accuracy improvement

**Implementation complexity:** ~300 lines of code

#### Graph Neural Networks (GNN)

**Status:** Research Only

Learn scanner relationships automatically using neural networks.

**How it works:**
1. Model scanners as nodes in a graph
2. Edges connect scanners that can "see" each other
3. GNN learns to propagate information through the network
4. Predicts area based on learned scanner relationships

**Pros:**
- Can learn complex spatial relationships
- No explicit coordinates needed
- State-of-the-art accuracy potential

**Cons:**
- Requires ML framework (PyTorch/TensorFlow)
- Needs training data
- Higher computational requirements
- Overkill for most home automation use cases

**Expected improvement:** 25-40% accuracy improvement

**Implementation complexity:** 300+ lines of code, ML dependencies

---

## Algorithm Selection Guide

| Use Case | Recommended Algorithm |
|----------|----------------------|
| Simple setup, few scanners | `min_distance` |
| Multiple scanners, boundary issues | `weighted_area` |
| Noisy RF environment | `weighted_area` (variance weighting enabled by default) |
| Complex floor plan | `weighted_area` (HMM enabled by default) |
| Frequent false switches | `weighted_area` with all features enabled |

## Feature Summary

The `weighted_area` selector includes several advanced features, all enabled by default:

| Feature | Purpose | Default |
|---------|---------|---------|
| Variance Weighting | Penalize noisy/unstable signals | Enabled |
| Recency Weighting | Prefer fresher readings | Enabled |
| HMM Transitions | Learn room connectivity | Enabled |
| Fingerprint Matching | Boost areas matching RSSI patterns | Enabled |
| Passive Learning | Auto-collect fingerprints | Enabled |
| Manual Training | User-triggered training | Via service/entity |

---

## References

### Academic Papers

- [MDPI: Wi-Fi and BLE Indoor Positioning Systems Review (2024)](https://www.mdpi.com/1424-8220/25/22/6946)
- [IEEE: Bayesian RSSI Indoor Localization](https://ieeexplore.ieee.org/document/1392744/)
- [Nature: Bayesian Optimized Positioning (2024)](https://www.nature.com/articles/s41598-024-79647-8)
- [Princeton: Zee Zero-Effort Crowdsourcing](https://www.cs.princeton.edu/courses/archive/spring17/cos598A/papers/zee.pdf)

### Implementation Resources

- [Kalman Filters for RSSI Noise Reduction](https://www.wouterbulten.nl/posts/kalman-filters-explained-removing-noise-from-rssi-signals/)
- [Hidden Markov Models for WiFi Fingerprinting](https://www.researchgate.net/publication/282206835_A_hidden_Markov_model_for_indoor_user_tracking_based_on_WiFi_fingerprinting_and_step_detection)
- [GMM Indoor Localization via BLE](https://www.researchgate.net/publication/338605568_Gaussian_Mixture-based_Indoor_Localization_via_Bluetooth_Low_Energy_Sensors)

---

## Contributing

To add a new area selection algorithm:

1. Create a new file in `custom_components/bermuda/area_selectors/`
2. Inherit from `AreaSelectorBase`
3. Implement `select_area(device, current_stamp) -> AreaSelectionResult`
4. Register in `area_selectors/__init__.py`
5. Add translations in `translations/en.json`

See `min_distance.py` and `weighted_area.py` for examples.
