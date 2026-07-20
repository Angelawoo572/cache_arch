"""Stateless SHA-256 keyed sampling for strict common random numbers.

Every variate is a pure function of an explicit experiment coordinate.  No
mutable RNG state is carried between callbacks, so a different outcome or
action count at an earlier callback cannot shift any later callback's random
numbers.  The same keyed uniform is transformed through each model's learned
inverse CDF, which makes capacity sweeps use genuine common random numbers.

This module intentionally supports Python 3.6: server-side ChampSim helpers
must not depend on newer annotation or dataclass syntax.
"""
import hashlib
import inspect
import json
import math
import struct

import numpy as np
import scipy
from scipy.stats import poisson


SAMPLER_REVISION = "sha256_event_keyed_inverse_cdf_crn_v1"
KEY_FIELDS = (
    "revision", "decoder_seed", "trace", "policy", "role",
    "event_key", "head", "action_rank",
)
_KEY_ENCODING = "uint32_be_length_prefixed_utf8"
_UNIFORM_DENOMINATOR = float(1 << 52)


def _field_bytes(value):
    encoded = str(value).encode("utf-8")
    if len(encoded) > 0xFFFFFFFF:
        raise ValueError("key field exceeds uint32 length domain")
    return struct.pack(">I", len(encoded)) + encoded


def _coordinate_bytes(
    decoder_seed, trace, policy, role, event_key, head, action_rank,
):
    values = (
        SAMPLER_REVISION, int(decoder_seed), trace, policy, role,
        event_key, head, int(action_rank),
    )
    return b"".join(_field_bytes(value) for value in values)


def keyed_uniform(
    decoder_seed, trace, policy, role, event_key, head, action_rank=0,
):
    """Return one deterministic open-interval uniform for a key coordinate."""
    coordinate = _coordinate_bytes(
        decoder_seed, trace, policy, role, event_key, head, action_rank,
    )
    digest = hashlib.sha256(coordinate).digest()
    # IEEE-754 float64 has 52 explicit fraction bits.  Using exactly 52 digest
    # bits plus a half-bin offset keeps the result strictly inside (0, 1),
    # including at both integer extremes, without platform-dependent rounding.
    integer = struct.unpack(">Q", digest[:8])[0] >> 12
    return (float(integer) + 0.5) / _UNIFORM_DENOMINATOR


def keyed_uniforms(
    decoder_seed, trace, policy, role, event_keys, head, action_rank=0,
):
    """Vector form of :func:`keyed_uniform` preserving the supplied order."""
    return np.asarray([
        keyed_uniform(
            decoder_seed, trace, policy, role, event_key, head, action_rank,
        )
        for event_key in event_keys
    ], dtype=np.float64)


def _probability_vector(values, name):
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or not np.all(np.isfinite(result)):
        raise ValueError("{} must be a finite vector".format(name))
    if np.any(result < 0.0) or np.any(result > 1.0):
        raise ValueError("{} is outside [0, 1]".format(name))
    return result


def bernoulli_icdf(probabilities, uniforms):
    """Transform uniforms through Bernoulli inverse CDFs."""
    probabilities = _probability_vector(probabilities, "Bernoulli probability")
    uniforms = _probability_vector(uniforms, "Bernoulli uniform")
    if len(probabilities) != len(uniforms):
        raise ValueError("Bernoulli probability/uniform lengths differ")
    if np.any(uniforms <= 0.0) or np.any(uniforms >= 1.0):
        raise ValueError("Bernoulli uniforms must lie strictly inside (0, 1)")
    return (uniforms < probabilities).astype(np.int64)


def poisson_icdf(means, uniforms):
    """Transform common uniforms through SciPy Poisson inverse CDFs."""
    means = np.asarray(means, dtype=np.float64)
    uniforms = _probability_vector(uniforms, "Poisson uniform")
    if means.ndim != 1 or not np.all(np.isfinite(means)):
        raise ValueError("Poisson means must be a finite vector")
    if np.any(means < 0.0):
        raise ValueError("Poisson means must be nonnegative")
    if len(means) != len(uniforms):
        raise ValueError("Poisson mean/uniform lengths differ")
    if np.any(uniforms <= 0.0) or np.any(uniforms >= 1.0):
        raise ValueError("Poisson uniforms must lie strictly inside (0, 1)")
    quantiles = np.asarray(poisson.ppf(uniforms, means), dtype=np.float64)
    maximum = float(np.iinfo(np.int64).max)
    if (
        not np.all(np.isfinite(quantiles))
        or np.any(quantiles < 0.0)
        or np.any(quantiles >= maximum)
        or np.any(quantiles != np.floor(quantiles))
    ):
        raise RuntimeError("Poisson inverse CDF exceeds the int64 count domain")
    return quantiles.astype(np.int64)


def event_keyed_hurdle_counts(
    trigger_probabilities, excess_means, event_keys,
    decoder_seed, trace, policy, role,
):
    """Sample zero/positive hurdle counts with two independent key namespaces."""
    trigger_probabilities = _probability_vector(
        trigger_probabilities, "trigger probability",
    )
    excess_means = np.asarray(excess_means, dtype=np.float64)
    if not hasattr(event_keys, "__len__") or not hasattr(
        event_keys, "__getitem__"
    ):
        event_keys = list(event_keys)
    if (
        excess_means.ndim != 1
        or len(trigger_probabilities) != len(excess_means)
        or len(trigger_probabilities) != len(event_keys)
    ):
        raise ValueError("hurdle probability/mean/event-key lengths differ")
    counts = np.zeros(len(event_keys), dtype=np.int64)
    maximum = int(np.iinfo(np.int64).max)
    # SHA-256 is necessarily evaluated per coordinate.  Bounded chunks avoid
    # retaining two additional full-trace uniform arrays on large captures.
    for start in range(0, len(event_keys), 65536):
        stop = min(start + 65536, len(event_keys))
        chunk_keys = event_keys[start:stop]
        trigger_uniforms = keyed_uniforms(
            decoder_seed, trace, policy, role,
            chunk_keys, "request_trigger",
        )
        excess_uniforms = keyed_uniforms(
            decoder_seed, trace, policy, role,
            chunk_keys, "request_excess",
        )
        triggers = bernoulli_icdf(
            trigger_probabilities[start:stop], trigger_uniforms,
        )
        excess = poisson_icdf(excess_means[start:stop], excess_uniforms)
        if np.any((triggers > 0) & (excess >= maximum)):
            raise RuntimeError("positive hurdle count exceeds the int64 domain")
        counts[start:stop] = triggers * (1 + excess)
    return counts


def categorical_icdf(probabilities, uniform):
    """Sample one categorical index by inverse CDF without an RNG object."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    uniform = float(uniform)
    if (
        probabilities.ndim != 1
        or not len(probabilities)
        or not np.all(np.isfinite(probabilities))
        or np.any(probabilities < 0.0)
    ):
        raise ValueError("invalid categorical distribution")
    if not math.isfinite(uniform) or uniform <= 0.0 or uniform >= 1.0:
        raise ValueError("categorical uniform must lie strictly inside (0, 1)")
    total = float(probabilities.sum())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("categorical distribution has no probability mass")
    cumulative = np.cumsum(probabilities / total, dtype=np.float64)
    cumulative[-1] = 1.0
    return min(
        int(np.searchsorted(cumulative, uniform, side="right")),
        len(probabilities) - 1,
    )


def canonical_component_order(means):
    """Return a deterministic mean-major order immune to label permutation."""
    means = np.asarray(means, dtype=np.float64)
    if means.ndim != 1 or not len(means) or not np.all(np.isfinite(means)):
        raise ValueError("mixture means must be a nonempty finite vector")
    original_indices = np.arange(len(means), dtype=np.int64)
    return np.lexsort((original_indices, means))


def sampler_source_sha256():
    """Hash the loaded implementation source recorded by every experiment."""
    return hashlib.sha256(inspect.getsource(
        __import__(__name__, fromlist=["*"])
    ).encode("utf-8")).hexdigest()


def key_schedule_sha256():
    """Hash the immutable serialization and namespace contract."""
    payload = {
        "sampler_revision": SAMPLER_REVISION,
        "key_fields": list(KEY_FIELDS),
        "key_encoding": _KEY_ENCODING,
        "digest": "sha256",
        "uniform_mapping": "sha256_top_52_bits_half_bin_open_interval",
        "count_heads": ["request_trigger", "request_excess"],
        "action_rank_origin": 0,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def key_stream_sha256(event_keys):
    """Hash an ordered event-key stream without materializing it as JSON."""
    digest = hashlib.sha256()
    for event_key in event_keys:
        digest.update(_field_bytes(event_key))
    return digest.hexdigest()


def sampling_schedule_sha256(
    decoder_seed, trace, policy, role, coordinates,
):
    """Hash ordered ``(event_key, head, action_rank)`` decoder coordinates."""
    digest = hashlib.sha256()
    for event_key, head, action_rank in coordinates:
        digest.update(_coordinate_bytes(
            decoder_seed, trace, policy, role,
            event_key, head, action_rank,
        ))
    return digest.hexdigest()


def sampler_metadata():
    """Return a stable, JSON-serializable description of this sampler."""
    return {
        "sampler_revision": SAMPLER_REVISION,
        "key_fields": list(KEY_FIELDS),
        "key_encoding": _KEY_ENCODING,
        "digest": "sha256",
        "uniform_mapping": "sha256_top_52_bits_half_bin_open_interval",
        "bernoulli_backend": "inverse_cdf_uniform_less_than_probability",
        "poisson_backend": "scipy.stats.poisson.ppf",
        "categorical_backend": "normalized_cdf_searchsorted_right",
        "scipy_version": scipy.__version__,
        "cross_event_rng_state": False,
    }


def self_test_keyed_crn():
    """Exercise reproducibility, namespace separation, and strict CRN rules."""
    seed = 1701
    trace = "self-test-trace"
    policy = "self-test-policy"
    role = "eval"
    keys = ["0", "1", "2", "3"]
    first = keyed_uniforms(
        seed, trace, policy, role, keys, "request_trigger",
    )
    second = keyed_uniforms(
        seed, trace, policy, role, keys, "request_trigger",
    )
    if not np.array_equal(first, second):
        raise RuntimeError("keyed uniforms are not reproducible")

    permutation = [2, 0, 3, 1]
    permuted = keyed_uniforms(
        seed, trace, policy, role,
        [keys[index] for index in permutation], "request_trigger",
    )
    if not np.array_equal(permuted, first[permutation]):
        raise RuntimeError("keyed uniforms depend on traversal order")
    if (
        keyed_uniform(seed, trace, policy, role, "1", "request_trigger", 0)
        == keyed_uniform(seed, trace, policy, role, "1", "request_excess", 0)
        or keyed_uniform(seed, trace, policy, role, "1", "delta_component", 0)
        == keyed_uniform(seed, trace, policy, role, "1", "delta_component", 1)
    ):
        raise RuntimeError("key namespaces or action ranks are not separated")

    probabilities = np.asarray([0.2, 0.8, 0.4, 0.9], dtype=np.float64)
    means = np.asarray([0.5, 1.0, 2.0, 4.0], dtype=np.float64)
    full = event_keyed_hurdle_counts(
        probabilities, means, keys, seed, trace, policy, role,
    )
    tail = event_keyed_hurdle_counts(
        probabilities[1:], means[1:], keys[1:],
        seed, trace, policy, role,
    )
    changed_first = probabilities.copy()
    changed_first[0] = 1.0 - changed_first[0]
    changed = event_keyed_hurdle_counts(
        changed_first, means, keys, seed, trace, policy, role,
    )
    if not np.array_equal(full[1:], tail) or not np.array_equal(
        full[1:], changed[1:]
    ):
        raise RuntimeError("an earlier outcome shifted later keyed counts")

    common_uniforms = keyed_uniforms(
        seed, trace, policy, role, keys, "poisson-monotonicity",
    )
    low = poisson_icdf(np.full(len(keys), 0.25), common_uniforms)
    high = poisson_icdf(np.full(len(keys), 8.0), common_uniforms)
    if np.any(high < low):
        raise RuntimeError("Poisson inverse CDF violated common-u monotonicity")
    categorical = [
        categorical_icdf([0.2, 0.3, 0.5], uniform)
        for uniform in (0.1, 0.25, 0.9)
    ]
    if categorical != [0, 1, 2]:
        raise RuntimeError("categorical inverse CDF self-test failed")
    return True
