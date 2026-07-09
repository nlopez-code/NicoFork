import numpy as np


class BinnedBackgroundRates:
    """Per-event background densities derived from a binned background estimate.

    Wraps both the per-event probability density b(Ω_i) = b_i / ΔΩ_i (used in
    the response matrix) and the total expected background rate sum_all_bins b_i
    (used as the RL background-norm denominator).
    """

    def __init__(self, per_event_density: np.ndarray, total_rate: float):
        self._per_event_density = np.asarray(per_event_density, dtype=float)
        self._total_rate = float(total_rate)

    @property
    def per_event_density(self) -> np.ndarray:
        return self._per_event_density

    @property
    def total_rate(self) -> float:
        return self._total_rate

    def __len__(self) -> int:
        return len(self._per_event_density)

    def __array__(self, dtype=None):
        return np.asarray(self._per_event_density, dtype=dtype)
