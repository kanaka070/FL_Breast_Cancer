
# ============================================================
# DIFFERENTIAL PRIVACY ACCOUNTANT
# GAUSSIAN MECHANISM - RDP ACCOUNTING
# ============================================================

import numpy as np


class DPAccountant:

    def __init__(
        self,
        noise_multiplier,
        delta=1e-5,
        sensitivity=1.0
    ):
        try:
            noise_multiplier = float(noise_multiplier)
            delta = float(delta)
            sensitivity = float(sensitivity)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("Accountant parameters must be finite numeric values.") from error

        if not np.isfinite(noise_multiplier) or noise_multiplier <= 0:
            raise ValueError(
                "Noise multiplier must be finite and positive."
            )

        if not np.isfinite(delta) or not 0 < delta < 1:
            raise ValueError(
                "Delta must be finite and between 0 and 1."
            )

        if not np.isfinite(sensitivity) or sensitivity <= 0:
            raise ValueError(
                "Sensitivity must be finite and positive."
            )

        self.noise_multiplier = noise_multiplier
        self.delta = delta
        self.sensitivity = sensitivity

        self.steps = 0

        # Renyi orders greater than 1
        self.orders = np.concatenate([
            np.arange(2, 65, dtype=float),
            np.array([80, 96, 128, 192, 256], dtype=float)
        ])
        self._rdp = np.zeros_like(self.orders)
        self._noise_multiplier_counts = {}


    def record_step(self, noise_multiplier=None):
        self.record_steps(1, noise_multiplier=noise_multiplier)


    def record_steps(self, count, noise_multiplier=None):
        if not isinstance(count, (int, np.integer)) or isinstance(count, bool) or count < 0:
            raise ValueError("Accounted mechanism count must be a non-negative integer.")
        if noise_multiplier is None:
            noise_multiplier = self.noise_multiplier
        try:
            noise_multiplier = float(noise_multiplier)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("Noise multiplier must be finite and positive.") from error
        if not np.isfinite(noise_multiplier) or noise_multiplier <= 0:
            raise ValueError("Noise multiplier must be finite and positive.")
        if count == 0:
            return
        self._rdp += count * self.orders / (2 * noise_multiplier ** 2)
        self.steps += count
        self._noise_multiplier_counts[noise_multiplier] = (
            self._noise_multiplier_counts.get(noise_multiplier, 0) + count
        )


    def get_epsilon(self):

        if self.steps == 0:
            return 0.0

        # For stddev = noise_multiplier * sensitivity, Gaussian RDP is
        # alpha / (2 * noise_multiplier**2); convert that bound to (epsilon, delta)-DP.
        epsilons = (
            self._rdp
            + (-np.log(self.delta))
            / (self.orders - 1)
        )

        return float(np.min(epsilons))


    def get_privacy_report(self):

        return {
            "noise_multiplier": (
                next(iter(self._noise_multiplier_counts))
                if len(self._noise_multiplier_counts) == 1
                else self.noise_multiplier
                if not self._noise_multiplier_counts
                else None
            ),
            "noise_multiplier_counts": dict(self._noise_multiplier_counts),
            "sensitivity": self.sensitivity,
            "delta": self.delta,
            "steps": self.steps,
            "epsilon_upper_bound": self.get_epsilon(),
            "method": "Gaussian mechanism with integer-order RDP composition",
        }


# ============================================================
# TEST THE ACCOUNTANT
# ============================================================

if __name__ == "__main__":

    accountant = DPAccountant(
        noise_multiplier=2.0,
        delta=1e-5,
        sensitivity=1.0
    )

    print("\nDIFFERENTIAL PRIVACY ACCOUNTANT")

    print("Before training:")
    print(accountant.get_privacy_report())

    for step in range(30):
        accountant.record_step()

    print("\nAfter 30 accounted Gaussian releases:")
    print(accountant.get_privacy_report())

    print("\nDP ACCOUNTANT TEST COMPLETED!")
