
import numpy as np


class FixedPrivateScaler:
    """
    Data-independent feature transformation.

    Does not fit means, standard deviations,
    minimums, or maximums from patient records.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        values = np.asarray(X, dtype=float)

        if not np.all(np.isfinite(values)):
            raise ValueError(
                "Input contains NaN or infinite values."
            )

        return (2.0 / np.pi) * np.arctan(values)

    def fit_transform(self, X, y=None):
        return self.transform(X)


if __name__ == "__main__":

    scaler = FixedPrivateScaler()

    example = np.array([
        [0.0, 1.0, 10.0],
        [2.0, 5.0, 20.0]
    ])

    transformed = scaler.fit_transform(example)

    print("\nORIGINAL DATA")
    print(example)

    print("\nTRANSFORMED DATA")
    print(transformed)

    print("\nAll values bounded:",
          np.all(np.abs(transformed) <= 1.0))

    print("\nFIXED SCALER TEST COMPLETED!")
