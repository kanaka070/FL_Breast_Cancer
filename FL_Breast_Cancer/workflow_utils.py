import numpy as np


def partition_hospital_indices(y_train, seed=42):
    """Create three reproducible, disjoint non-IID partitions."""
    labels = np.asarray(y_train)
    indices = np.asarray(
        getattr(y_train, "index", np.arange(len(labels)))
    )
    if len(indices) != len(labels) or not np.isin(labels, [0, 1]).all():
        raise ValueError("Training labels must be aligned binary values.")
    if len(labels) < 3 or not np.any(labels == 0) or not np.any(labels == 1):
        raise ValueError("Training data must contain both classes and three or more records.")

    benign_indices = indices[labels == 0].copy()
    malignant_indices = indices[labels == 1].copy()
    rng = np.random.default_rng(seed)
    rng.shuffle(benign_indices)
    rng.shuffle(malignant_indices)

    benign_groups = np.array_split(benign_indices, 3)
    malignant_sizes = np.floor(
        len(malignant_indices) * np.array([2, 5, 10]) / 17
    ).astype(int)
    for remainder_index in range(len(malignant_indices) - malignant_sizes.sum()):
        malignant_sizes[remainder_index] += 1
    first_cut = malignant_sizes[0]
    second_cut = first_cut + malignant_sizes[1]
    malignant_groups = [
        malignant_indices[:first_cut],
        malignant_indices[first_cut:second_cut],
        malignant_indices[second_cut:],
    ]
    return [
        np.concatenate([benign_groups[i], malignant_groups[i]])
        for i in range(3)
    ]


def federated_averaging(client_results):
    if not client_results:
        raise ValueError("No client updates available for aggregation.")

    counts = []
    parameter_sets = []
    for result in client_results:
        count = result.get("num_patients")
        parameters = result.get("parameters", {})
        if not isinstance(count, (int, np.integer)) or count < 0:
            raise ValueError("Client patient counts must be non-negative integers.")
        if "weights" not in parameters or "bias" not in parameters:
            raise ValueError("Every client update must include weights and bias.")
        weights = np.asarray(parameters["weights"], dtype=float)
        bias = float(parameters["bias"])
        if weights.ndim != 1 or not np.isfinite(weights).all() or not np.isfinite(bias):
            raise ValueError("Client parameters must be finite one-dimensional weights and a finite bias.")
        counts.append(count)
        parameter_sets.append((weights, bias))

    total_patients = sum(counts)
    if total_patients == 0:
        raise ValueError("No patients available for aggregation.")

    shape = parameter_sets[0][0].shape
    if any(weights.shape != shape for weights, _ in parameter_sets):
        raise ValueError("Client weight vectors must have matching shapes.")

    global_weights = np.zeros(shape, dtype=float)
    global_bias = 0.0
    for count, (weights, bias) in zip(counts, parameter_sets):
        proportion = count / total_patients
        global_weights += proportion * weights
        global_bias += proportion * bias
    return {"weights": global_weights, "bias": float(global_bias)}


def trust_weighted_averaging(client_results):
    """Aggregate client parameters with weights proportional to n_i * trust_i."""
    if not client_results:
        raise ValueError("No client updates available for aggregation.")

    weighted_results = []
    for result in client_results:
        count = result.get("num_patients")
        if not isinstance(count, (int, np.integer)) or count < 0:
            raise ValueError("Client patient counts must be non-negative integers.")
        trust = result.get("trust_score")
        if not isinstance(trust, (int, float, np.integer, np.floating)):
            raise ValueError("Every client update must include a numeric trust score.")
        if not np.isfinite(trust) or not 0 <= trust <= 1:
            raise ValueError("Client trust scores must be finite and between 0 and 1.")
        weighted_results.append({
            "num_patients": count,
            "parameters": result.get("parameters", {}),
            "trust_score": float(trust),
        })

    trust_mass = sum(item["num_patients"] * item["trust_score"] for item in weighted_results)
    if trust_mass <= 0:
        raise ValueError("Trust-weighted aggregation requires positive total trust mass.")

    counts = []
    parameter_sets = []
    for item in weighted_results:
        count = item["num_patients"]
        if not isinstance(count, (int, np.integer)) or count < 0:
            raise ValueError("Client patient counts must be non-negative integers.")
        parameters = item["parameters"]
        if "weights" not in parameters or "bias" not in parameters:
            raise ValueError("Every client update must include weights and bias.")
        weights = np.asarray(parameters["weights"], dtype=float)
        bias = float(parameters["bias"])
        if weights.ndim != 1 or not np.isfinite(weights).all() or not np.isfinite(bias):
            raise ValueError("Client parameters must be finite one-dimensional weights and a finite bias.")
        counts.append(count * item["trust_score"] / trust_mass)
        parameter_sets.append((weights, bias))

    shape = parameter_sets[0][0].shape
    if any(weights.shape != shape for weights, _ in parameter_sets):
        raise ValueError("Client weight vectors must have matching shapes.")
    global_weights = np.zeros(shape, dtype=float)
    global_bias = 0.0
    for proportion, (weights, bias) in zip(counts, parameter_sets):
        global_weights += proportion * weights
        global_bias += proportion * bias
    return {"weights": global_weights, "bias": float(global_bias)}
