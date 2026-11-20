import altair as alt
import numpy as np
import pandas as pd


PERCENTAGE_METRICS = {
    "accuracy": "Accuracy (%)",
    "precision": "Precision (%)",
    "malignant_recall": "Malignant Recall (%)",
    "specificity": "Specificity (%)",
    "f1_score": "F1 Score (%)",
    "roc_auc": "ROC AUC (%)",
}
COUNT_METRICS = {
    "false_negatives": "False Negatives (patients)",
    "false_positives": "False Positives (patients)",
    "true_negatives": "True Negatives (patients)",
    "true_positives": "True Positives (patients)",
}

CHART_BACKGROUND = "#f4f7fb"
CHART_TEXT = "#243b53"
CHART_MUTED = "#52677c"
CHART_GRID = "#d9e2ec"
CHART_TEAL = "#087e8b"
CHART_NAVY = "#102a43"


def _apply_research_chart_theme(chart):
    return (
        chart.configure(
            background=CHART_BACKGROUND,
            font="Segoe UI",
            title=alt.TitleConfig(color=CHART_NAVY, fontSize=14, fontWeight=600),
            axis=alt.AxisConfig(
                labelColor=CHART_MUTED,
                titleColor=CHART_TEXT,
                gridColor=CHART_GRID,
                domainColor=CHART_GRID,
                tickColor=CHART_GRID,
            ),
            legend=alt.LegendConfig(
                labelColor=CHART_TEXT,
                titleColor=CHART_TEXT,
            ),
            view=alt.ViewConfig(stroke="transparent"),
        )
        .configure_range(
            category=[CHART_TEAL, CHART_NAVY, "#6b879c", "#74a9ad"],
        )
    )


def format_comparison_for_display(data):
    """Format a copy of saved comparison values with explicit units."""
    display = data.copy()
    for source, label in PERCENTAGE_METRICS.items():
        if source not in display:
            continue
        values = pd.to_numeric(display[source], errors="raise")
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"{source} contains non-finite values.")
        if ((values < 0) | (values > 1)).any():
            raise ValueError(f"{source} must be stored as fractions between 0 and 1.")
        display[source] = (values * 100).round(2)
        display = display.rename(columns={source: label})
    for source, label in COUNT_METRICS.items():
        if source not in display:
            continue
        values = pd.to_numeric(display[source], errors="raise")
        if not np.isfinite(values.to_numpy(dtype=float)).all() or (values < 0).any():
            raise ValueError(f"{source} counts must be finite and non-negative.")
        display = display.rename(columns={source: label})
    return display


def horizontal_metric_chart(data, *, category, metric, title, percentage=False):
    """Build a horizontal bar chart with explicit units and axis bounds."""
    values = pd.to_numeric(data[metric], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{metric} contains non-finite values.")

    chart_data = pd.DataFrame({
        "category": data[category].astype(str).to_numpy(),
        "value": values,
    })
    if percentage:
        if np.any((values < 0) | (values > 1)):
            raise ValueError(f"{metric} must be stored as fractions between 0 and 1.")
        chart_data["value"] = chart_data["value"] * 100
        x_axis = alt.Axis(title=f"{title} (%)", values=[0, 20, 40, 60, 80, 100])
        x_scale = alt.Scale(domain=[0, 100], clamp=True, nice=False)
        tooltip_format = ".2f"
    else:
        if np.any(values < 0):
            raise ValueError(f"{metric} counts cannot be negative.")
        maximum = max(float(values.max(initial=0)), 1.0)
        x_axis = alt.Axis(title=title, format="d")
        x_scale = alt.Scale(domain=[0, maximum], clamp=True, nice=False)
        tooltip_format = ".0f"

    chart = (
        alt.Chart(chart_data)
        .mark_bar(color=CHART_TEAL)
        .encode(
            x=alt.X("value:Q", title=x_axis.title, axis=x_axis, scale=x_scale),
            y=alt.Y(
                "category:N",
                title=None,
                sort=alt.EncodingSortField(field="value", order="descending"),
                axis=alt.Axis(labelLimit=320),
            ),
            tooltip=[
                alt.Tooltip("category:N", title=category),
                alt.Tooltip("value:Q", title=title, format=tooltip_format),
            ],
        )
        .properties(title=title, height=max(150, 42 * len(chart_data)))
    )
    return _apply_research_chart_theme(chart)


def horizontal_grouped_count_chart(data, *, category, metrics, title):
    values = data.loc[:, [category, *metrics]].copy()
    for metric in metrics:
        values[metric] = pd.to_numeric(values[metric], errors="raise")
        if not np.isfinite(values[metric].to_numpy(dtype=float)).all():
            raise ValueError(f"{metric} contains non-finite values.")
        if (values[metric] < 0).any():
            raise ValueError(f"{metric} counts cannot be negative.")
    long_data = values.melt(
        id_vars=[category],
        value_vars=metrics,
        var_name="group",
        value_name="value",
    ).rename(columns={category: "category"})
    maximum = max(float(long_data["value"].max()), 1.0)
    chart = (
        alt.Chart(long_data)
        .mark_bar()
        .encode(
            x=alt.X(
                "value:Q",
                title="Patients",
                axis=alt.Axis(title="Patients", format="d"),
                scale=alt.Scale(domain=[0, maximum], clamp=True, nice=False),
            ),
            y=alt.Y(
                "category:N",
                title=None,
                sort=alt.EncodingSortField(field="value", op="sum", order="descending"),
                axis=alt.Axis(labelLimit=320),
            ),
            color=alt.Color("group:N", title="Diagnosis"),
            yOffset="group:N",
            tooltip=[
                alt.Tooltip("category:N", title=category),
                alt.Tooltip("group:N", title="Diagnosis"),
                alt.Tooltip("value:Q", title="Patients", format=".0f"),
            ],
        )
        .properties(title=title, height=max(150, 42 * len(data)))
    )
    return _apply_research_chart_theme(chart)


def fedavg_history_chart_data(history):
    """Return chart-safe loss observations without filling or inventing values."""
    loss_columns = [
        column for column in history.columns if "loss" in column.lower()
    ]
    if not loss_columns:
        return pd.DataFrame()

    chart_data = history.copy()
    if "round" in chart_data.columns:
        rounds = pd.to_numeric(chart_data["round"], errors="coerce")
        valid_rounds = np.isfinite(rounds.to_numpy(dtype=float))
        chart_data = chart_data.loc[valid_rounds].copy()
        chart_data["round"] = rounds.loc[valid_rounds].to_numpy(dtype=float)
        x_column = "round"
    else:
        x_column = "observation"
        chart_data[x_column] = np.arange(len(chart_data), dtype=float)

    for column in loss_columns:
        values = pd.to_numeric(chart_data[column], errors="coerce")
        finite = np.isfinite(values.to_numpy(dtype=float))
        chart_data[column] = values.where(finite, np.nan)

    loss_columns = [
        column for column in loss_columns if chart_data[column].notna().any()
    ]
    if not loss_columns:
        return pd.DataFrame()

    return chart_data.loc[:, [x_column, *loss_columns]].melt(
        id_vars=x_column,
        value_vars=loss_columns,
        var_name="metric",
        value_name="value",
    )


def fedavg_training_history_chart(history):
    """Build a line chart that breaks at missing or invalid observations."""
    chart_data = fedavg_history_chart_data(history)
    if chart_data.empty:
        return None

    x_column = "round" if "round" in chart_data.columns else "observation"
    chart = (
        alt.Chart(chart_data)
        .mark_line(point=True, invalid="break-paths-filter-domains")
        .encode(
            x=alt.X(f"{x_column}:Q", title="Round" if x_column == "round" else "Observation"),
            y=alt.Y("value:Q", title="Training Loss"),
            color=alt.Color("metric:N", title="Metric"),
            tooltip=[
                alt.Tooltip(f"{x_column}:Q", title=x_column.title(), format=".0f"),
                alt.Tooltip("metric:N", title="Metric"),
                alt.Tooltip("value:Q", title="Training Loss", format=".4f"),
            ],
        )
        .properties(title="FedAvg Training History")
    )
    return _apply_research_chart_theme(chart)
