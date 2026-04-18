"""
Time-Series Anomaly Detection Module
======================================
Detects anomalous sales patterns over time at store / department level.

Approach
─────────
Sales anomalies are identified by computing a rolling baseline (moving
average + standard deviation) and flagging weeks where actual sales
deviate beyond a configurable number of standard deviations.

This is a lightweight, interpretable method suited to the retail context
where a full ARIMA / AD-specific model would add complexity without
substantial benefit for the downstream Random-Forest prediction task.

Functions
──────────
``detect_sales_spikes``  – flag unusually high or low weekly sales.
``get_anomaly_report``   – aggregate spike counts per store/department.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def detect_sales_spikes(
    df: pd.DataFrame,
    sales_col: str = "Weekly_Sales",
    group_cols: list[str] | None = None,
    window: int = 4,
    n_sigma: float = 2.5,
) -> pd.DataFrame:
    """Flag weeks with abnormally high or low sales using a rolling-mean baseline.

    For each group (default: Store + Dept), a rolling mean and rolling
    standard deviation are computed over *window* weeks.  A record is marked
    as anomalous if::

        |actual − rolling_mean| > n_sigma × rolling_std

    Args:
        df:         DataFrame with at least a ``Weekly_Sales`` column and
                    ideally a ``Date`` column for temporal ordering.
        sales_col:  Name of the sales column (default ``'Weekly_Sales'``).
        group_cols: Columns to group by before computing rolling stats.
                    Defaults to ``['Store', 'Dept']`` if present.
        window:     Rolling window size in weeks (default 4).
        n_sigma:    Number of standard deviations beyond which a record is
                    flagged as anomalous (default 2.5).

    Returns:
        Input DataFrame with two additional columns:
          - ``rolling_mean`` – the rolling baseline for that group.
          - ``is_spike``     – True if the record is anomalous.
    """
    df = df.copy()

    # Choose automatic grouping columns if not specified
    if group_cols is None:
        group_cols = [c for c in ["Store", "Dept"] if c in df.columns]

    # Sort by date if available so rolling window is chronologically correct
    if "Date" in df.columns:
        df = df.sort_values(group_cols + ["Date"]).reset_index(drop=True)

    def _rolling_stats(grp: pd.DataFrame) -> pd.DataFrame:
        """Compute rolling mean and std; flag values exceeding the band."""
        roll_mean  = grp[sales_col].rolling(window=window, min_periods=1).mean()
        roll_std   = grp[sales_col].rolling(window=window, min_periods=1).std().fillna(0)
        grp["rolling_mean"] = roll_mean
        grp["is_spike"]     = (grp[sales_col] - roll_mean).abs() > (n_sigma * roll_std)
        return grp

    if group_cols:
        df = df.groupby(group_cols, group_keys=False).apply(_rolling_stats)
    else:
        df = _rolling_stats(df)

    n_spikes = df["is_spike"].sum()
    logger.info(
        "Time-series spike detection: %d anomalies found "
        "(window=%d weeks, threshold=%.1f sigma).",
        n_spikes, window, n_sigma,
    )
    return df


def get_anomaly_report(
    df_with_spikes: pd.DataFrame,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Aggregate spike counts per store/department for a quick summary.

    Args:
        df_with_spikes: Output of :func:`detect_sales_spikes` (contains
                        the ``is_spike`` boolean column).
        group_cols:     Columns to aggregate by.
                        Defaults to ``['Store', 'Dept']`` if present.

    Returns:
        DataFrame with columns: group keys, ``spike_count``, ``spike_pct``.
    """
    if "is_spike" not in df_with_spikes.columns:
        raise ValueError(
            "'is_spike' column not found.  Run detect_sales_spikes() first."
        )

    if group_cols is None:
        group_cols = [c for c in ["Store", "Dept"] if c in df_with_spikes.columns]

    report = (
        df_with_spikes
        .groupby(group_cols)["is_spike"]
        .agg(spike_count="sum", spike_pct="mean")
        .reset_index()
    )
    report["spike_pct"] = (report["spike_pct"] * 100).round(2)
    report = report.sort_values("spike_count", ascending=False)

    logger.info("Anomaly report generated for %d groups.", len(report))
    return report
