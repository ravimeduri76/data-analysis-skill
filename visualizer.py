"""Chart generator — saves plots to disk, never calls plt.show() in server contexts."""

from pathlib import Path
import pandas as pd


_CHART_TYPES = ("bar", "hist", "scatter", "heatmap", "line", "box")

_INSTALL_HINT = (
    "Install charting dependencies with:\n"
    "    pip install matplotlib seaborn"
)


def _require_matplotlib():
    """Import and return (matplotlib, pyplot), raising a helpful error if missing."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend — safe for scripts
        import matplotlib.pyplot as plt
        return matplotlib, plt
    except ImportError:
        raise ImportError(f"'matplotlib' is required for charts.\n{_INSTALL_HINT}")


def _require_seaborn():
    """Import and return seaborn, raising a helpful error if missing."""
    try:
        import seaborn as sns
        return sns
    except ImportError:
        raise ImportError(f"'seaborn' is required for heatmap charts.\n{_INSTALL_HINT}")


def make_chart(
    df: pd.DataFrame,
    chart_type: str,
    title: str = "",
    x: str | None = None,
    y: str | None = None,
    save_path: str = r"C:\Users\Ravi\research\data-analysis\output\chart.png",
) -> str:
    """Generate and save a chart. Returns the save path.

    Requires: pip install matplotlib seaborn
    """
    if chart_type not in _CHART_TYPES:
        raise ValueError(f"chart_type must be one of {_CHART_TYPES}")

    _, plt = _require_matplotlib()
    fig, ax = plt.subplots(figsize=(10, 6))

    if chart_type == "hist":
        col = x or _first_numeric(df)
        df[col].dropna().hist(ax=ax, bins=30, color="steelblue", edgecolor="white")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")

    elif chart_type == "bar":
        col_x = x or _first_categorical(df)
        col_y = y or _first_numeric(df)
        data = df.groupby(col_x)[col_y].mean().sort_values(ascending=False).head(20)
        data.plot(kind="bar", ax=ax, color="steelblue")
        ax.set_xlabel(col_x)
        ax.set_ylabel(f"Mean {col_y}")
        plt.xticks(rotation=45, ha="right")

    elif chart_type == "scatter":
        col_x = x or _first_numeric(df)
        col_y = y or _second_numeric(df, col_x)
        ax.scatter(df[col_x], df[col_y], alpha=0.5, color="steelblue", s=15)
        ax.set_xlabel(col_x)
        ax.set_ylabel(col_y)

    elif chart_type == "heatmap":
        sns = _require_seaborn()
        numeric = df.select_dtypes(include="number")
        sns.heatmap(numeric.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)

    elif chart_type == "line":
        col_x = x or df.columns[0]
        col_y = y or _first_numeric(df)
        df.plot(x=col_x, y=col_y, ax=ax, color="steelblue")

    elif chart_type == "box":
        col = y or _first_numeric(df)
        df[[col]].boxplot(ax=ax)

    ax.set_title(title or chart_type.capitalize())
    plt.tight_layout()

    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return str(out)


# ── helpers ────────────────────────────────────────────────────────────────────

def _first_numeric(df: pd.DataFrame) -> str:
    nums = df.select_dtypes(include="number").columns
    if len(nums) == 0:
        raise ValueError("No numeric column found for chart.")
    return nums[0]


def _second_numeric(df: pd.DataFrame, exclude: str) -> str:
    nums = [c for c in df.select_dtypes(include="number").columns if c != exclude]
    if not nums:
        raise ValueError("Need at least two numeric columns for scatter chart.")
    return nums[0]


def _first_categorical(df: pd.DataFrame) -> str:
    cats = df.select_dtypes(exclude="number").columns
    if len(cats) == 0:
        raise ValueError("No categorical column found for bar chart.")
    return cats[0]
