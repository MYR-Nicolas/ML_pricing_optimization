import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats

# =========== EDA ==========

# Boxplot
def futuristic_boxplot(
    data,
    title="Distribution of Quantity Sold (PCS)",
    xlabel="Quantity",
    figsize=(10, 4),
    show=True
):
    """
    Create a futuristic styled boxplot with median annotation.

    Parameters
    ----------
    data : array-like
        Input data (e.g., df["PCS"]).
    title : str, default="Distribution of Quantity Sold (PCS)"
        Plot title.
    xlabel : str, default="Quantity"
        X-axis label.
    figsize : tuple, default=(10, 4)
        Figure size.
    show : bool, default=True
        Whether to display the plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """

    palette = {
        "FIG_BG": "#05070F",
        "AX_BG": "#05070F",
        "GRID": "#FFFFFF",
        "TEXT": "#D6E3FF",
        "TITLE": "#E6EDFF",
        "XTICK": "#9FB3FF",
        "YTICK": "#9FB3FF",
        "ACCENT": "#1E90FF",
        "MEDIAN": "#FF2E88",
    }

    fig, ax = plt.subplots(figsize=figsize, facecolor=palette["FIG_BG"])
    ax.set_facecolor(palette["AX_BG"])

    # Boxplot
    ax.boxplot(
        data,
        vert=False,
        patch_artist=True,
        boxprops=dict(facecolor=palette["ACCENT"], alpha=0.75),
        medianprops=dict(color=palette["MEDIAN"], linewidth=2.5),
        whiskerprops=dict(color=palette["XTICK"]),
        capprops=dict(color=palette["XTICK"]),
        flierprops=dict(
            marker="o",
            markerfacecolor=palette["MEDIAN"],
            markeredgecolor=palette["MEDIAN"],
            markersize=4,
            alpha=0.6
        )
    )

    # Median
    median = np.median(data)

    ax.axvline(median, color=palette["MEDIAN"], linestyle="--", alpha=0.8)

    ax.text(
        median,
        1.05,
        f"Median = {median:.2f}",
        color=palette["TEXT"],
        ha="center",
        fontsize=11
    )

    # Style
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.tick_params(axis="x", colors=palette["XTICK"], length=0)
    ax.tick_params(axis="y", colors=palette["YTICK"], length=0)

    ax.xaxis.label.set_color(palette["TEXT"])
    ax.title.set_color(palette["TITLE"])

    ax.grid(axis="x", color=palette["GRID"], alpha=0.12, linewidth=0.7)
    ax.grid(axis="y", visible=False)

    # Labels
    ax.set_title(title, pad=15, fontsize=14)
    ax.set_xlabel(xlabel)

    plt.tight_layout()

    if show:
        plt.show()

    return fig, ax

# ========================================================== Histogramme ==============================================================

# ==================== Distribution price and quantity 
def plot_histogram(
    data,
    title,
    xlabel,
    ylabel="Frequency",
    bin_width=None,
    discrete=False,
    figsize=(9, 4),
    bar_color="#2563EB",
    mean_color="#f59e0b",
    show_mean=True,
    show=True,
):
    """
    Plot a futuristic styled histogram with rounded bars and an optional mean line.

    Parameters
    ----------
    data : array-like
        Input values.
    title : str
        Plot title.
    xlabel : str
        X-axis label.
    ylabel : str, default="Frequency"
        Y-axis label.
    bin_width : float or None, default=None
        Width of histogram bins. Ignored when `discrete=True`.
    discrete : bool, default=False
        Whether the input data should be treated as discrete values.
    figsize : tuple, default=(9, 4)
        Figure size.
    bar_color : str, default="#2563EB"
        Histogram bar color.
    mean_color : str, default="#f59e0b"
        Mean line color.
    show_mean : bool, default=True
        Whether to display the mean line.
    show : bool, default=True
        Whether to display the plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """

    palette = {
        "FIG_BG": "#05070F",
        "AX_BG": "#05070F",
        "GRID": "#FFFFFF",
        "TEXT": "#D6E3FF",
        "TITLE": "#E6EDFF",
        "XTICK": "#9FB3FF",
        "YTICK": "#9FB3FF",
    }

    data = np.asarray(data)
    data = data[~np.isnan(data)] if np.issubdtype(data.dtype, np.number) else data

    if len(data) == 0:
        raise ValueError("Input data is empty.")

    if discrete:
        max_val = int(np.max(data))
        bins = np.arange(-0.5, max_val + 1, 1)
    elif bin_width is not None:
        data_min = np.min(data)
        data_max = np.max(data)
        bins = np.arange(data_min, data_max + bin_width, bin_width)
    else:
        bins = 20

    counts, bins = np.histogram(data, bins=bins)

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(palette["FIG_BG"])
    ax.set_facecolor(palette["AX_BG"])

    width_factor = 0.85

    for left, right, h in zip(bins[:-1], bins[1:], counts):
        width = (right - left) * width_factor
        x = left + (right - left - width) / 2

        glow = FancyBboxPatch(
            (x - 0.02, 0),
            width + 0.04,
            h,
            boxstyle="round,pad=0,rounding_size=0.08",
            linewidth=0,
            facecolor=bar_color,
            alpha=0.25,
            zorder=1,
        )
        ax.add_patch(glow)

        bar = FancyBboxPatch(
            (x, 0),
            width,
            h,
            boxstyle="round,pad=0,rounding_size=0.08",
            linewidth=1.2,
            edgecolor="#05070F",
            facecolor=bar_color,
            zorder=3,
        )
        ax.add_patch(bar)

    if show_mean:
        mean_val = np.mean(data)

        ax.axvline(
            mean_val,
            color=mean_color,
            linestyle="--",
            linewidth=2.5,
            label=f"Mean: {mean_val:.2f}",
            zorder=4,
        )

        ax.axvline(
            mean_val,
            color=mean_color,
            linewidth=8,
            alpha=0.12,
            zorder=2,
        )

        legend = ax.legend(frameon=False)
        for text in legend.get_texts():
            text.set_color(palette["TEXT"])

    ax.set_title(title, color=palette["TITLE"], fontsize=13, fontweight="bold")
    ax.set_xlabel(xlabel, color=palette["TEXT"])
    ax.set_ylabel(ylabel, color=palette["TEXT"])

    ax.tick_params(axis="x", colors=palette["XTICK"], length=0)
    ax.tick_params(axis="y", colors=palette["YTICK"], length=0)

    ax.grid(axis="y", color=palette["GRID"], alpha=0.18)
    ax.grid(axis="x", visible=False)

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_xlim(bins[0], bins[-1])

    ymax = max(counts) * 1.2 if len(counts) > 0 else 1
    ax.set_ylim(0, ymax)

    plt.tight_layout()

    if show:
        plt.show()

    return fig, ax

# # ==================== Distribution categorical

def _draw_rounded_bar(ax, x, height, width, color):
    """
    Draw a rounded bar with a glow effect.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axis.
    x : float
        Bar center position.
    height : float
        Bar height.
    width : float
        Bar width.
    color : str
        Bar color.
    """
    left = x - width / 2

    glow = FancyBboxPatch(
        (left - 0.02, 0),
        width + 0.04,
        height,
        boxstyle="round,pad=0,rounding_size=0.08",
        linewidth=0,
        facecolor=color,
        alpha=0.25,
        zorder=1
    )
    ax.add_patch(glow)

    bar = FancyBboxPatch(
        (left, 0),
        width,
        height,
        boxstyle="round,pad=0,rounding_size=0.08",
        linewidth=1.2,
        edgecolor="#05070F",
        facecolor=color,
        zorder=3
    )
    ax.add_patch(bar)


def plot_category_distribution(
    category_counts,
    title="Product Distribution by Category",
    ylabel="Number of Products",
    figsize=(10, 5),
    highlight_max=True,
    show_values=True,
    show_percentages=True,
    show=True
):
    """
    Plot a futuristic bar chart showing product distribution by category.

    Parameters
    ----------
    category_counts : pandas.Series
        Series containing counts per category.
    title : str, default="Product Distribution by Category"
        Plot title.
    ylabel : str, default="Number of Products"
        Y-axis label.
    figsize : tuple, default=(10, 5)
        Figure size.
    highlight_max : bool, default=True
        Whether to highlight the category with the maximum count.
    show_values : bool, default=True
        Whether to display raw counts above bars.
    show_percentages : bool, default=True
        Whether to display percentages above bars.
    show : bool, default=True
        Whether to display the plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    palette = {
        "FIG_BG": "#05070F",
        "AX_BG": "#05070F",
        "GRID": "#FFFFFF",
        "TEXT": "#D6E3FF",
        "TITLE": "#E6EDFF",
        "XTICK": "#9FB3FF",
        "YTICK": "#9FB3FF",
    }

    INDIGO = "#2563EB"
    GREEN = "#22C55E"

    total = category_counts.sum()

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(palette["FIG_BG"])
    ax.set_facecolor(palette["AX_BG"])

    x = np.arange(len(category_counts))
    width = 0.65
    max_val = category_counts.max()

    for i, (cat, val) in enumerate(category_counts.items()):
        color = GREEN if (highlight_max and val == max_val) else INDIGO

        _draw_rounded_bar(
            ax=ax,
            x=i,
            height=val,
            width=width,
            color=color
        )

        pct = (val / total) * 100 if total > 0 else 0

        label_parts = []
        if show_values:
            label_parts.append(f"{val:,}")
        if show_percentages:
            label_parts.append(f"{pct:.1f}%")

        if label_parts:
            ax.text(
                i,
                val + max_val * 0.03,
                "\n".join(label_parts),
                ha="center",
                va="bottom",
                color=palette["TEXT"],
                fontsize=10,
                fontweight="bold",
                zorder=5
            )

    ax.set_title(
        title,
        color=palette["TITLE"],
        fontsize=13,
        fontweight="bold"
    )
    ax.set_ylabel(ylabel, color=palette["TEXT"])

    ax.set_xticks(x)
    ax.set_xticklabels(
        category_counts.index,
        rotation=45,
        ha="right",
        color=palette["XTICK"]
    )

    ax.tick_params(axis="y", colors=palette["YTICK"], length=0)

    ax.grid(axis="y", color=palette["GRID"], alpha=0.18)
    ax.grid(axis="x", visible=False)

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_xlim(-0.6, len(category_counts) - 0.4)
    ax.set_ylim(0, max_val * 1.25)

    plt.tight_layout()

    if show:
        plt.show()

    return fig, ax

# ==================== Line plot =========================================

def _draw_glow_line(ax, x, y, color, label=None):
    """
    Draw a futuristic line with glow and halo effects.

    Args:
        ax (matplotlib.axes.Axes): Target axis.
        x (array-like): X-axis values.
        y (array-like): Y-axis values.
        color (str or tuple): Line color.
        label (str, optional): Legend label.
    """

    # Glow layer (background soft line)
    ax.plot(
        x, y,
        color=color,
        linewidth=7,
        alpha=0.12,
        solid_capstyle="round",
        zorder=1
    )

    # Main line
    ax.plot(
        x, y,
        color=color,
        linewidth=2.4,
        marker="o",
        markersize=6,
        markerfacecolor="#05070F",
        markeredgecolor=color,
        markeredgewidth=1.6,
        solid_capstyle="round",
        label=label,
        zorder=3
    )

    # Point halo effect
    ax.scatter(
        x, y,
        s=140,
        color=color,
        alpha=0.10,
        zorder=2
    )


def plot_evolution_category(
    df,
    date_col="DATE",
    category_col="Category",
    value_col="quantity",
    categories=None,
    agg="sum",
    title=None,
    xlabel="Month",
    ylabel=None,
    figsize=(9, 4),
    rotation=45,
    show=True
):
    """
    Plot the monthly evolution of a numerical variable grouped by category.

    This function aggregates data at a monthly level and displays a
    futuristic styled line chart with glow effects.

    Args:
        df (pandas.DataFrame): Input dataset.
        date_col (str): Column containing date values.
        category_col (str): Column containing category labels.
        value_col (str): Numerical column to aggregate.
        categories (list, optional): Subset of categories to display.
        agg (str): Aggregation method ("sum", "mean", "median").
        title (str, optional): Plot title.
        xlabel (str): Label for x-axis.
        ylabel (str, optional): Label for y-axis.
        figsize (tuple): Figure size.
        rotation (int): Rotation angle for x-axis labels.
        show (bool): Whether to display the plot.

    Returns:
        tuple: (fig, ax) Matplotlib figure and axis.
    """

    palette = {
        "FIG_BG": "#05070F",
        "AX_BG": "#05070F",
        "GRID": "#FFFFFF",
        "TEXT": "#D6E3FF",
        "TITLE": "#E6EDFF",
        "XTICK": "#9FB3FF",
        "YTICK": "#9FB3FF",
    }

    
    # Data preparation
   
    data = df.copy()

    # Filter categories if specified
    if categories is not None:
        data = data[data[category_col].isin(categories)].copy()

    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    data = data.dropna(subset=[date_col])

    # Extract monthly period
    data["month"] = data[date_col].dt.to_period("M")

    # Aggregation

    grouped = data.groupby(["month", category_col], observed=True)[value_col]

    if agg == "mean":
        df_month = grouped.mean().unstack(category_col)
    elif agg == "median":
        df_month = grouped.median().unstack(category_col)
    else:
        df_month = grouped.sum().unstack(category_col)

    # Ensure full time continuity 
    full_range = pd.period_range(df_month.index.min(), df_month.index.max(), freq="M")
    df_month = df_month.reindex(full_range)
    df_month.index.name = "month"

    # Convert to long format for plotting
    df_plot = (
        df_month.reset_index()
        .melt(id_vars="month", var_name=category_col, value_name=value_col)
    )
    df_plot["month"] = df_plot["month"].dt.to_timestamp()

   
    # Plot construction
   
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(palette["FIG_BG"])
    ax.set_facecolor(palette["AX_BG"])

    unique_categories = df_plot[category_col].dropna().unique()
    n_cat = len(unique_categories)

    # Generate color gradien
    cmap = plt.cm.Blues
    color_positions = np.linspace(0.45, 0.95, n_cat)
    line_colors = [cmap(pos) for pos in color_positions]

    for i, cat in enumerate(unique_categories):
        sub = (
            df_plot[df_plot[category_col] == cat]
            .dropna(subset=[value_col])
            .sort_values("month")
        )

        if sub.empty:
            continue

        _draw_glow_line(
            ax=ax,
            x=sub["month"].values,
            y=sub[value_col].values,
            color=line_colors[i],
            label=str(cat)
        )

  
    # Labels and formatting
    
    if title is None:
        title = f"Monthly {agg} of {value_col} by {category_col}"

    if ylabel is None:
        ylabel = value_col

    ax.set_title(title, fontsize=13, fontweight="bold", color=palette["TITLE"], pad=12)
    ax.set_xlabel(xlabel, color=palette["TEXT"])
    ax.set_ylabel(ylabel, color=palette["TEXT"])

    # Time axis formatting
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.get_xticklabels(), rotation=rotation, ha="right")

    # Grid
    ax.grid(axis="y", color=palette["GRID"], alpha=0.18, linewidth=0.7)
    ax.grid(axis="x", color=palette["GRID"], alpha=0.10, linewidth=0.6)

    # Clean axes
    ax.tick_params(axis="x", colors=palette["XTICK"], length=0)
    ax.tick_params(axis="y", colors=palette["YTICK"], length=0)

    for spine in ax.spines.values():
        spine.set_visible(False)

    # Legend
    legend = ax.legend(
        title=category_col,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=False
    )

    if legend is not None:
        legend.get_title().set_color(palette["TEXT"])
        for text in legend.get_texts():
            text.set_color(palette["TEXT"])

    plt.tight_layout()

    if show:
        plt.show()

    return fig, ax

# ==================== SCATTER PLOT ====================

def plot_scatter_price_quantity(
    df,
    price_col="price",
    quantity_col="quantity",
    category_col="Category",
    title="Price vs Quantity Sold",
    xlabel="Unit Price",
    ylabel="Quantity",
    figsize=(7, 5),
    show=True
):
    """
    Plot a futuristic scatter plot showing the relationship between price and quantity.

    The plot includes:
    - A glow effect (background halo)
    - Colored points by category
    - A dark futuristic theme

    Args:
        df (pandas.DataFrame): Input dataset.
        price_col (str): Column representing price values.
        quantity_col (str): Column representing quantity values.
        category_col (str): Column representing categories (used for color encoding).
        title (str): Plot title.
        xlabel (str): Label for x-axis.
        ylabel (str): Label for y-axis.
        figsize (tuple): Figure size.
        show (bool): Whether to display the plot.

    Returns:
        tuple: (fig, ax) Matplotlib figure and axis.
    """

    palette = {
        "FIG_BG": "#05070F",
        "AX_BG": "#05070F",
        "GRID": "#FFFFFF",
        "TEXT": "#D6E3FF",
        "TITLE": "#E6EDFF",
        "XTICK": "#9FB3FF",
        "YTICK": "#9FB3FF",
    }

    FUTURISTIC_BLUES = [
        "#38BDF8",
        "#2563EB",
        "#1D4ED8",
        "#0A84FF",
        "#60A5FA",
        "#1E40AF",
        "#3B82F6",
        "#93C5FD",
    ]

    # Plot setup
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(palette["FIG_BG"])
    ax.set_facecolor(palette["AX_BG"])


    # Glow layer (background points)

    sns.scatterplot(
        data=df,
        x=price_col,
        y=quantity_col,
        hue=category_col,
        palette=FUTURISTIC_BLUES,
        alpha=0.08,
        s=180,
        linewidth=0,
        legend=False,
        ax=ax
    )

    # Main scatter layer

    sns.scatterplot(
        data=df,
        x=price_col,
        y=quantity_col,
        hue=category_col,
        palette=FUTURISTIC_BLUES,
        alpha=0.78,
        s=60,
        edgecolor="#05070F",
        linewidth=0.8,
        ax=ax
    )

    # Labels and styling

    ax.set_title(
        title,
        color=palette["TITLE"],
        fontsize=13,
        fontweight="bold",
        pad=12
    )

    ax.set_xlabel(xlabel, color=palette["TEXT"])
    ax.set_ylabel(ylabel, color=palette["TEXT"])

    # Grid
    ax.grid(axis="both", color=palette["GRID"], alpha=0.14, linewidth=0.7)

    # Ticks
    ax.tick_params(axis="x", colors=palette["XTICK"], length=0)
    ax.tick_params(axis="y", colors=palette["YTICK"], length=0)

    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Legend
    legend = ax.legend(
        title=category_col,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=False
    )

    if legend is not None:
        legend.get_title().set_color(palette["TEXT"])
        for text in legend.get_texts():
            text.set_color(palette["TEXT"])

    plt.tight_layout()

    if show:
        plt.show()

    return fig, ax

# ==================== QQ plot ====================

def plot_qq_residuals(
    residuals,
    title="QQ Plot of Residuals",
    xlabel="Theoretical Quantiles",
    ylabel="Observed Quantiles",
    figsize=(7, 5),
    show=True
):
    """
    Plot a futuristic QQ plot to assess normality of residuals.

    This plot compares the distribution of residuals to a theoretical
    normal distribution. It is commonly used to evaluate model assumptions.

    Args:
        residuals (array-like): Residual values from a model.
        title (str): Plot title.
        xlabel (str): Label for x-axis.
        ylabel (str): Label for y-axis.
        figsize (tuple): Figure size.
        show (bool): Whether to display the plot.

    Returns:
        tuple: (fig, ax) Matplotlib figure and axis.
    """

    palette = {
        "FIG_BG": "#05070F",
        "AX_BG": "#05070F",
        "GRID": "#FFFFFF",
        "TEXT": "#D6E3FF",
        "TITLE": "#E6EDFF",
        "XTICK": "#9FB3FF",
        "YTICK": "#9FB3FF",
        "POINT": "#38BDF8",
        "LINE": "#2563EB"
    }

    residuals = np.asarray(residuals)

    if len(residuals) == 0:
        raise ValueError("Residuals array is empty.")

    # QQ computation

    (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm")

 
    # Plot setup
   
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(palette["FIG_BG"])
    ax.set_facecolor(palette["AX_BG"])

    # Glow layer (background points)
    ax.scatter(
        osm,
        osr,
        color=palette["POINT"],
        alpha=0.15,
        s=140,
        zorder=1
    )

    # Main points
    ax.scatter(
        osm,
        osr,
        color=palette["POINT"],
        edgecolor="#05070F",
        linewidth=0.8,
        s=50,
        zorder=3
    )

    # Theoretical normal line
    x_line = np.linspace(np.min(osm), np.max(osm), 100)

    ax.plot(
        x_line,
        slope * x_line + intercept,
        color=palette["LINE"],
        linewidth=2.5,
        zorder=2
    )

    # Labels and styling

    ax.set_title(
        title,
        fontsize=13,
        fontweight="bold",
        color=palette["TITLE"],
        pad=12
    )

    ax.set_xlabel(xlabel, color=palette["TEXT"])
    ax.set_ylabel(ylabel, color=palette["TEXT"])

    # Grid
    ax.grid(color=palette["GRID"], alpha=0.18)

    # Ticks
    ax.tick_params(axis="x", colors=palette["XTICK"], length=0)
    ax.tick_params(axis="y", colors=palette["YTICK"], length=0)

    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()

    if show:
        plt.show()

    return fig, ax

# ==================== Plot elasticite ====================

def _prepare_visualization_series(
    sub,
    value_col,
    date_col="DATE",
    freq="ME",
    smooth_window=2,
    interp_method="time",
    interp_limit=None,
):
    """
    Prepare a resampled and smoothed time series for visualization.

    Args:
        sub (pandas.DataFrame): Input subset of data.
        value_col (str): Numerical column to visualize.
        date_col (str): Date column name.
        freq (str): Resampling frequency.
        smooth_window (int): Rolling window size used for smoothing.
        interp_method (str): Interpolation method passed to pandas.
        interp_limit (int | None): Maximum number of consecutive NaN values
            to fill during interpolation.

    Returns:
        pandas.DataFrame: Resampled, interpolated, and smoothed time series.
    """
    sub = sub.sort_values(date_col).set_index(date_col)
    sub = sub[[value_col]].resample(freq).mean()

    if interp_limit is None:
        sub[value_col] = sub[value_col].interpolate(method=interp_method)
    else:
        sub[value_col] = sub[value_col].interpolate(
            method=interp_method,
            limit=interp_limit
        )

    # Apply rolling smoothing to reduce short-term noise.
    sub[value_col] = sub[value_col].rolling(
        smooth_window,
        min_periods=1
    ).mean()

    return sub


def _draw_glow_line(ax, x, y, color, label=None):
    """
    Draw a futuristic line with glow and halo effects.

    Args:
        ax (matplotlib.axes.Axes): Target axis.
        x (array-like): X-axis values.
        y (array-like): Y-axis values.
        color (str or tuple): Line color.
        label (str | None): Legend label.
    """
    # Soft glow layer behind the main line.
    ax.plot(
        x,
        y,
        color=color,
        linewidth=7,
        alpha=0.12,
        solid_capstyle="round",
        zorder=1
    )

    # Main line with markers.
    ax.plot(
        x,
        y,
        color=color,
        linewidth=2.4,
        linestyle="-",
        drawstyle="default",
        marker="o",
        markersize=5,
        markerfacecolor="#05070F",
        markeredgecolor=color,
        markeredgewidth=1.4,
        solid_capstyle="round",
        label=label,
        zorder=3
    )

    # Point halo effect.
    ax.scatter(
        x,
        y,
        s=110,
        color=color,
        alpha=0.10,
        zorder=2
    )


def _style_futuristic_axis(ax, title, text_color="#D6E3FF"):
    """
    Apply futuristic styling to a matplotlib axis.

    Args:
        ax (matplotlib.axes.Axes): Target axis.
        title (str): Axis title.
        text_color (str): General text color used for legend text.
    """
    palette = {
        "AX_BG": "#05070F",
        "GRID": "#FFFFFF",
        "TEXT": "#D6E3FF",
        "TITLE": "#E6EDFF",
        "XTICK": "#9FB3FF",
        "YTICK": "#9FB3FF",
    }

    ax.set_facecolor(palette["AX_BG"])
    ax.set_title(
        title,
        color=palette["TITLE"],
        fontsize=13,
        fontweight="bold",
        pad=10
    )

    ax.grid(axis="y", color=palette["GRID"], alpha=0.18, linewidth=0.7)
    ax.grid(axis="x", color=palette["GRID"], alpha=0.10, linewidth=0.6)

    ax.tick_params(axis="x", colors=palette["XTICK"], length=0)
    ax.tick_params(axis="y", colors=palette["YTICK"], length=0)

    for spine in ax.spines.values():
        spine.set_visible(False)

    legend = ax.legend(frameon=False, loc="best")
    if legend is not None:
        legend.get_title().set_color(text_color)
        for text in legend.get_texts():
            text.set_color(text_color)


def _get_futuristic_blue_colors(n):
    """
    Generate a sequence of blue shades for multiple time series.

    Args:
        n (int): Number of colors required.

    Returns:
        list: List of RGBA colors from the matplotlib Blues colormap.
    """
    cmap = plt.cm.Blues
    positions = np.linspace(0.45, 0.95, max(n, 1))
    return [cmap(p) for p in positions]


def plot_elasticity_dashboard(
    df,
    top_categories_sales,
    top_categories_variance,
    top_skus_sales,
    top_skus_variance,
    date_col="DATE",
    category_col="Category",
    sku_col="SKU",
    cat_roll_col="elasticite_rolling_Category",
    cat_var_col="var_elasticite_rolling_Category",
    sku_roll_col="elasticite_rolling_SKU",
    sku_var_col="var_elasticite_rolling_SKU",
    freq="ME",
    smooth_window=2,
    interp_method="time",
    interp_limit=None,
    figsize=(18, 10),
    show_reference_line=True,
    reference_value=-1,
    show=True,
):
    """
    Plot a 2x2 futuristic dashboard for elasticity and elasticity variance.

    The dashboard contains:
    - category rolling elasticity
    - category elasticity variance
    - SKU rolling elasticity
    - SKU elasticity variance

    Args:
        df (pandas.DataFrame): Input dataset.
        top_categories_sales (list): Top categories for rolling elasticity.
        top_categories_variance (list): Top categories for elasticity variance.
        top_skus_sales (list): Top SKUs for rolling elasticity.
        top_skus_variance (list): Top SKUs for elasticity variance.
        date_col (str): Date column name.
        category_col (str): Category column name.
        sku_col (str): SKU column name.
        cat_roll_col (str): Column for category rolling elasticity.
        cat_var_col (str): Column for category rolling elasticity variance.
        sku_roll_col (str): Column for SKU rolling elasticity.
        sku_var_col (str): Column for SKU rolling elasticity variance.
        freq (str): Resampling frequency.
        smooth_window (int): Rolling smoothing window.
        interp_method (str): Interpolation method.
        interp_limit (int | None): Maximum interpolation gap.
        figsize (tuple): Figure size.
        show_reference_line (bool): Whether to draw the elasticity reference line.
        reference_value (float): Reference value for elasticity.
        show (bool): Whether to display the figure.

    Returns:
        tuple: Matplotlib figure and axes array.
    """
    palette = {
        "FIG_BG": "#05070F",
        "REF_LINE": "#93C5FD",
        "TEXT": "#D6E3FF",
    }

    # Defensive copy and date conversion for robust plotting.
    data = df.copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    data = data.dropna(subset=[date_col])

    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=False)
    fig.patch.set_facecolor(palette["FIG_BG"])

   
    # Category rolling elasticity

    ax = axes[0, 0]
    colors = _get_futuristic_blue_colors(len(top_categories_sales))

    for color, cat in zip(colors, top_categories_sales):
        sub = data[data[category_col] == cat]
        series = _prepare_visualization_series(
            sub=sub,
            value_col=cat_roll_col,
            date_col=date_col,
            freq=freq,
            smooth_window=smooth_window,
            interp_method=interp_method,
            interp_limit=interp_limit,
        )

        _draw_glow_line(
            ax=ax,
            x=series.index,
            y=series[cat_roll_col],
            color=color,
            label=str(cat)
        )

    if show_reference_line:
        ax.axhline(
            reference_value,
            color=palette["REF_LINE"],
            linestyle="--",
            linewidth=1.6,
            alpha=0.7,
            zorder=0
        )
        ax.axhline(
            reference_value,
            color=palette["REF_LINE"],
            linewidth=6,
            alpha=0.08,
            zorder=0
        )

    _style_futuristic_axis(ax, "Categories - Rolling Elasticity", text_color=palette["TEXT"])

    # Category elasticity variance

    ax = axes[0, 1]
    colors = _get_futuristic_blue_colors(len(top_categories_variance))

    for color, cat in zip(colors, top_categories_variance):
        sub = data[data[category_col] == cat]
        series = _prepare_visualization_series(
            sub=sub,
            value_col=cat_var_col,
            date_col=date_col,
            freq=freq,
            smooth_window=smooth_window,
            interp_method=interp_method,
            interp_limit=interp_limit,
        )

        _draw_glow_line(
            ax=ax,
            x=series.index,
            y=series[cat_var_col],
            color=color,
            label=str(cat)
        )

    _style_futuristic_axis(ax, "Categories - Elasticity Variance", text_color=palette["TEXT"])

 
    # SKU rolling elasticity
  
    ax = axes[1, 0]
    colors = _get_futuristic_blue_colors(len(top_skus_sales))

    for color, sku in zip(colors, top_skus_sales):
        sub = data[data[sku_col] == sku]
        series = _prepare_visualization_series(
            sub=sub,
            value_col=sku_roll_col,
            date_col=date_col,
            freq=freq,
            smooth_window=smooth_window,
            interp_method=interp_method,
            interp_limit=interp_limit,
        )

        _draw_glow_line(
            ax=ax,
            x=series.index,
            y=series[sku_roll_col],
            color=color,
            label=str(sku)
        )

    if show_reference_line:
        ax.axhline(
            reference_value,
            color=palette["REF_LINE"],
            linestyle="--",
            linewidth=1.6,
            alpha=0.7,
            zorder=0
        )
        ax.axhline(
            reference_value,
            color=palette["REF_LINE"],
            linewidth=6,
            alpha=0.08,
            zorder=0
        )

    _style_futuristic_axis(ax, "SKU - Rolling Elasticity", text_color=palette["TEXT"])


    # SKU elasticity variance

    ax = axes[1, 1]
    colors = _get_futuristic_blue_colors(len(top_skus_variance))

    for color, sku in zip(colors, top_skus_variance):
        sub = data[data[sku_col] == sku]
        series = _prepare_visualization_series(
            sub=sub,
            value_col=sku_var_col,
            date_col=date_col,
            freq=freq,
            smooth_window=smooth_window,
            interp_method=interp_method,
            interp_limit=interp_limit,
        )

        _draw_glow_line(
            ax=ax,
            x=series.index,
            y=series[sku_var_col],
            color=color,
            label=str(sku)
        )

    _style_futuristic_axis(ax, "SKU - Elasticity Variance", text_color=palette["TEXT"])

  
    # Global date formatting

    for ax in axes.ravel():
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()

    if show:
        plt.show()

    return fig, axes


# ==================== Basket intensity ====================

def _prepare_monthly_series(
    sub,
    value_col,
    date_col="DATE",
    freq="ME",
    interp_method="time",
    smooth_window=2
):
    """
    Prepare a monthly resampled and smoothed time series.

    Args:
        sub (pandas.DataFrame): Input subset.
        value_col (str): Column to aggregate.
        date_col (str): Date column.
        freq (str): Resampling frequency.
        interp_method (str): Interpolation method.
        smooth_window (int): Rolling smoothing window.

    Returns:
        pandas.DataFrame: Processed time series.
    """
    sub = sub.sort_values(date_col).set_index(date_col)[[value_col]]
    s = sub.resample(freq).mean()

    # Fill missing values
    s[value_col] = s[value_col].interpolate(method=interp_method)

    # Smooth series
    s[value_col] = s[value_col].rolling(
        smooth_window,
        min_periods=1
    ).mean()

    return s


def _draw_glow_line(ax, x, y, color, label=None):
    """
    Draw a futuristic line with glow and halo effects.

    Args:
        ax (matplotlib.axes.Axes): Target axis.
        x (array-like): X-axis values.
        y (array-like): Y-axis values.
        color (str or tuple): Line color.
        label (str | None): Legend label.
    """
    ax.plot(x, y, color=color, linewidth=7, alpha=0.12, zorder=1)

    ax.plot(
        x, y,
        color=color,
        linewidth=2.4,
        linestyle="-",
        marker="o",
        markersize=5,
        markerfacecolor="#05070F",
        markeredgecolor=color,
        markeredgewidth=1.4,
        label=label,
        zorder=3
    )

    ax.scatter(x, y, s=110, color=color, alpha=0.10, zorder=2)


def _style_axis(ax, title, xlabel, ylabel):
    """
    Apply futuristic styling to an axis.

    Args:
        ax (matplotlib.axes.Axes): Target axis.
        title (str): Plot title.
        xlabel (str): X-axis label.
        ylabel (str): Y-axis label.
    """
    palette = {
        "AX_BG": "#05070F",
        "GRID": "#FFFFFF",
        "TEXT": "#D6E3FF",
        "TITLE": "#E6EDFF",
        "XTICK": "#9FB3FF",
        "YTICK": "#9FB3FF",
    }

    ax.set_facecolor(palette["AX_BG"])
    ax.set_title(title, color=palette["TITLE"], fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, color=palette["TEXT"])
    ax.set_ylabel(ylabel, color=palette["TEXT"])

    ax.grid(axis="y", color=palette["GRID"], alpha=0.18)
    ax.grid(axis="x", color=palette["GRID"], alpha=0.10)

    ax.tick_params(axis="x", colors=palette["XTICK"], length=0)
    ax.tick_params(axis="y", colors=palette["YTICK"], length=0)

    for spine in ax.spines.values():
        spine.set_visible(False)

    legend = ax.legend(frameon=False, loc="best")
    if legend is not None:
        legend.get_title().set_color(palette["TEXT"])
        for text in legend.get_texts():
            text.set_color(palette["TEXT"])


def _get_blue_colors(n):
    """
    Generate blue color palette for multiple series.

    Args:
        n (int): Number of colors.

    Returns:
        list: List of colors.
    """
    cmap = plt.cm.Blues
    positions = np.linspace(0.45, 0.95, max(n, 1))
    return [cmap(p) for p in positions]


def plot_basket_intensity_dashboard(
    df,
    top_categories,
    top_skus,
    date_col="DATE",
    category_col="Category",
    sku_col="SKU",
    figsize=(18, 10),
    show=True
):
    """
    Plot a 2x2 dashboard for basket intensity analysis.

    The dashboard includes:
    - Category basket intensity (normalized by category)
    - Category basket intensity (normalized by SKU)
    - SKU basket intensity (normalized by SKU)
    - SKU average monthly quantity

    Args:
        df (pandas.DataFrame): Input dataset.
        top_categories (list): Top categories to display.
        top_skus (list): Top SKUs to display.
        date_col (str): Date column name.
        category_col (str): Category column.
        sku_col (str): SKU column.
        figsize (tuple): Figure size.
        show (bool): Whether to display the plot.

    Returns:
        tuple: (fig, axes)
    """

    palette = {
        "FIG_BG": "#05070F",
        "REF_LINE": "#93C5FD",
    }

    data = df.copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    data = data.dropna(subset=[date_col])

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.patch.set_facecolor(palette["FIG_BG"])

    
    # Category - norm cat
 
    ax = axes[0, 0]
    colors = _get_blue_colors(len(top_categories))

    for color, cat in zip(colors, top_categories):
        sub = data[data[category_col] == cat]
        s = _prepare_monthly_series(sub, "basket_intensity_cat_norm_plot")

        _draw_glow_line(ax, s.index, s.iloc[:, 0], color, label=cat)

    ax.axhline(1.0, color=palette["REF_LINE"], linestyle="--", alpha=0.7)
    _style_axis(ax,
        "Top Categories – Basket Intensity (Normalized by Category)",
        "Date",
        "Intensity (≈1 = baseline)"
    )


    # Category - norm SKU
   
    ax = axes[0, 1]
    colors = _get_blue_colors(len(top_categories))

    for color, cat in zip(colors, top_categories):
        sub = data[data[category_col] == cat]
        s = _prepare_monthly_series(sub, "basket_intensity_sku_norm_plot")

        _draw_glow_line(ax, s.index, s.iloc[:, 0], color, label=cat)

    ax.axhline(1.0, color=palette["REF_LINE"], linestyle="--", alpha=0.7)
    _style_axis(ax,
        "Top Categories – Basket Intensity (Normalized by SKU)",
        "Date",
        "Intensity (≈1 = baseline)"
    )

   
    # SKU - norm SKU
   
    ax = axes[1, 0]
    colors = _get_blue_colors(len(top_skus))

    for color, sku in zip(colors, top_skus):
        sub = data[data[sku_col] == sku]
        s = _prepare_monthly_series(sub, "basket_intensity_sku_norm_plot")

        _draw_glow_line(ax, s.index, s.iloc[:, 0], color, label=sku)

    ax.axhline(1.0, color=palette["REF_LINE"], linestyle="--", alpha=0.7)
    _style_axis(ax,
        "Top SKUs – Basket Intensity (Normalized by SKU)",
        "Date",
        "Intensity (≈1 = baseline)"
    )


    # SKU - quantity

    ax = axes[1, 1]
    colors = _get_blue_colors(len(top_skus))

    for color, sku in zip(colors, top_skus):
        sub = data[data[sku_col] == sku]
        s = _prepare_monthly_series(sub, "quantity")

        _draw_glow_line(ax, s.index, s.iloc[:, 0], color, label=sku)

    _style_axis(ax,
        "Top SKUs – Average Monthly Quantity",
        "Date",
        "Quantity (mean)"
    )

    # Date formatting
    for ax in axes.ravel():
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()

    if show:
        plt.show()

    return fig, axes