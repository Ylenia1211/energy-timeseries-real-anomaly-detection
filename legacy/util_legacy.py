from typing import Dict, List, Tuple
import seaborn as sns
from matplotlib import pyplot as plt, ticker
import pandas as pd


def load_building_df(
    column: str, filename: str, model: str, building_id: str, resample: str = "1h"
):
    """
    Load and preprocess building data from a CSV file.

    Parameters:
        column (str): The column name to be extracted from the CSV file.
        filename (str): The name of the CSV file containing the data.
        model (str): The model type of the building data.
        building_id (str): The ID of the building for which data is being loaded.
        resample (str, optional): The time frequency to resample the data to. Default is "1h" (1 hour).

    Returns:
        pandas.DataFrame: A DataFrame containing the preprocessed building data with the specified column, resampled per the specified frequency,
                          and with an additional 'dayofweek' column representing the day of the week for each timestamp.
    """
    # Read the CSV file
    df_raw = pd.read_csv(f"./data/{filename}")

    # Filter the DataFrame based on model and building ID, and select required columns
    df = df_raw[(df_raw["model"] == model) & (df_raw["id"] == building_id)][
        ["datetime", column]
    ]

    # Convert 'datetime' column to datetime datatype
    df["datetime"] = pd.to_datetime(df["datetime"])

    # Set 'datetime' column as the index
    df = df.set_index("datetime")

    # Resample the data per hour and sum the values
    df = df.resample(resample).sum()

    # Add a column representing the day of the week for each timestamp
    df["hour"] = df.index.hour  # type: ignore
    df["dayofweek"] = df.index.dayofweek  # type: ignore
    df["week"] = df.index.day_of_year // 7  # type: ignore

    return df


def show_overview_for_value(df: pd.DataFrame, column: str):
    """
    Display an overview of descriptive statistics for a specific column of a DataFrame, grouped by day of the week.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        column (str): The column name for which an overview is to be shown.

    Returns:
        pd.DataFrame: A DataFrame displaying descriptive statistics (min, max, mean, std) of the specified column,
                      grouped by day of the week.
    """
    df = df.groupby("dayofweek").agg({column: ["min", "max", "mean", "std"]})
    df.style.background_gradient()

    return df.style.background_gradient()


def draw_barplot(
    df: pd.DataFrame | pd.Series,
    x: str | pd.Index,
    y: str | pd.Index | pd.Series,
    lw: int = 1,
    major_locator: int = 0,
    figsize: tuple = (20, 5),
    rotation: int = 90,
    visible_labels: bool = True,
    hue: str | None = None,
    title: str | None = None,
) -> Tuple:
    """
    Draw a bar plot using Seaborn.

    Parameters:
        df (pd.DataFrame | pd.Series): The data to be plotted.
        x (str | pd.Index): The column name or index to be used as the x-axis.
        y (str | pd.Index | pd.Series): The column name, index, or Series to be used as the y-axis.
        lw (int): Width of lines.
        major_locator (int): Interval for major tick marks on the x-axis.
        figsize (tuple): Figure size (width, height) in inches.
        rotation (int): Rotation angle for x-axis labels.
        visible_labels (bool): Whether to display x-axis labels.
        hue (str | None): Optional, column name for color encoding.
        title (str | None): Optional, title for the plot.

    Returns:
        Tuple: Tuple containing the axis and figure objects.
    """
    ax, fig = plt.subplots(figsize=figsize)

    ax = sns.barplot(data=df, x=x, y=y, hue=hue, lw=lw)  # type: ignore

    if major_locator > 0:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(major_locator))

    ax.set(title=title)
    ax.set_xticklabels(labels=df.index, rotation=rotation)

    if not visible_labels:
        ax.xaxis.set_ticklabels([])

    return ax, fig


def add_zscore_plot(ax, df, colname, color="#000000AA"):
    """
    Add a plot of z-scores for a specific column to the given axis.

    Parameters:
        ax (matplotlib.axes._subplots.AxesSubplot): The axis to which the z-score plot will be added.
        df (pd.DataFrame): The DataFrame containing the data.
        colname (str): The column name for which the z-score plot will be generated.
        color (str): The color of the z-score plot.
    """
    ax_zscore = ax.twinx()
    ax_zscore.set_ylabel(f"Z-Score ({colname})")
    ax_zscore.grid(False)
    ax_zscore.plot(df[colname], color=color)


def plot_dataframe(
    df: pd.DataFrame,
    columns: List[str],
    colors: Dict = {},
    figsize: tuple = (18, 8),
    tick_interval_mul: float = 0.025,
    ax=None,
):
    """Plots a list of dataframe columns, also dynamically adjusting the number
    of plotted xlabels.

    Args:
        columns (List[str]): List of strings representing the columns' names
        figsize (tuple, optional): Size of plot. Defaults to (18, 7).
        tick_interval_mul (float): Regulates frequency of xlabels.
    """

    if not ax:
        _, ax = plt.subplots(1, 1, figsize=figsize)

    tick_interval = int(len(df.index) * tick_interval_mul)

    # plot all selected columns
    for col in columns:
        ax.plot(df[col], label=col, c=colors.get(col, None))

    # set plot features
    ax.set_xticks(df.index[::tick_interval])
    ax.set_xticklabels(df.index[::tick_interval], rotation=90)
    ax.legend(loc="upper left")
    ax.grid()

    # plt.ylim(df.min())
    plt.tight_layout()

    return ax
