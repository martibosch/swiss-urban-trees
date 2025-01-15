"""Plotting utils."""

from os import path

import matplotlib.pyplot as plt
import pandas as pd
import rasterio as rio
import seaborn as sns
from deepforest import visualize

from swiss_urban_trees import utils


def plot_annot_vs_pred(
    annot_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    tile_dir: utils.PathDType,
    *,
    figwidth: float | None = None,
    figheight: float | None = None,
    pred_title: str = "predictions",
    plot_annot_kwargs: utils.KwargsDType = None,
    plot_pred_kwargs: utils.KwargsDType = None,
):
    """Plot annotations and predictions side by side for each image.

    Parameters
    ----------
    annot_df, pred_df : pd.DataFrame
        Annotations and predictions data frames respectively.
    tile_dir : path-like
        Path to the directory containing the tiles.
    figwidth, figheight : numeric, optional
        Figure width and height. If None, the matplotlib defaults are used.
    pred_title : str, optional
        Title for the predictions plots.
    plot_annot_kwargs, plot_pred_kwargs : mapping, optional
        Keyword arguments to pass to `deepforest.visualize.plot_predictions` when
        plotting annotations and predictions, respectively.

    Returns
    -------
    fig : matplotlib.figure.Figure

    """
    img_filenames = annot_df["image_path"].unique()
    patch_sizes = pred_df["patch_size"].unique()

    if figwidth is None:
        figwidth = plt.rcParams["figure.figsize"][0]
    if figheight is None:
        figheight = plt.rcParams["figure.figsize"][1]
    if plot_annot_kwargs is None:
        plot_annot_kwargs = {}
    if plot_pred_kwargs is None:
        plot_pred_kwargs = {}

    num_rows = len(img_filenames)
    num_cols = len(patch_sizes) + 1
    fig, axes = plt.subplots(
        num_rows, num_cols, figsize=(figwidth * num_cols, figheight * num_rows)
    )
    # in case there is only one image
    if num_rows == 1:
        axes = [axes]

    for img_filename, ax_row in zip(img_filenames, axes):
        with rio.open(
            path.join(tile_dir, f"{path.splitext(img_filename)[0]}.tif")
        ) as src:
            img_arr = src.read()
            ax_row[0].imshow(
                visualize.plot_predictions(
                    img_arr,
                    annot_df[annot_df["image_path"] == img_filename],
                    **plot_annot_kwargs,
                )[:, :, ::-1]
            )
            # ACHTUNG: `pred_df` was assigned the `.tif` image instead of the jpeg
            # TODO: how to handle this properly?
            for (patch_size, patch_df), ax in zip(
                pred_df[
                    pred_df["image_path"] == f"{path.splitext(img_filename)[0]}.tif"
                ].groupby("patch_size"),
                ax_row[1:],
            ):
                ax.imshow(
                    visualize.plot_predictions(
                        img_arr,
                        patch_df,
                        **plot_pred_kwargs,
                    )[:, :, ::-1]
                )

    for title, ax in zip(
        ["annotations"]
        + [f"{pred_title} (patch size: {patch_size})" for patch_size in patch_sizes],
        axes[0],
    ):
        ax.set_title(title)

    return fig


def metrics_barplot(
    metrics_df: pd.DataFrame, **barplot_kwargs: utils.KwargsDType
) -> plt.Axes:
    """Plot a barplot of the metrics computed for different patch sizes.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Data frame containing the metrics computed for different patch sizes.
    barplot_kwargs : mapping, optional
        Keyword arguments to pass to `seaborn.barplot`.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes object containing the plot.

    """
    return sns.barplot(
        pd.melt(
            metrics_df.reset_index(),
            id_vars="patch_size",
            var_name="metric",
        ),
        x="patch_size",
        y="value",
        hue="metric",
        **barplot_kwargs,
    )
