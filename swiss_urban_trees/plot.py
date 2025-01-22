"""Plotting utils."""

from os import path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio as rio
import seaborn as sns
from rasterio import plot

from swiss_urban_trees import utils


def plot_img_and_gdf(src, gdf, *, ax=None, **plot_kwargs) -> plt.Axes:
    """Small helper to plot an image and bounding box geo-data frame over it."""
    if ax is None:
        _, ax = plt.subplots()
    plot.show(src, with_bounds=False, ax=ax)
    left, right = ax.get_xlim()
    bottom, top = ax.get_ylim()
    gdf.assign(**{"geometry": gdf.boundary}).plot(ax=ax, **plot_kwargs)
    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)

    return ax


def plot_annot_vs_pred(
    annot_gdf: gpd.GeoDataFrame,
    pred_gdf: gpd.GeoDataFrame,
    tile_dir: utils.PathDType,
    *,
    figwidth: float | None = None,
    figheight: float | None = None,
    col_wrap: int | None = 3,
    pred_title: str = "predictions",
    legend: bool = True,
    plot_annot_kwargs: utils.KwargsDType = None,
    plot_pred_kwargs: utils.KwargsDType = None,
) -> plt.Figure:
    """Plot annotations and predictions side by side for each image.

    Parameters
    ----------
    annot_gdf, pred_gdf : gpd.GeoDataFrame
        Annotations and predictions geo-data frames respectively.
    tile_dir : path-like
        Path to the directory containing the tiles.
    figwidth, figheight : numeric, optional
        Figure width and height. If None, the matplotlib defaults are used.
    col_wrap : int, default 3
        Number of columns to wrap the plots at. Ignored if the provided value is greater
        than the number of patch sizes.
    pred_title : str, default "predictions"
        Title for the predictions plots.
    legend : bool, default True
        Whether to show the legend on the last plot.
    plot_annot_kwargs, plot_pred_kwargs : mapping, optional
        Keyword arguments to pass to `geopandas.GeoDataFrame.plot` when plotting
        annotations and predictions respectively.

    Returns
    -------
    fig : matplotlib.figure.Figure

    """
    img_filenames = annot_gdf["image_path"].unique()
    patch_sizes = pred_gdf["patch_size"].unique()

    if figwidth is None:
        figwidth = plt.rcParams["figure.figsize"][0]
    if figheight is None:
        figheight = plt.rcParams["figure.figsize"][1]
    if plot_annot_kwargs is None:
        _plot_annot_kwargs = {}
    else:
        _plot_annot_kwargs = plot_annot_kwargs.copy()
    if plot_pred_kwargs is None:
        _plot_pred_kwargs = {}
    else:
        _plot_pred_kwargs = plot_pred_kwargs.copy()

    if legend:
        _plot_annot_kwargs["legend"] = _plot_annot_kwargs.pop("legend", False)
        _plot_pred_kwargs["legend"] = _plot_pred_kwargs.pop("legend", True)
        _plot_pred_kwargs["legend_kwds"] = _plot_annot_kwargs.pop(
            "legend_kwds", {"loc": "center right", "bbox_to_anchor": (1.5, 0.5)}
        )

    num_cols = min(len(patch_sizes) + 1, col_wrap)
    # num_rows = len(img_filenames)
    num_plots = len(img_filenames) * (len(patch_sizes) + 1)
    num_rows = num_plots // num_cols
    if num_plots % num_cols > 0:
        num_rows += 1

    fig, axes = plt.subplots(
        num_rows, num_cols, figsize=(figwidth * num_cols, figheight * num_rows)
    )
    # in case there is only one image
    if num_rows == 1:
        axes = np.array([axes])

    num_rows_per_img = (len(patch_sizes) + 1) // num_cols
    if (len(patch_sizes) + 1) % num_cols > 0:
        num_rows_per_img += 1
    i = 0
    for img_filename in img_filenames:
        with rio.open(path.join(tile_dir, img_filename)) as src:
            plot_img_and_gdf(
                src,
                annot_gdf[annot_gdf["image_path"] == img_filename],
                ax=axes.flat[i],
                **_plot_annot_kwargs,
            )
            # ACHTUNG: `pred_df` was assigned the `.tif` image instead of the jpeg
            # TODO: how to handle this properly?
            for j, (patch_size, patch_gdf) in enumerate(
                pred_gdf[pred_gdf["image_path"] == img_filename].groupby("patch_size"),
                start=1,
            ):
                plot_img_and_gdf(
                    src, patch_gdf, ax=axes.flat[i + j], **_plot_pred_kwargs
                )
            for k in range(j + 1, num_cols * num_rows_per_img):
                axes.flat[i + k].axis("off")
            i += num_cols * num_rows_per_img

    if legend:
        # ideally we could do something like
        # fig.legend(
        #     *fig.axes[-1].get_legend_handles_labels(),
        #     loc="center right",
        #     bbox_to_anchor=(1, 1),
        # )
        # but we get no handles and labels for the legend, maybe related to
        # https://github.com/geopandas/geopandas/issues/660
        def remove_legend(ax):
            try:
                ax.get_legend().remove()
            except AttributeError:
                # do not raise error if there is no legend
                pass

        i = 0
        for img_filename in img_filenames:
            remove_legend(axes.flat[i])
            for j in range(1, len(patch_sizes)):
                remove_legend(axes.flat[i + j])

    # axis titles
    i = 0
    for img_filename in img_filenames:
        axes.flat[i].set_title("annotations")
        for j, patch_size in enumerate(patch_sizes, start=1):
            axes.flat[i + j].set_title(f"{pred_title} (patch size: {patch_size})")
        i += num_cols * num_rows_per_img

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
