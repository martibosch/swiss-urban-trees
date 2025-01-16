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


def plot_annot_vs_pred(
    annot_gdf: gpd.GeoDataFrame,
    pred_gdf: gpd.GeoDataFrame,
    tile_dir: utils.PathDType,
    *,
    figwidth: float | None = None,
    figheight: float | None = None,
    pred_title: str = "predictions",
    legend: bool = True,
    plot_annot_kwargs: utils.KwargsDType = None,
    plot_pred_kwargs: utils.KwargsDType = None,
):
    """Plot annotations and predictions side by side for each image.

    Parameters
    ----------
    annot_gdf, pred_gdf : gpd.GeoDataFrame
        Annotations and predictions geo-data frames respectively.
    tile_dir : path-like
        Path to the directory containing the tiles.
    figwidth, figheight : numeric, optional
        Figure width and height. If None, the matplotlib defaults are used.
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

    # small helper to plot image and gdf
    def _plot_img_and_gdf(src, gdf, ax, **plot_kwargs):
        plot.show(src, with_bounds=False, ax=ax)
        left, right = ax.get_xlim()
        bottom, top = ax.get_ylim()
        gdf.assign(**{"geometry": gdf.boundary}).plot(ax=ax, **plot_kwargs)
        ax.set_xlim(left, right)
        ax.set_ylim(bottom, top)

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

    num_rows = len(img_filenames)
    num_cols = len(patch_sizes) + 1
    fig, axes = plt.subplots(
        num_rows, num_cols, figsize=(figwidth * num_cols, figheight * num_rows)
    )
    # in case there is only one image
    if num_rows == 1:
        axes = np.array([axes])

    if legend:
        _plot_annot_kwargs["legend"] = _plot_annot_kwargs.pop("legend", False)
        _plot_pred_kwargs["legend"] = _plot_pred_kwargs.pop("legend", True)
        _plot_pred_kwargs["legend_kwds"] = _plot_annot_kwargs.pop(
            "legend_kwds", {"loc": "center right", "bbox_to_anchor": (1.5, 0.5)}
        )

    for img_filename, ax_row in zip(img_filenames, axes):
        with rio.open(
            path.join(tile_dir, f"{path.splitext(img_filename)[0]}.tif")
        ) as src:
            _plot_img_and_gdf(
                src,
                annot_gdf[annot_gdf["image_path"] == img_filename],
                ax_row[0],
                **_plot_annot_kwargs,
            )
            # ACHTUNG: `pred_df` was assigned the `.tif` image instead of the jpeg
            # TODO: how to handle this properly?
            for (patch_size, patch_gdf), ax in zip(
                pred_gdf[
                    pred_gdf["image_path"] == f"{path.splitext(img_filename)[0]}.tif"
                ].groupby("patch_size"),
                ax_row[1:],
            ):
                _plot_img_and_gdf(src, patch_gdf, ax, **_plot_pred_kwargs)

    if legend:
        # ideally we could do something like
        # fig.legend(
        #     *fig.axes[-1].get_legend_handles_labels(),
        #     loc="center right",
        #     bbox_to_anchor=(1, 1),
        # )
        # but we get no handles and labels for the legend, maybe related to
        # https://github.com/geopandas/geopandas/issues/660
        row_indices = np.arange(len(axes))
        # TODO: how to handle even number of rows?
        middle_row = row_indices[len(row_indices) // 2]
        # iterate first half of rows
        for row_i in row_indices:
            # the first axis of each row is the annotations, so no need to remove
            if row_i == middle_row:
                # in the middle row, leave the legend of the last axis
                _axes = axes[row_i][1:-1]
            else:
                _axes = axes[row_i][1:]
            for ax in _axes:
                ax.get_legend().remove()

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
