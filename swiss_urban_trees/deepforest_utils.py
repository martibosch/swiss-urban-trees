"""DeepForest utils."""

import tempfile
import time
from os import path

import geopandas as gpd
import pandas as pd
from deepforest import evaluate, main

from swiss_urban_trees import utils


def eval_metrics(
    eval_df: pd.DataFrame, pred_df: pd.DataFrame, iou_threshold: float = 0.4
):
    """Compute IoU, recall, precision and F1 score for a given patch size.

    Parameters
    ----------
    eval_df : pd.DataFrame
        Evaluation data frame.
    pred_df : pd.DataFrame
        Predictions data frame, only used to get the number of predictions and compute
        the precision metric.
    iou_threshold : numeric, optional
        Intersection over union threshold to consider a prediction a match.

    Returns
    -------
    pd.Series
        IoU, recall, precision and F1 metrics.

    """
    n_true_positives = (eval_df["IoU"] > iou_threshold).sum()
    recall = n_true_positives / eval_df.shape[0]
    precision = (
        n_true_positives / pred_df[pred_df["patch_size"] == eval_df.name].shape[0]
    )
    return pd.Series(
        {
            "IoU": eval_df["IoU"].mean(),
            "recall": recall,
            "precision": precision,
            "F1": 2 * (precision * recall) / (precision + recall),
        }
    )


def compute_metrics_df(
    annot_gdf: gpd.GeoDataFrame,
    pred_gdf: gpd.GeoDataFrame,
    tile_dir: utils.PathDType,
    iou_threshold: float = 0.4,
) -> pd.DataFrame:
    """Compute IoU, recall and precision for the predictions with different patch sizes.

    Parameters
    ----------
    annot_gdf, pred_gdf : gpd.GeoDataFrame
        Annotations and predictions geo-data frames respectively.
    tile_dir : path-like
        Path to the directory containing the tiles.
    iou_threshold : numeric, optional
        Intersection over union threshold to consider a prediction a match.

    Returns
    -------
    metrics_df : pd.DataFrame
        Metrics data frame.

    """
    eval_gdf = pd.concat(
        [
            evaluate.evaluate_image_boxes(patch_gdf.reset_index(), annot_gdf, tile_dir)
            .dropna()
            .assign(**{"patch_size": patch_size})
            for patch_size, patch_gdf in pred_gdf.groupby("patch_size")
        ],
        ignore_index=True,
    )

    return eval_gdf.groupby("patch_size").apply(
        eval_metrics,
        pred_gdf,
        iou_threshold=iou_threshold,
        include_groups=False,
    )


def retrain_model(
    model: main.deepforest,
    img_dir: utils.PathDType,
    train_df: pd.DataFrame,
    *,
    test_df: pd.DataFrame | None = None,
    gpus: str | None = None,
    epochs: int = 2,
) -> main.deepforest:
    """Retrain the model on the annotated data."""

    def save_annot_df(annot_df, dst_filepath):
        """Update image paths and labels and save the annotated data frame."""
        annot_df.assign(
            **{
                "image_path": annot_df["image_path"].str.replace(".jpeg", ".tif"),
                # "label": annot_df["label"].str.capitalize(),
            }
        ).to_csv(dst_filepath)
        return dst_filepath

    # configure training
    if gpus is not None:
        model.config["gpus"] = gpus
    model.config["train"]["epochs"] = epochs
    model.config["train"]["root_dir"] = img_dir
    with tempfile.TemporaryDirectory() as tmp_dir:
        # save training data to a temporary file
        train_df_filepath = path.join(tmp_dir, "train.csv")
        save_annot_df(train_df, train_df_filepath)
        model.config["train"]["csv_file"] = train_df_filepath

        if test_df is not None:
            # save training data to a temporary file
            test_df_filepath = path.join(tmp_dir, "test.csv")
            save_annot_df(test_df, test_df_filepath)
            model.config["validation"]["root_dir"] = img_dir
            model.config["validation"]["csv_file"] = test_df_filepath

        model.create_trainer()
        # model.use_release()
        start_time = time.time()
        model.trainer.fit(model)
    print(f"--- Model retrained in {(time.time() - start_time):.2f} seconds ---")
    return model
