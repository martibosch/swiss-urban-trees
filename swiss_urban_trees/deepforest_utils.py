"""DeepForest utils."""

import contextlib
import glob
import tempfile
import time
from collections.abc import Sequence
from os import path

import geopandas as gpd
import pandas as pd
import rasterio as rio
from sklearn import model_selection

from deepforest import evaluate as evaluate_iou
from deepforest import main, utilities
from swiss_urban_trees import utils

DEFAULT_IOU_THRESHOLD = 0.4


def get_annot_gdf(
    *,
    annot_filepaths: Sequence[utils.PathDType] | None = None,
    annot_dir: utils.PathDType | None = None,
    img_dir: utils.PathDType | None = None,
    annot_ext: str | None = "gpkg",
    img_ext: str = "tiff",
) -> gpd.GeoDataFrame:
    """Get the annotations geo-data frame from the given file paths or directory.

    Parameters
    ----------
    annot_filepaths : list-like of path-like, optional
        Paths to the annotation files. If not provided, `annot_dir` must be provided.
    annot_dir : path-like, optional
        Directory containing the annotation files. If not provided, `annot_filepaths`
        must be provided. Ignored if `annot_filepaths` is provided.
    img_dir : path-like, optional
        Directory containing the images. If not provided, it will be inferred from the
        first annotation file path.
    annot_ext, img_ext : str, optional
        Extension of the annotation and image files respectively. Defaults to "gpkg" and
        "tiff" respectively.

    Returns
    -------
    annot_gdf : gpd.GeoDataFrame
        Annotations as geo-data frame with bounding boxes, labels and image paths.

    """
    if annot_filepaths is None:
        if annot_dir is None:
            raise ValueError(
                "Either `annot_filepaths` or `annot_dir` must be provided."
            )
        annot_filepaths = glob.glob(path.join(annot_dir, f"*.{annot_ext}"))
    if img_dir is None:
        img_dir = path.dirname(annot_filepaths[0])
    annot_gdfs = []
    for annot_filepath in annot_filepaths:
        img_filename = path.basename(annot_filepath).replace(annot_ext, img_ext)
        with rio.open(path.join(img_dir, img_filename)) as src:
            # hide deepforest's print statements on reading annotations
            with contextlib.redirect_stdout(None):
                annot_gdf = utilities.shapefile_to_annotations(
                    gpd.read_file(annot_filepath).assign(
                        **{"label": "Tree", "image_path": img_filename}
                    ),
                    root_dir=img_dir,
                ).clip([0, 0, src.width, src.height])
        annot_gdfs.append(
            annot_gdf.assign(
                **annot_gdf.bounds.rename(columns=lambda col: col[-1] + col[:-1])
            )
        )
    return pd.concat(annot_gdfs, ignore_index=True)


def train_test_split(
    annot_df: pd.DataFrame | gpd.GeoDataFrame,
    **train_test_split_kwargs: utils.KwargsDType,
) -> tuple[pd.DataFrame | gpd.GeoDataFrame, pd.DataFrame | gpd.GeoDataFrame]:
    """Split the annotations into training and testing sets.

    Parameters
    ----------
    annot_df : pd.DataFrame or gpd.GeoDataFrame
        Annotation data, as pandas or geopandas data frame with multi-class (under the
        "label" column) bounding box annotations.
    train_test_split_kwargs : keyword arguments
        Additional keyword arguments to pass to
        `sklearn.model_selection.train_test_split`, e.g., `train_size`, `random_state`.

    Returns
    -------
    train_df, test_df : tuple of pd.DataFrame or gpd.GeoDataFrame
        Training and testing annotations as geo-data frames.

    """
    img_filenames = pd.Series(annot_df["image_path"].unique())

    train_imgs, test_imgs = model_selection.train_test_split(
        img_filenames, **train_test_split_kwargs
    )
    return annot_df[annot_df["image_path"].isin(train_imgs)], annot_df[
        annot_df["image_path"].isin(test_imgs)
    ]


def multiscale_eval_df(
    predictions: pd.DataFrame,
    ground_df: pd.DataFrame,
    tile_dir: utils.PathDType,
    *,
    label_dict: dict = None,
    iou_threshold: float | None = None,
    compute_f1: bool = False,
) -> pd.DataFrame:
    """Compute IoU, recall and precision for multiscale (patch size) predictions.

    Parameters
    ----------
    predictions, ground_df : pd.DataFrame
        Predictions and annotations data frames respectively. The prediction data frame
        must have a `patch_size` column.
    tile_dir : path-like
        Path to the directory containing the tiles.
    label_dict : dict, optional
        Label dictionary mapping the class names to the integer ids. If None, it will be
        inferred from the `label` column of `pred_df` and `annot_df`.
    iou_threshold : float, optional
        IoU threshold to use.
    compute_f1 : bool, optional
        Whether to compute the F1 score. Defaults to False.

    Returns
    -------
    eval_df : pd.DataFrame
        Evaluation data frame with the patch size, IoU, recall, precision and optionally
        F1 score columns.

    """
    if iou_threshold is None:
        iou_threshold = DEFAULT_IOU_THRESHOLD
    if label_dict is None:
        label_dict = {
            label: i
            for i, label in enumerate(
                set(predictions["label"]).union(ground_df["label"])
            )
        }
    # numeric_to_label_dict = {val: label for label, val in label_dict.items()}

    def _compute_metrics(_predictions):
        eval_dict = evaluate_iou.__evaluate_wrapper__(
            predictions=_predictions,
            ground_df=ground_df,
            iou_threshold=iou_threshold,
            label_dict=label_dict,
        )

        return [
            eval_dict["results"]["IoU"].mean(),
            eval_dict["box_recall"],
            eval_dict["box_precision"],
        ], eval_dict["class_recall"]

    box_evals = []
    class_eval_dfs = []
    for patch_size, _predictions in predictions.groupby("patch_size"):
        box_eval, class_eval_df = _compute_metrics(_predictions)
        box_evals.append(box_eval)
        class_eval_dfs.append(class_eval_df.assign(patch_size=patch_size))

    # eval_df = predictions.groupby("patch_size").apply(_compute_metrics)
    box_eval_df = pd.DataFrame(
        box_evals,
        columns=["IoU", "box_recall", "box_precision"],
    ).assign(
        **{
            "patch_size": predictions["patch_size"].unique(),
        }
    )
    class_eval_df = pd.concat(class_eval_dfs, ignore_index=True)

    def _compute_f1(eval_df, prefix):
        return (
            2
            * eval_df[f"{prefix}precision"]
            * eval_df[f"{prefix}recall"]
            / (eval_df[f"{prefix}precision"] + eval_df[f"{prefix}recall"])
        ).fillna(0)

    if compute_f1:
        box_eval_df["F1"] = _compute_f1(box_eval_df, "box_")
    if len(label_dict) > 1:
        if compute_f1:
            class_eval_df["F1"] = _compute_f1(class_eval_df, "")
        return box_eval_df, class_eval_df
    else:
        return box_eval_df


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
        """Save the annotated data frame."""
        # we are just using a function to DRY any eventual required preprocessing
        annot_df.to_csv(dst_filepath)
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

        print(model.config)
        model.create_trainer()
        # model.use_release()
        start_time = time.time()
        model.trainer.fit(model)
    print(f"--- Model retrained in {(time.time() - start_time):.2f} seconds ---")
    return model
