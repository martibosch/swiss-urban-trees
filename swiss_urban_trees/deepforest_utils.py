"""DeepForest utils."""

import tempfile
import time
from os import path

import numpy as np
import pandas as pd

from deepforest import evaluate as evaluate_iou
from deepforest import main
from swiss_urban_trees import utils


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
        Label dictionary mapping the integer ids to the class names. If None, it will be
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
    if label_dict is None:
        label_dict = {
            label: i
            for i, label in enumerate(
                set(predictions["label"]).union(ground_df["label"])
            )
        }
    numeric_to_label_dict = {val: label for label, val in label_dict.items()}

    def _compute_metrics(_predictions):
        eval_dict = evaluate_iou.__evaluate_wrapper__(
            predictions=_predictions,
            ground_df=ground_df,
            iou_threshold=iou_threshold,
            numeric_to_label_dict=numeric_to_label_dict,
        )
        precision = eval_dict["precision"]
        return pd.Series(
            [
                eval_dict["results"]["IoU"].mean(),
                # TODO: multi-class version of this method
                eval_dict["recall"].item(),
                precision[np.nonzero(precision)[-1][0]].item(),
            ]
        )

    eval_df = predictions.groupby("patch_size").apply(_compute_metrics)
    eval_df.columns = ["IoU", "recall", "precision"]
    if compute_f1:
        eval_df["F1"] = (
            2
            * eval_df["precision"]
            * eval_df["recall"]
            / (eval_df["precision"] + eval_df["recall"])
        )
    return eval_df.reset_index()


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

        model.create_trainer()
        # model.use_release()
        start_time = time.time()
        model.trainer.fit(model)
    print(f"--- Model retrained in {(time.time() - start_time):.2f} seconds ---")
    return model
