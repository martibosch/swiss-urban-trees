"""DeepForest utils."""

import tempfile
import time
from os import path

import numpy as np
import pandas as pd
import torch
from deepforest import main
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from swiss_urban_trees import utils


def _torch_dict_from_df(df, label_dict):
    d = {}
    d["boxes"] = torch.tensor(
        df[["xmin", "ymin", "xmax", "ymax"]].astype(int).values,
    )
    if "score" in df.columns:
        d["scores"] = torch.tensor(df["score"].values)
    d["labels"] = torch.tensor(df["label"].map(label_dict).astype(int).values)
    return d


def eval_metrics(
    pred_df: pd.DataFrame,
    annot_df: pd.DataFrame,
    label_dict: dict,
    **mean_ap_kwargs: utils.KwargsDType,
) -> dict:
    """Compute IoU, recall, precision and F1 score for a given patch size.

    Parameters
    ----------
    pred_df, annot_df : pd.DataFrame
        Predictions and annotations data frames respectively.
    label_dict : dict
        Label dictionary mapping the class names to the numeric ids.
    mean_ap_kwargs : mapping, optional
        Keyword arguments to pass to
        `torchmetrics.detection.mean_ap.MeanAveragePrecision`.

    Returns
    -------
    results : dict
        Dictionary with the torchmetrics.detection.mean_ap.MeanAveragePrecision results.

    """
    # n_true_positives = (eval_df["IoU"] > iou_threshold).sum()
    # recall = n_true_positives / eval_df.shape[0]
    # precision = (
    #     n_true_positives / pred_df[pred_df["patch_size"] == eval_df.name].shape[0]
    # )
    metric = MeanAveragePrecision(
        **mean_ap_kwargs,
    )

    metric.update(
        [
            _torch_dict_from_df(pred_df, label_dict),
        ],
        [
            _torch_dict_from_df(annot_df, label_dict),
        ],
    )
    return metric.compute()


def compute_metrics(
    pred_df: pd.DataFrame,
    annot_df: pd.DataFrame,
    tile_dir: utils.PathDType,
    label_dict: dict,
    iou_thresholds: list[float],
    rec_thresholds: list[float],
    max_detection_thresholds: list[float],
) -> tuple:
    """Compute IoU, recall and precision.

    Parameters
    ----------
    pred_df, annot_df : pd.DataFrame
        Predictions and annotations data frames respectively.
    tile_dir : path-like
        Path to the directory containing the tiles.
    label_dict : dict, optional
        Label dictionary mapping the integer ids to the class names. If None, it will be
        inferred from the `label` column of `pred_df` and `annot_df`.
    iou_thresholds, rec_thresholds, max_detection_thresholds : list of float
        IoU, recall and maximum detection thresholds thresholds to use, passed to
        `torchmetrics.detection.mean_ap.MeanAveragePrecision`.

    Returns
    -------
    ious, recall, precision : tuple of torch.Tensor

    """
    mean_ap_kwargs = dict(
        iou_type="bbox",
        iou_thresholds=iou_thresholds,
        rec_thresholds=rec_thresholds,
        max_detection_thresholds=max_detection_thresholds,
        extended_summary=True,
    )

    mean_ap_dict = eval_metrics(
        pred_df,
        annot_df,
        label_dict,
        **mean_ap_kwargs,
    )

    # ious
    # shape: dict with keys of the form (image, class) and values with the corresponding
    # ious for each detection (row) and ground truth (column)
    # TODO: support multi-class
    # TODO: support multi-image predictions and annotations
    ious = mean_ap_dict["ious"][(0, 0)].max(dim=0).values

    # recall
    # shape: n_iou_thresholds, n_classes, n_areas, n_max_detections
    # selection:
    # - one iou_threshold
    # - all classes
    # - first area, i.e., 'all' https://github.com/cocodataset/cocoapi/blob/master/
    #   PythonAPI/pycocotools/cocoeval.py#L509
    # - last max_detection_threshold
    recall = mean_ap_dict["recall"][0, :, 0, -1]

    # precision
    # shape: n_iou_thresholds, n_recall_thresholds, n_classes, n_areas, n_max_detections
    # selection:
    # - one iou_threshold
    # - all recall_thresholds
    # - all classes
    # - first area, i.e., 'all'
    # - last max_detection_threshold
    # idea: emulate deepforest and return recall_threshold closest to the actual recall?
    # note that we'd need to ensure that rec_thresholds is a numpy array
    # np.argmin(
    #     np.abs(np.array(rec_thresholds)[:, np.newaxis] - recall.numpy()), axis=0
    # ),
    # ACHTUNG: we could use np.searchsorted, which would be quicker than argmin, but
    # performance should not be an issue here
    # see https://stackoverflow.com/questions/44526121/
    # finding-closest-values-in-two-numpy-arrays
    precision = mean_ap_dict["precision"][
        0,
        :,
        :,
        0,
        -1,
    ]

    return ious, recall, precision


def multiscale_eval_df(
    pred_df: pd.DataFrame,
    annot_df: pd.DataFrame,
    tile_dir: utils.PathDType,
    *,
    label_dict: dict = None,
    iou_thresholds: list[float] = None,
    rec_thresholds: list[float] = None,
    max_detection_thresholds: list[float] = None,
    iou_threshold: float = 0.4,
    rec_sep: float = 0.01,
    max_detection_threshold: int = 1000,
    compute_f1: bool = False,
) -> pd.DataFrame:
    """Compute IoU, recall and precision for multiscale (patch size) predictions.

    Parameters
    ----------
    pred_df, annot_df : pd.DataFrame
        Predictions and annotations data frames respectively. The prediction data frame
        must have a `patch_size` column.
    tile_dir : path-like
        Path to the directory containing the tiles.
    label_dict : dict, optional
        Label dictionary mapping the integer ids to the class names. If None, it will be
        inferred from the `label` column of `pred_df` and `annot_df`.
    iou_thresholds : list of float, optional
        IoU thresholds to use, passed to
        `torchmetrics.detection.mean_ap.MeanAveragePrecision`. If None, defaults to
        a list with `iou_threshold` as the only element.
    rec_thresholds : list of float, optional
        Recall thresholds to use, passed to
        `torchmetrics.detection.mean_ap.MeanAveragePrecision`. If None, defaults to
        a list with values from 0 to 1 (both included) with a step of `rec_sep`.
    max_detection_thresholds : list of float, optional
        Maximum detection thresholds to use, passed to
        `torchmetrics.detection.mean_ap.MeanAveragePrecision`. Must be a list of three
        items. If None, defaults to [0, 0, max_detection_threshold].
    iou_threshold : float, optional
        IoU threshold to use. Ignored if `iou_thresholds` is not None.
    rec_sep : float, optional
        Recall step to use. Ignored if `rec_thresholds` is not None.
    max_detection_threshold : int, optional
        Maximum number of detected objects threshold to use. Ignored if
        `max_detection_thresholds` is not None.
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
            for i, label in enumerate(set(pred_df["label"]).union(annot_df["label"]))
        }
    # TODO: support multi-image predictions and annotations
    if iou_thresholds is None:
        iou_thresholds = [iou_threshold]
    if rec_thresholds is None:
        rec_thresholds = np.arange(0, 1 + rec_sep, rec_sep).tolist()
    if max_detection_thresholds is None:
        max_detection_thresholds = [0, 0, max_detection_threshold]

    def _compute_metrics(_pred_df):
        ious, recall, precision = compute_metrics(
            _pred_df,
            annot_df,
            tile_dir,
            label_dict,
            iou_thresholds,
            rec_thresholds,
            max_detection_thresholds,
        )
        return pd.Series(
            [
                ious.mean().item(),
                recall.item(),
                precision[np.nonzero(precision)[-1][0]].item(),
            ]
        )

    eval_df = pred_df.groupby("patch_size").apply(_compute_metrics)
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
