"""DeepForest utils."""

import tempfile
import time
from os import path

import numpy as np
import pandas as pd
import torch
from deepforest import main
from scipy import optimize
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from swiss_urban_trees import utils

DEFAULT_IOU_THRESHOLD = 0.4
DEFAULT_REC_SEP = 0.01
DEFAULT_MAX_DETECTION_THRESHOLD = 1000


def _torch_dict_from_df(df, label_dict):
    d = {}
    d["boxes"] = torch.tensor(
        df[["xmin", "ymin", "xmax", "ymax"]].astype(int).values,
    )
    if "score" in df.columns:
        d["scores"] = torch.tensor(df["score"].values)
    d["labels"] = torch.tensor(df["label"].map(label_dict).astype(int).values)
    return d


def _compute_torchmetrics(
    pred_df: pd.DataFrame,
    annot_df: pd.DataFrame,
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
    label_dict : dict
        Label dictionary mapping the integer ids to the class names.
    iou_thresholds : list of float
        IoU thresholds to use, passed to
        `torchmetrics.detection.mean_ap.MeanAveragePrecision`.
    rec_thresholds : list of float
        Recall thresholds to use, passed to
        `torchmetrics.detection.mean_ap.MeanAveragePrecision`.
    max_detection_thresholds : list of float
        Maximum detection thresholds to use, passed to
        `torchmetrics.detection.mean_ap.MeanAveragePrecision`. Must be a list of three
        items.

    Returns
    -------
    iou_dict : dict of torch.Tensor
        Dictionary with keys of the form (image, class) and values are a tensor of IoU
        values of shape (n, m) where n is the number of detections and m is the number
        of ground truth boxes for that image/class combination.
    recall : torch.Tensor
        Recall values for each class, as a tensor of shape K where K is the number of
        classes.
    precision : torch.Tensor
        Precision values for each recall threshold and class, as a tensor of shape (R,
        K) where R is the number of recall thresholds and K is the number of classes.

    """
    mean_ap_kwargs = dict(
        iou_type="bbox",
        iou_thresholds=iou_thresholds,
        rec_thresholds=rec_thresholds,
        max_detection_thresholds=max_detection_thresholds,
        extended_summary=True,
    )

    # mean_ap_dict = eval_metrics(
    #     pred_df,
    #     annot_df,
    #     label_dict,
    #     **mean_ap_kwargs,
    # )

    metric = MeanAveragePrecision(
        **mean_ap_kwargs,
    )

    metric.update(
        [
            _torch_dict_from_df(_pred_df, label_dict)
            for _, _pred_df in pred_df.groupby("image_path")
        ],
        [
            _torch_dict_from_df(_annot_df, label_dict)
            for _, _annot_df in annot_df.groupby("image_path")
        ],
    )
    mean_ap_dict = metric.compute()

    # ious
    # shape: dict with keys of the form (image, class) and values with the corresponding
    # ious for each detection (row) and ground truth (column)
    # TODO: support multi-class
    # TODO: support multi-image predictions and annotations
    # ious = mean_ap_dict["ious"]  # [(0, 0)].max(dim=0).values

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

    # return ious, recall, precision
    return mean_ap_dict["ious"], recall, precision


def evaluate(
    pred_df: pd.DataFrame,
    annot_df: pd.DataFrame,
    *,
    label_dict: dict | None = None,
    iou_thresholds: list[float] | None = None,
    rec_thresholds: list[float] | None = None,
    max_detection_thresholds: list[float] | None = None,
    iou_threshold: float | None = None,
    rec_sep: float | None = None,
    max_detection_threshold: int | None = None,
) -> dict:
    """Compute evaluation data frame.

    Parameters
    ----------
    pred_df, annot_df : pd.DataFrame
        Predictions and annotations data frames respectively.
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
        IoU threshold to use. Ignored if `iou_thresholds` is not None. Defaults to
        `DEFAULT_IOU_THRESHOLD`.
    rec_sep : float, optional
        Recall step to use. Ignored if `rec_thresholds` is not None. Defaults to
        `DEFAULT_REC_SEP`.
    max_detection_threshold : int, optional
        Maximum number of detected objects threshold to use. Ignored if
        `max_detection_thresholds` is not None. Defaults to
        `DEFAULT_MAX_DETECTION_THRESHOLD`.

    Returns
    -------
    eval_dict : dict
        Evaluation dictionary with the following keys:
        - "results": evaluation results data frame.
        - "recall": recall value.
        - "precision": precision value.

    """
    # prepare kwargs
    if label_dict is None:
        label_dict = {
            label: i
            for i, label in enumerate(set(pred_df["label"]).union(annot_df["label"]))
        }
    if iou_thresholds is None:
        if iou_threshold is None:
            iou_threshold = DEFAULT_IOU_THRESHOLD
        iou_thresholds = [iou_threshold]
    if rec_thresholds is None:
        if rec_sep is None:
            rec_sep = DEFAULT_REC_SEP
        rec_thresholds = np.arange(0, 1 + rec_sep, rec_sep).tolist()
    if max_detection_thresholds is None:
        if max_detection_threshold is None:
            max_detection_threshold = DEFAULT_MAX_DETECTION_THRESHOLD
        max_detection_thresholds = [0, 0, max_detection_threshold]

    # compute metrics using torchmetrics
    iou_dict, recall, precision = _compute_torchmetrics(
        pred_df,
        annot_df,
        label_dict,
        iou_thresholds,
        rec_thresholds,
        max_detection_thresholds,
    )

    # build a deepforest-like evaluation data frame
    # TODO: are keys in `ious` sorted based on preds or targets? does that depend on the
    # backend used (i.e., pycocotools or faster_coco_eval)?
    image_path_dict = {
        image_key: image_path
        for image_key, (image_path, _) in enumerate(annot_df.groupby("image_path"))
    }
    numeric_to_label_dict = {val: label for label, val in label_dict.items()}

    def process_image_class(image_key, class_val):
        iou_tensor = iou_dict[(image_key, class_val)]
        # check that `iou_tensor` is a two-dimensional tensor
        if iou_tensor != [] and len(iou_tensor.shape) == 2:
            # create cost matrix for assignment, with rows and columns respectively
            # representing predictions and ground truths and the cost being the area of
            # the intersection
            # ACHTUNG: since in most cases images are tiles (hence are rather small),
            # using the spatial index is slower
            # annot_sindex = annot_df.sindex
            # cost_arr = pred_df.geometry.apply(
            #     lambda pred_geom: annot_df.iloc[
            #         annot_sindex.intersection(pred_geom.bounds)
            #     ]
            #     .intersection(pred_geom)
            #     .area
            # ).fillna(0)
            # TODO: can we get the ground truth-prediction matches from torchmetrics?
            # we'd need to access the `coco_eval` variable within the
            # `MeanAveragePrecision.compute` method (see https://github.com/
            # Lightning-AI/torchmetrics/blob/master/src/torchmetrics/detection/
            # mean_ap.py#L522), but this would require modifying torchmetrics.
            # Additionally, how to access the matches would depend on the backend used
            # (i.e., pycocotools or faster_coco_eval).
            cost_arr = (
                pred_df[
                    (pred_df["image_path"] == image_path_dict[image_key])
                    & (pred_df["label"] == numeric_to_label_dict[class_val])
                ]
                .geometry.apply(
                    lambda pred_geom: annot_df[
                        (annot_df["image_path"] == image_path_dict[image_key])
                        & (annot_df["label"] == numeric_to_label_dict[class_val])
                    ]
                    .intersection(pred_geom)
                    .area
                )
                .values
            )

            row_ind, col_ind = optimize.linear_sum_assignment(
                cost_arr,
                maximize=True,
            )
            return pd.DataFrame(
                {
                    "prediction_id": row_ind,
                    "truth_id": col_ind,
                    "IoU": iou_tensor[row_ind, col_ind],
                },
            ).sort_values("truth_id", ascending=True)
        else:
            # when iou_dict[(image_key, class_val)] is [], aka, no predictions
            # except ValueError, IndexError:
            return None

    eval_df = pd.concat(
        [
            process_image_class(image_key, class_val)
            for image_key, class_val in iou_dict.keys()
        ],
        axis="rows",
        ignore_index=True,
    )

    # TODO: does this work even if we shuffle the data frames?
    eval_df["image_path"] = annot_df["image_path"]

    # TODO: are the columns below optional?
    # set true labels
    eval_df["true_label"] = annot_df["label"]  # .map(label_dict)
    # set the geometry
    eval_df["geometry"] = annot_df["geometry"]
    # set the score and predicted label
    # TODO: how to best manage the prediction_id-to-index of `pred_df` mapping?
    eval_df = eval_df.merge(
        pred_df[["score", "label"]].reset_index(drop=True),
        left_on="prediction_id",
        right_index=True,
    )
    eval_df = eval_df.rename(columns={"label": "predicted_label"})
    # set whether it is a match
    # eval_df["match"] = eval_df["IoU"] >= iou_threshold

    return {
        "results": eval_df,
        "recall": recall,
        "precision": precision,
    }


def multiscale_eval_df(
    pred_df: pd.DataFrame,
    annot_df: pd.DataFrame,
    tile_dir: utils.PathDType,
    *,
    label_dict: dict = None,
    iou_thresholds: list[float] = None,
    rec_thresholds: list[float] = None,
    max_detection_thresholds: list[float] = None,
    iou_threshold: float | None = None,
    rec_sep: float | None = None,
    max_detection_threshold: int | None = None,
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
    # TODO: multi-class version of this method
    pred_df = pred_df.assign(label="Tree")
    annot_df = annot_df.assign(label="Tree")
    label_dict = {"Tree": 1}

    def _compute_metrics(_pred_df):
        eval_dict = evaluate(
            _pred_df,
            annot_df,
            label_dict=label_dict,
            iou_thresholds=iou_thresholds,
            rec_thresholds=rec_thresholds,
            max_detection_thresholds=max_detection_thresholds,
            iou_threshold=iou_threshold,
            rec_sep=rec_sep,
            max_detection_threshold=max_detection_threshold,
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
