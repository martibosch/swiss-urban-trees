"""Label Studio utilities."""

from collections.abc import Iterable
from os import path

import numpy as np
import pandas as pd
from label_studio_sdk import types
from label_studio_sdk.client import LabelStudio


def get_project_by_title(ls: LabelStudio, project_title: str) -> types.Project | None:
    """Get a project by its title."""
    for project in ls.projects.list():
        if project.title == project_title:
            return ls.projects.get(project.id)
    return None


def get_annot_df_from_task(
    task: types.Task,
    *,
    negative_annot: bool = False,
    negative_labels: str | Iterable[str] = "Tree",
) -> pd.DataFrame:
    """Get deepforest-compatible annotations from a Label Studio task."""
    annot_df = pd.json_normalize(task.annotations, record_path="result").rename(
        columns=lambda col: col.replace("value.", "")
    )  # [["x", "y", "width", "height", "rectanglelabels"]]
    # annot_df["rectanglelabels"] = annot_df["rectanglelabels"].str[0]
    if annot_df.empty:
        if negative_annot:
            # maybe allow getting the negative labels from the project
            # project.parsed_label_config["label"]["labels"]
            if isinstance(negative_labels, str):
                negative_labels = [negative_labels]
            annot_df = pd.DataFrame(
                np.zeros((len(negative_labels), 4), dtype=int),
                columns=["xmin", "ymin", "xmax", "ymax"],
            ).assign(
                **{
                    "label": negative_labels,
                }
            )
        else:
            return annot_df
    else:
        scale_x = annot_df["original_width"].iloc[0] / 100
        scale_y = annot_df["original_height"].iloc[0] / 100

        def _process_ser(ser, scale):
            return (ser * scale).astype(int)

        assign_dict = {
            "xmin": _process_ser(annot_df["x"], scale_x),
            "ymin": _process_ser(annot_df["y"], scale_y),
            "xmax": _process_ser(annot_df["x"] + annot_df["width"], scale_x),
            "ymax": _process_ser(annot_df["y"] + annot_df["height"], scale_y),
            "label": annot_df["rectanglelabels"].str[0],
        }
        annot_df = annot_df.assign(**assign_dict)[assign_dict.keys()]
    return annot_df.assign(**{"image_path": path.basename(task.data["image"])})


def get_annot_df_from_project(
    ls_client: LabelStudio, project: str | types.Project
) -> pd.DataFrame:
    """Get deepforest-compatible annotation data frame from a Label Studio project.

    Parameters
    ----------
    ls_client : LabelStudio
        The Label Studio client.
    project : str or label studio project
        The project to get annotations from. If a string, it is assumed to be the
        project title.

    Returns
    -------
    annot_df : pd.DataFrame
        The annotations in a data frame.

    """
    if isinstance(project, str):
        project = get_project_by_title(ls_client, project)
    return pd.concat(
        get_annot_df_from_task(task)
        for task in ls_client.tasks.list(project=project.id)
    )
