from os import path

PROJECT_NAME = "swiss-urban-trees"
CODE_DIR = "swiss_urban_trees"
PYTHON_VERSION = "3.12"

NOTEBOOKS_DIR = "notebooks"
NOTEBOOKS_OUTPUT_DIR = path.join(NOTEBOOKS_DIR, "output")

DATA_DIR = "data"
DATA_RAW_DIR = path.join(DATA_DIR, "raw")
DATA_INTERIM_DIR = path.join(DATA_DIR, "interim")
DATA_PROCESSED_DIR = path.join(DATA_DIR, "processed")

MODELS_DIR = "models"


# 0. conda/mamba environment -----------------------------------------------------------
rule create_environment:
    shell:
        "mamba env create -f environment.yml"


rule register_ipykernel:
    shell:
        "python -m ipykernel install --user --name {PROJECT_NAME} --display-name"
        " 'Python ({PROJECT_NAME})'"


# 1. SITG ------------------------------------------------------------------------------
# 1.1 download SITG tiles
# we will use SITG aerial imagery istead of SWISSIMAGE because there is a better
# temporal match with SWISSSURFACE3D data (both from spring/summer 2019). Instead, the
# summer SWISSIMAGE in Geneva is from 2017 or 2023, see
# https://www.swisstopo.admin.ch/fr/acquisition-images-aeriennes
SITG_WMS_URL = (
    "https://raster.sitg.ge.ch/arcgis/services/ORTHOPHOTOS_COLLECTION/MapServer/"
    "WMSServer?request=GetCapabilities&service=WMS"
)
SITG_LAYER_NAME = "22"  # orthophoto may 2019

SITG_AOI = "Geneva, Switzerland"
SITG_TILES_DIR = path.join(DATA_RAW_DIR, "sitg-orthophoto-2019-tiles")
SITG_WMS_DOWNLOAD_RES = 0.1  # in meters
SITG_WMS_DOWNLOAD_TILE_SIZE = 2500  # in pixels
SITG_WMS_DOWNLOAD_FORMAT = "image/jpeg"


rule download_sitg_tiles:
    input:
        notebook=path.join(NOTEBOOKS_DIR, "wms-download.ipynb"),
    output:
        tiles=path.join(SITG_TILES_DIR, "tiles.gpkg"),
        notebook=path.join(
            NOTEBOOKS_OUTPUT_DIR,
            "wms-download-"
            f"{SITG_LAYER_NAME}-{SITG_AOI.replace(', ' , '-').lower()}.ipynb",
        ),
    shell:
        "papermill {input.notebook} {output.notebook}"
        " -p wms_url '{SITG_WMS_URL}'"
        " -p layer_name {SITG_LAYER_NAME}"
        " -p res {SITG_WMS_DOWNLOAD_RES}"
        " -p tile_size {SITG_WMS_DOWNLOAD_TILE_SIZE}"
        " -p format {SITG_WMS_DOWNLOAD_FORMAT}"
        " -p aoi '{SITG_AOI}'"
        " -p dst_dir {SITG_TILES_DIR}"
        " -p dst_filepath {output.tiles}"


# 1.2 download ICA tree inventory
ICA_WFS_URL = (
    "https://app2.ge.ch/tergeoservices/services/Hosted/SIPV_ICA_ARBRE_ISOLE/MapServer/"
    "WFSServer"
)


rule download_ica_trees:
    input:
        notebook=path.join(NOTEBOOKS_DIR, "wfs-download.ipynb"),
    output:
        trees=path.join(DATA_RAW_DIR, "ica-trees.gpkg"),
        notebook=path.join(NOTEBOOKS_OUTPUT_DIR, "wfs-download-ica-trees.ipynb"),
    shell:
        "papermill {input.notebook} {output.notebook}"
        " -p wfs_url '{ICA_WFS_URL}'"
        " -p aoi '{SITG_AOI}'"
        " -p dst_filepath {output.trees}"


# 1.3 train/test split
TRAIN_TEST_SPLIT_IPYNB_FILENAME = "train-test-split.ipynb"
N_COMPONENTS = 24
N_CLUSTERS = 4


rule train_test_split:
    input:
        tiles=rules.download_sitg_tiles.output.tiles,
        notebook=path.join(NOTEBOOKS_DIR, TRAIN_TEST_SPLIT_IPYNB_FILENAME),
    output:
        split=path.join(SITG_TILES_DIR, "split.csv"),
        notebook=path.join(NOTEBOOKS_OUTPUT_DIR, TRAIN_TEST_SPLIT_IPYNB_FILENAME),
    shell:
        "papermill {input.notebook} {output.notebook}"
        " -p tile_gdf_filepath {input.tiles}"
        " -p tile_dir {SITG_TILES_DIR}"
        " -p dst_filepath {output.split}"
        " -p n_components {N_COMPONENTS}"
        " -p n_clusters {N_CLUSTERS}"


# 1.4 label studio init
LABEL_STUDIO_PROJECT_TITLE = "sitg-orhtophoto-2019"
LABEL_STUDIO_LOCAL_DIR = path.join(DATA_RAW_DIR, "label-studio")
LABEL_STUDIO_INIT_IPYNB_FILENAME = "label-studio-init.ipynb"
LABEL_STUDIO_STORAGE_DIR = path.join(LABEL_STUDIO_LOCAL_DIR, LABEL_STUDIO_PROJECT_TITLE)


rule label_studio_init:
    input:
        split=rules.train_test_split.output.split,
        label_config=path.join(DATA_RAW_DIR, "label-config.xml"),
        notebook=path.join(NOTEBOOKS_DIR, LABEL_STUDIO_INIT_IPYNB_FILENAME),
    output:
        task_ids=path.join(SITG_TILES_DIR, "ls-init-task-ids.txt"),
        notebook=path.join(NOTEBOOKS_OUTPUT_DIR, LABEL_STUDIO_INIT_IPYNB_FILENAME),
    shell:
        "papermill {input.notebook} {output.notebook}"
        " -p split_filepath {input.split}"
        " -p tile_dir {SITG_TILES_DIR}"
        " -p project_title {LABEL_STUDIO_PROJECT_TITLE}"
        " -p label_config_filepath {input.label_config}"
        " -p storage_dir {LABEL_STUDIO_STORAGE_DIR}"
        " -p dst_filepath {output.task_ids}"


# 1.5 label studio export
LABEL_STUDIO_EXPORT_IPYNB_FILENAME = "label-studio-export.ipynb"
LABEL_STUDIO_EXPORT_DIR = path.join(SITG_TILES_DIR, "annot.csv")


rule label_studio_export:
    input:
        # task_ids=rules.label_studio_init.output.task_ids,
        notebook=path.join(NOTEBOOKS_DIR, LABEL_STUDIO_EXPORT_IPYNB_FILENAME),
    output:
        annot=path.join(SITG_TILES_DIR, "annot.csv"),
        notebook=path.join(NOTEBOOKS_OUTPUT_DIR, LABEL_STUDIO_EXPORT_IPYNB_FILENAME),
    shell:
        "papermill {input.notebook} {output.notebook}"

        " -p project_title {LABEL_STUDIO_PROJECT_TITLE}"
        " -p dst_filepath {output.annot}"
        # " -p task_ids_filepath {input.task_ids}"


# 1.6 train/fine-tune a model to predict tree crowns
TRAIN_CROWN_IPYNB_FILENAME = "train-crown.ipynb"


rule train_crown:
    input:
        annot=rules.label_studio_export.output.annot,
        notebook=path.join(NOTEBOOKS_DIR, TRAIN_CROWN_IPYNB_FILENAME),
    output:
        model=path.join(MODELS_DIR, "fine-tuned.pl"),
        notebook=path.join(NOTEBOOKS_OUTPUT_DIR, TRAIN_CROWN_IPYNB_FILENAME),
    shell:
        "papermill {input.notebook} {output.notebook}"
        " -p annot_filepath {input.annot}"
        " -p tile_dir {SITG_TILES_DIR}"
        " -p dst_filepath {output.model}"
