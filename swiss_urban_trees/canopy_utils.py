"""Canopy utils."""

import tempfile
from os import path

import pdal
import rasterio as rio


def rasterize_lidar(lidar_filepath, dst_filepath, lidar_values, **kwargs):
    """Rasterize LiDAR file."""
    pipeline = (
        pdal.Reader(lidar_filepath)
        | pdal.Filter.expression(
            expression=" || ".join(
                [f"Classification == {value}" for value in lidar_values]
            )
        )
        | pdal.Writer.gdal(
            filename=dst_filepath,
            # resolution=dst_res,
            # output_type="count",
            # data_type="int32",
            # nodata=0,
            # default_srs=stac_utils.SWISSALTI3D_CRS,
            **kwargs,
        )
    )
    _ = pipeline.execute()
    return dst_filepath


def lidar_to_dsm(lidar_filepath, dst_filepath, *, lidar_values=None, **kwargs):
    """Convert LiDAR file to DSM raster."""
    pipeline = pdal.Reader(lidar_filepath) | pdal.Filter.returns(groups="only")
    if lidar_values is not None:
        pipeline = pipeline | pdal.Filter.expression(
            expression=" || ".join(
                [f"Classification == {value}" for value in lidar_values]
            )
        )
    pipeline = pipeline | pdal.Writer.gdal(
        filename=dst_filepath,
        **kwargs,
    )
    _ = pipeline.execute()
    return dst_filepath


def lidar_to_dtm(
    lidar_filepath,
    dst_filepath,
    *,
    elm_kwargs=None,
    outlier_kwargs=None,
    smrf_kwargs=None,
    gdal_kwargs=None,
):
    """Convert LiDAR file to DTM raster."""
    if elm_kwargs is None:
        elm_kwargs = {}
    if outlier_kwargs is None:
        outlier_kwargs = {}
    if smrf_kwargs is None:
        smrf_kwargs = {}
    if gdal_kwargs is None:
        gdal_kwargs = {}
    pipeline = (
        pdal.Reader(lidar_filepath)
        | pdal.Filter.assign(assignment="Classification[:]=0")
        | pdal.Filter.elm(**elm_kwargs)
        | pdal.Filter.outlier(**outlier_kwargs)
        | pdal.Filter.smrf(ignore="Classification[7:7]", **smrf_kwargs)
        | pdal.Filter.range(limits="Classification[2:2]")
        | pdal.Writer.gdal(filename=dst_filepath, **gdal_kwargs)
    )
    _ = pipeline.execute()
    return dst_filepath


def get_chm(
    lidar_filepath,
    dst_filepath,
    *,
    dtm_filepath=None,
    lidar_values=None,
    gdal_kwargs=None,
    elm_kwargs=None,
    outlier_kwargs=None,
    smrf_kwargs=None,
):
    """Get Canopy Height Model from raw LiDAR and DEM raster.

    The extents are assumed to be aligned.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        veg_dsm_filepath = path.join(tmp_dir, "dsm.tif")
        # with rio.open(dem_filepath) as dem_src:
        # _kwargs.update(
        #     resolution=dem_src.res[0],
        #     default_srs=dem_src.crs,
        # )
        _ = lidar_to_dsm(
            lidar_filepath, veg_dsm_filepath, lidar_values=lidar_values, **gdal_kwargs
        )
        if dtm_filepath is None:
            dtm_filepath = path.join(tmp_dir, "dtm.tif")
            _ = lidar_to_dtm(
                lidar_filepath,
                dtm_filepath,
                elm_kwargs=elm_kwargs,
                outlier_kwargs=outlier_kwargs,
                smrf_kwargs=smrf_kwargs,
                gdal_kwargs=gdal_kwargs,
            )
        with rio.open(veg_dsm_filepath) as veg_dsm_src:
            with rio.open(dtm_filepath) as dtm_src:
                with rio.open(dst_filepath, "w", **veg_dsm_src.meta) as dst:
                    chm_arr = veg_dsm_src.read(1) - dtm_src.read(1)
                    chm_arr[chm_arr < 0] = 0
                    dst.write(chm_arr, 1)
    return dst_filepath
