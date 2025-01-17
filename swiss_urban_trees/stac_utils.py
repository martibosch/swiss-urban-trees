"""STAC utils."""

import warnings

import geopandas as gpd
import numpy as np
import pystac_client
import rasterio as rio
import tqdm
import xarray as xr
from rasterio import mask, transform
from shapely import geometry

CLIENT_URL = "https://data.geo.admin.ch/api/stac/v0.9"
# CLIENT_CRS = "EPSG:4326"  # CRS used by the client
CLIENT_CRS = "OGC:CRS84"
CH_CRS = "EPSG:2056"

SWISSALTI3D_COLLECTION_ID = "ch.swisstopo.swissalti3d"
SWISSALTI3D_CRS = "EPSG:2056"
SWISSALTI3D_RES = 0.5
SWISSALTI3D_NODATA = -9999
SWISSIMAGE10_COLLECTION_ID = "ch.swisstopo.swissimage-dop10"
SWISSIMAGE10_CRS = "EPSG:2056"
SWISSIMAGE10_RES = 0.1
SWISSIMAGE10_NODATA = 0
SWISSSURFACE3D_RASTER_COLLECTION_ID = "ch.swisstopo.swisssurface3d-raster"
SWISSSURFACE3D_RASTER_CRS = "EPSG:2056"
SWISSSURFACE3D_COLLECTION_ID = "ch.swisstopo.swisssurface3d"
SWISSSURFACE3D_CRS = "EPSG:2056"

# TODO: get CRS and resolution from collection's metadata, i.e.:
# `"summaries":{"proj:epsg":[2056],"eo:gsd":[2.0,0.1]}`
COLLECTION_CRS_DICT = {
    SWISSSURFACE3D_RASTER_COLLECTION_ID: SWISSSURFACE3D_RASTER_CRS,
    SWISSSURFACE3D_COLLECTION_ID: SWISSSURFACE3D_CRS,
    SWISSALTI3D_COLLECTION_ID: SWISSALTI3D_CRS,
    SWISSIMAGE10_COLLECTION_ID: SWISSIMAGE10_CRS,
}


class SwissTopoClient:
    """swisstopo client."""

    def __init__(self):
        """Initialize a swisstopo client."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            client = pystac_client.Client.open(CLIENT_URL)
        client.add_conforms_to("ITEM_SEARCH")
        client.add_conforms_to("COLLECTIONS")
        self._client = client

    def gdf_from_collection(
        self,
        collection_id,
        *,
        extent_geom=None,
        datetime="2019/2019",
        extension=".tif",
        collection_extents_crs=None,
    ):
        """Get geo-data frame of tiles of a collection."""

        def _get_hrefs(item):
            return [
                href.href
                for href in item.assets.values()
                if href.href.endswith(extension)
            ]

        if collection_extents_crs is None:
            collection_extents_crs = self._client.get_collection(
                collection_id
            ).extra_fields["crs"][0]
        search = self._client.search(
            collections=[collection_id], intersects=extent_geom, datetime=datetime
        )

        # return pd.DataFrame(
        #     [(get_tif(item), str(item.bbox)) for item in list(search.items())],
        #     columns=[f"{collection}_filepath", "geometry"],
        # )
        return gpd.GeoDataFrame(
            [
                (_get_hrefs(item), geometry.box(*item.bbox))
                for item in list(search.items())
            ],
            crs=collection_extents_crs,
            columns=[collection_id, "geometry"],
        ).explode(collection_id)

    def read_extent_img(self, collection_id, extent_geom, **kwargs):
        """Read collection image for a given extent only."""
        collection_imgs_gdf = self.gdf_from_collection(
            collection_id, extent_geom=extent_geom, **kwargs
        )
        # TODO: extent to more than one intersecting img
        # TODO: improve way to filter to keep only images at 10 cm resolution
        collection_img_filepath = collection_imgs_gdf[
            collection_imgs_gdf[collection_id].str.contains("_0.1_2056.tif")
        ][collection_id].iloc[0]
        collection_img_bbox_geom = (
            gpd.GeoSeries(extent_geom, crs=CLIENT_CRS)
            .to_crs(COLLECTION_CRS_DICT[collection_id])
            .iloc[0]
        )
        with rio.open(collection_img_filepath) as src:
            out_img, out_transform = mask.mask(
                src, [collection_img_bbox_geom], crop=True
            )
            out_meta = src.meta.copy()
        out_meta.update(
            transform=out_transform, width=out_img.shape[2], height=out_img.shape[1]
        )
        return out_img, out_meta

    def yodo_process(
        self,
        samples_gdf,
        collection_id,
        collection_data_crs,
        file_href_col,
        process_stac_file,
        *,
        dst_res=None,
        dst_fill=0,
        dst_dtype=None,
        sample_id_col="sample",
        process_stac_file_kwargs=None,
    ):
        """You Only Download Once (YODO) processing of STAC tiles.

        Given a geo-data frame with the geometry of each sample, an ordered ascending
        list of buffer distances, the STAC collection id
        """
        # process args
        if process_stac_file_kwargs is None:
            process_stac_file_kwargs = {}
        # # buffer around sample locations:
        # # 1. projecting the sample locations to CH1903+/LV95 so that we can apply
        # #    buffers in meters
        # # 2. apply the largest buffer to each location
        # # 3. reproject to the STAC's client CRS
        # # TODO: use osmnx.project_gdf to generalize outside Switzerland
        # buffered_samples_gdf = gpd.GeoDataFrame(
        #     sample_gdf.to_crs(CH_CRS).buffer(buffer_dists[-1]).to_crs(CLIENT_CRS)
        # )

        # get the extent of our area of interest to query get the intersecting STAC
        # collection's tiles
        extent_geom = samples_gdf.union_all().convex_hull

        # get the collection gdf
        collection_gdf = self.gdf_from_collection(
            collection_id, extent_geom=extent_geom
        )

        # we want to minimize the download and opening of STAC tiles so we perform an
        # overlay of the tile extent and buffered samples. This way we get, for each
        # tile, the list of samples whose geometry intersects the tile's extent
        tile_samples_gdf = collection_gdf.overlay(
            samples_gdf.reset_index(), how="intersection"
        )

        # note that a sample geometry may intersect multiple tiles, so we may need to
        # process several STAC files for each sample. In order to download and open each
        # STAC file only once, we create a data array to store the feature array for
        # each sample, which will be modified in place as we process each STAC tile.
        # largest_buffer_pixels = int(buffer_dists[-1] / dst_res)
        # TODO: use another approach (instead of data array) to
        # TODO: use osmnx.project_gdf to generalize outside Switzerland
        minx, miny, maxx, maxy = samples_gdf.to_crs(CH_CRS)["geometry"].iloc[0].bounds
        # arr_side = 2 * largest_buffer_pixels
        arr_width = int((maxx - minx) / dst_res)
        arr_height = int((maxy - miny) / dst_res)
        dst_da = xr.DataArray(
            np.full(
                (len(samples_gdf.index), arr_height, arr_width),
                dst_fill,
                dtype=dst_dtype,
            ),
            dims=(sample_id_col, "i", "j"),
            coords={sample_id_col: samples_gdf.index},
            attrs={"nodata": dst_fill, "res": dst_res},
        )

        # we also create a dictionary to store the transform of each sample's array.
        # Note that we do have to reproject the sample geometries to the same CRS as the
        # STAC collection - again, not to be confused with the CRS of the STAC tiles in
        # the API.
        transform_dict = {
            sample_id: transform.from_bounds(*sample_geom.bounds, arr_width, arr_height)
            for sample_id, sample_geom in samples_gdf["geometry"]
            .to_crs(collection_gdf.crs)
            .items()
        }

        # ax = tile_samples_gdf.plot()
        # samples_gdf.plot(ax=ax, color="red")

        # for each STAC tile, we download the data, process it and update `dst_da`
        # accordingly. Note again that we have to reproject the sample geometries to the
        # same CRS as the STAC collection.
        for url, intersecting_samples_gdf in tqdm.tqdm(
            tile_samples_gdf.to_crs(collection_data_crs).groupby(by=file_href_col)
        ):
            _ = process_stac_file(
                dst_da,
                url,
                intersecting_samples_gdf,
                sample_id_col,
                transform_dict,
                arr_width,
                arr_height,
                dst_fill,
                **process_stac_file_kwargs,
            )

        return dst_da
