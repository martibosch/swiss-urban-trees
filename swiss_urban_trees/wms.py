"""WMS."""

import warnings
from collections.abc import Callable
from os import path

import geopandas as gpd
import rasterio as rio
from owslib.wms import WebMapService
from rasterio import transform
from requests import exceptions
from tqdm.auto import tqdm

from swiss_urban_trees import geo_utils, settings, utils

# to show progress bar for pandas apply
tqdm.pandas()

FORMAT_TO_DRIVER_DICT = {
    "image/tiff": "GTiff",
    "image/png": "PNG",
    "image/jpeg": "JPEG",
}


class WMSDownloader:
    """Download tiles from a WMS service."""

    def __init__(
        self,
        wms_url: str,
        layer_name: str,
        dst_dir: utils.PathDType,
        res: float,
        crs: utils.CRSDType,
        max_size: int,
        format: str,
        *,
        nodata: int | None = None,
        print_func: Callable | None = None,
    ) -> None:
        """Initialize WMSDownloader.

        Parameters
        ----------
        wms_url : str
            WMS URL.
        layer_name : str
            WMS layer name.
        dst_dir : path-like
            Path to an *existing* directory where the downloaded tiles will be saved.
        res : numeric
            Target resolution (in meters) for the downloaded tiles.
        crs : crs-like
            CRS of the WMS service. Can be any object that can be passed to
            rasterio.open.
        max_size : int
            Maximum size (in number of pixels) of the downloaded tiles. The value
            corresponds to the side of a square, e.g., if max_size=256, then the
            downloaded tiles will be 256x256 pixels.
        format : str
            Image format for the downloaded tiles. Must be a valid format for the
            WMS service.
        nodata : int
            No data value for the downloaded tiles.
        print_func : Callable, optional
            Function to use to print messages. If not provided, the default function is
            `utils.log`.

        """
        self.wms = WebMapService(wms_url)
        self.layer_name = layer_name
        self.dst_dir = dst_dir
        self.res = res
        self.crs = crs
        self.max_size = max_size
        self.tile_size = self.res * self.max_size
        self.format = format
        self.ext = format.split("/")[-1]
        self.base_meta = {
            "driver": FORMAT_TO_DRIVER_DICT[format],
            "dtype": "uint8",
            "nodata": nodata,
            "count": 3,
            "height": self.max_size,
            "width": self.max_size,
            "crs": self.crs,
        }
        if print_func is None:
            print_func = utils.log
        self.print_func = print_func

    def _tile_filepath(self, xmin: float, ymin: float, xmax: float, ymax: float) -> str:
        """Return the path to a tile.

        Parameters
        ----------
        xmin, xmax, ymin, ymax : numeric
            Bounding box of the tile, in the CRS of the WMS service.

        Returns
        -------
        tile_filepath : str
            Path to the tile.

        """
        # using only one decimal place due to the WMS resolution of 0.2m
        return path.join(
            self.dst_dir, f"{xmin:.1f}_{ymin:.1f}_{xmax:.1f}_{ymax:.1f}.{self.ext}"
        )

    def _dump_geo_tile(
        self,
        xmin: float,
        ymin: float,
        xmax: float,
        ymax: float,
    ) -> str:
        """Download a tile from the WMS service and save it to disk.

        Parameters
        ----------
        xmin, xmax, ymin, ymax : numeric
            Bounding box of the tile to download, in the CRS of the WMS service.

        Returns
        -------
        tile_filepath : str
            Path to the downloaded tile.

        """
        for retry_count in range(settings.MAX_RETRIES):
            try:
                request = self.wms.getmap(
                    layers=[self.layer_name],
                    srs=self.crs,
                    format=self.format,
                    bbox=(xmin, ymin, xmax, ymax),
                    size=(self.max_size, self.max_size),
                )
                break
            except exceptions.Timeout:
                self.print_func(
                    f"Timeout when downloading tile {xmin:.1f} {ymin:.1f} {xmax:.1f} "
                    f"{ymax:.1f}. Retrying..."
                )
                if retry_count == settings.MAX_RETRIES - 1:
                    self.print_func(
                        f"Max retries ({settings.MAX_RETRIES}) reached when downloading"
                        f" tile {xmin:.1f} {ymin:.1f} {xmax:.1f} {ymax:.1f}. Skipping."
                    )

        tile_filepath = self._tile_filepath(xmin, ymin, xmax, ymax)

        meta = self.base_meta.copy()
        meta["transform"] = transform.from_bounds(
            xmin, ymin, xmax, ymax, self.max_size, self.max_size
        )

        with warnings.catch_warnings():
            # filter NotGeoreferencedWarning when opening memfile
            warnings.filterwarnings("ignore")
            with rio.MemoryFile(request) as memfile:
                with memfile.open() as src:
                    with rio.open(
                        tile_filepath,
                        "w",
                        **meta,
                    ) as dst:
                        # dst.write(dst_arr)
                        dst.write(src.read()[:3])

        # size_kb = len(response.content) / 1000
        # domain = re.findall(r"(?s)//(.*?)/", url)[0]
        # utils.log(f"Downloaded {size_kb:,.1f}kB from {domain}")

        return tile_filepath

    def get_tile_gser(self, aoi: utils.AOIDType) -> gpd.GeoDataFrame:
        """Get a geo-data frame with the tiles covering the AOI."""
        return geo_utils.get_tile_gser(aoi, self.tile_size, self.tile_size, self.crs)

    def download_aoi(
        self,
        aoi: utils.AOIDType,
        *,
        keep_existing: bool = True,
        filepath_col="tile_filepath",
    ) -> gpd.GeoDataFrame:
        """Download tiles for an area of interest (AOI).

        Parameters
        ----------
        aoi : shapely.geometry.Polygon or shapely.geometry.MultiPolygon
            Geometry representing the area of interest (AOI) to download tiles for.
        keep_existing : bool, default True
            If True, tiles that already exist in the destination directory will not be
            downloaded again.
        filepath_col : str, default "tile_filepath"
            Name of the column in the returned geo-data frame that will contain the
            paths to the downloaded tiles.

        Returns
        -------
        tile_gdf : geopandas.GeoDataFrame
            Geo-data frame with the paths to the downloaded tiles and extent bounds.

        """
        tile_gser = self.get_tile_gser(aoi)
        num_tiles = len(tile_gser)
        if tile_gser.empty:
            self.print_func(
                "No intersecting tiles found, returning empty GeoDataFrame."
            )
            return gpd.GeoDataFrame()
        else:
            self.print_func(f"Splitting the provided AOI into {num_tiles} tiles.")
            tile_gdf = gpd.GeoDataFrame(
                {
                    "tile_filepath": tile_gser.bounds.apply(
                        lambda row: self._tile_filepath(*row), axis=1
                    )
                },
                geometry=tile_gser,
            )

            if keep_existing:
                # filter tiles that already exist in the destination directory
                tile_gser = tile_gser[~tile_gdf["tile_filepath"].apply(path.exists)]
                num_skip_tiles = num_tiles - len(tile_gser.index)
                if num_skip_tiles > 0:
                    self.print_func(
                        f"Skipping {num_skip_tiles} tiles that already exist in "
                        f"{self.dst_dir}."
                    )

            # download tiles
            if tile_gser.empty:
                self.print_func("No tiles to download.")
            else:
                self.print_func(f"Downloading {len(tile_gser)} tiles...")
                _ = list(
                    tile_gser.bounds.progress_apply(
                        lambda row: self._dump_geo_tile(*row),
                        axis=1,
                    )
                )
                self.print_func(f"Downloaded {len(tile_gser)} tiles at {self.dst_dir}.")

        return tile_gdf
