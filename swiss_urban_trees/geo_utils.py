"""Geospatial utilities."""

import typing

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import geometry


def get_tile_gser(
    geom: geometry.base.BaseGeometry,
    tile_width: float,
    tile_height: float,
    crs: typing.Any,
) -> gpd.GeoSeries:
    """Get a GeoSeries of tiles of a given width and height that cover the geometry.

    Parameters
    ----------
    geom : shapely.geometry
        The geometry to cover with windows.
    tile_width, tile_height : numeric
        The width and height of the tiles.
    crs : crs-like
        The CRS of the geometry, can be anything accepted by geopandas.Geoseries.

    """
    grid_x, grid_y = np.meshgrid(
        np.arange(geom.bounds[0], geom.bounds[2], tile_width),
        np.arange(geom.bounds[1], geom.bounds[3], tile_height),
        indexing="xy",
    )
    # vectorize the grid as a geo series
    flat_grid_x = grid_x.flatten()
    flat_grid_y = grid_y.flatten()
    region_gser = gpd.GeoSeries(
        pd.DataFrame(
            {
                "xmin": flat_grid_x,
                "ymin": flat_grid_y,
                "xmax": flat_grid_x + tile_width,
                "ymax": flat_grid_y + tile_height,
            }
        ).apply(lambda row: geometry.box(*row), axis=1),
        crs=crs,
    )

    # filter out tiles that don't intersect the geometry
    return region_gser[region_gser.intersects(geom)]
