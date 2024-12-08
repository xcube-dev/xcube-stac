import numpy as np
import pystac
import xarray as xr

from ..constants import LOG


def apply_offset_scaling(
    ds: xr.Dataset, item: pystac.Item, asset_name: str
) -> xr.Dataset:
    """This function applies offset and scale to the data and fills no-data pixel
    with np.nan. Note that the item needs to conform to the stac-extension 'raster'
    https://github.com/stac-extensions/raster/tree/main. Note that the digital numbers
    DN are converted into radiance values by L = scale * DN + offset.

     Args:
         ds: dataset
         item: item object
         asset_name: name/key of asset

     Returns:
         Dataset where offset, scale, and filling nodata values are applied.
    """
    raster_bands = item.assets[asset_name].extra_fields.get("raster:bands")
    if raster_bands is None:
        LOG.warning(
            f"The asset {asset_name} in item {item.id} is not conform to "
            f"the stac-extension 'raster'. No scaling is applied."
        )
        return ds

    if asset_name.lower() != "scl":
        nodata_val = raster_bands[0].get("nodata")
        if nodata_val is not None:
            ds[asset_name] = ds[asset_name].where(ds[asset_name] != nodata_val)
    scale = raster_bands[0].get("scale", 1)
    ds[asset_name] *= scale
    offset = raster_bands[0].get("offset", 0)
    ds[asset_name] += offset
    return ds


def apply_offset_scaling_odc_stac(ds: xr.Dataset, grouped_items: dict) -> xr.Dataset:
    for asset_name in ds.keys():
        if (
            asset_name == "crs"
            or asset_name == "spatial_ref"
            or str(asset_name).lower() == "scl"
        ):
            continue
        scale = np.zeros(len(grouped_items))
        offset = np.zeros(len(grouped_items))
        nodata_val = np.zeros(len(grouped_items))
        for i, (date, items) in enumerate(grouped_items.items()):
            raster_bands = items[0].assets[asset_name].extra_fields.get("raster:bands")
            if raster_bands is None:
                LOG.warning(
                    f"Item {items[0].id} is not conform to the stac-extension "
                    f"'raster'. No scaling is applied."
                )
                return ds

            nodata_val[i] = raster_bands[0].get("nodata")
            scale[i] = raster_bands[0].get("scale", 1)
            offset[i] = raster_bands[0].get("offset", 0)
        assert np.unique(nodata_val).size == 1
        ds[asset_name] = ds[asset_name].where(ds[asset_name] != nodata_val[0])
        ds[asset_name] *= scale[:, np.newaxis, np.newaxis]
        ds[asset_name] += offset[:, np.newaxis, np.newaxis]
    return ds
