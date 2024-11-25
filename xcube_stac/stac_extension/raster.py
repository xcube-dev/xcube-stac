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
            f"Item {item.id} is not conform to the stac-extension 'raster'. "
            f"No scaling is applied."
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
