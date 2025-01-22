# The MIT License (MIT)
# Copyright (c) 2024 by the xcube development team and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NON INFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import Union, Any

import pystac
import xarray as xr

from ..constants import LOG


_RASTER_STAC_EXTENSION_VERSIONS = {
    "v1.0.0": "https://stac-extensions.github.io/raster/v1.0.0/schema.json",
    "v1.1.0": "https://stac-extensions.github.io/raster/v1.1.0/schema.json",
    "v2.0.0": "https://stac-extensions.github.io/raster/v2.0.0/schema.json",
}


def apply_offset_scaling(
    da: xr.DataArray, item: pystac.Item, asset_name: str, raster_version: str = None
) -> xr.DataArray:
    """This function applies offset and scale to the data and fills no-data pixel
    with np.nan. The digital numbers DN are converted into radiance values by
    L = scale * DN + offset.

     Args:
         da: data array
         item: item object
         asset_name: name/key of asset
         raster_version: version of raster stac extension; on off [v1, v2].

     Returns:
         Data array where offset, scale, and filling nodata values are applied.

    See Also:
        `get_stac_extension` for getting the version of raster stac extension.
    """
    if not raster_version:
        raster_version = _get_stac_extension(item)
        if not raster_version:
            LOG.warning(
                f"The item {item.id!r} is not conform to "
                f"the stac-extension 'raster'. No scaling is applied."
            )
            return da

    if raster_version == "v1":
        scale, offset, nodata = _get_scaling_v1(item, asset_name)
    elif raster_version == "v2":
        scale, offset, nodata = _get_scaling_v2(item, asset_name)
    else:
        LOG.warning(
            f"Stac extension raster exists only for version 'v1' and 'v2', not "
            f"for {raster_version!r}. No scaling is applied."
        )
        return da

    if nodata is not None:
        da = da.where(da != nodata)
    da *= scale
    da += offset
    return da


def _get_stac_extension(item: pystac.Item) -> str:
    """Determine the version of the stac extension 'raster'.

    Args:
        item: stac item object

    Returns:
        found_version: version of raster stac extension

    Note:
        reference of raster stac extension: https://github.com/stac-extensions/raster

    """
    found_version = None
    for version, extension_url in _RASTER_STAC_EXTENSION_VERSIONS.items():
        if extension_url in item.stac_extensions:
            found_version = version.split(".")[0]
            break
    return found_version


def _get_scaling_v1(item: pystac.Item, asset_name: str) -> tuple[Any, Any, Any]:
    raster_bands = item.assets[asset_name].extra_fields.get("raster:bands")
    nodata_val = raster_bands[0].get("nodata")
    scale = raster_bands[0].get("scale", 1)
    offset = raster_bands[0].get("offset", 0)
    return scale, offset, nodata_val


def _get_scaling_v2(item: pystac.Item, asset_name: str) -> Any | Any | Any:
    nodata_val = item.assets[asset_name].extra_fields.get("nodata", None)
    scale = item.assets[asset_name].extra_fields.get("raster:scale", 1)
    offset = item.assets[asset_name].extra_fields.get("raster:offset", 0)
    return scale, offset, nodata_val
