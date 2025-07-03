# The MIT License (MIT)
# Copyright (c) 2024-2025 by the xcube development team and contributors
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

import numpy as np
import pystac
import xarray as xr

from xcube_stac.constants import LOG

_RASTER_STAC_EXTENSION_VERSIONS = {
    "v1.0.0": "https://stac-extensions.github.io/raster/v1.0.0/schema.json",
    "v1.1.0": "https://stac-extensions.github.io/raster/v1.1.0/schema.json",
    "v2.0.0": "https://stac-extensions.github.io/raster/v2.0.0/schema.json",
}


def get_stac_extension(item: pystac.Item) -> str:
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


def apply_offset_scaling_stack_mode(
    ds: xr.Dataset, access_params: xr.DataArray
) -> xr.Dataset:
    """Apply offset and scale factors to variables in a dataset per timestep
    using STAC raster extension metadata.

    This function extracts scaling and offset metadata for each asset from the
    corresponding STAC item. It supports both raster extension versions "v1" and "v2".
    Assets containing "scl" (scene classification layers in Sentinel-2 L2A product)
    are skipped. If the STAC item does not conform to the raster extension, no scaling
    is applied and a warning is logged.

    Args:
        ds: The input datasett with time-varying data variables to scale.
        access_params: A 4D DataArray containing access metadata, including the
            STAC item and asset name, indexed by time.

    Returns:
        A dataset with variables scaled and offset applied where metadata
        is available. Nodata values are masked before scaling if nodata metadata
        is present and consistent across time steps.

    Notes:
        - If multiple nodata values are detected for a single variable across time,
          an assertion will fail.
        - This function assumes that `access_params` contains at least one non-None
          value for each time step.
    """
    params = next(value for value in access_params.values.ravel() if value is not None)
    raster_version = _get_stac_extension(params["item"])
    if not raster_version:
        LOG.warning(
            f"The item {params['item'].id!r} is not conform to "
            f"the stac-extension 'raster'. No scaling is applied."
        )
        return ds

    for asset_name in list(ds.data_vars):
        if "scl" in str(asset_name).lower():
            continue
        offsets = xr.DataArray(
            np.zeros(ds.sizes["time"], dtype=np.float32),
            dims="time",
            coords=dict(time=ds.time),
        )
        scales = xr.DataArray(
            np.zeros(ds.sizes["time"], dtype=np.float32),
            dims="time",
            coords=dict(time=ds.time),
        )
        nodatas = xr.DataArray(
            np.zeros(ds.sizes["time"], dtype=np.int16),
            dims="time",
            coords=dict(time=ds.time),
        )
        for dt in ds.time.values:
            params = next(
                value
                for value in access_params.sel(time=dt).values.ravel()
                if value is not None
            )
            if raster_version == "v1":
                scale, offset = _get_scaling_v1(params["item"], params["name"])
                nodata = _get_nodata_v1(params["item"], params["name"])
            elif raster_version == "v2":
                scale, offset = _get_scaling_v2(params["item"], params["name"])
                nodata = _get_nodata_v2(params["item"], params["name"])
            else:
                LOG.warning(
                    f"Stac extension raster exists only for version 'v1' and 'v2', not "
                    f"for {raster_version!r}. No scaling is applied."
                )
                return ds
            scales.loc[dt] = scale
            offsets.loc[dt] = offset
            nodatas.loc[dt] = nodata

        assert len(np.unique(nodatas.values)) == 1
        nodata = nodatas[0].item()
        if nodata is not None:
            ds[asset_name] = ds[asset_name].where(ds[asset_name] != nodata)

        ds[asset_name] *= scales
        ds[asset_name] += offsets
    return ds


def apply_offset_scaling(
    ds: xr.Dataset, asset: pystac.Asset, raster_version: str
) -> xr.DataArray:
    """This function applies offset and scale to the data and fills no-data pixel
    with np.nan. The digital numbers DN are converted into radiance values by
    L = scale * DN + offset.

     Args:
         ds: data arrayif crs_final == crs_data:
         asset: asset object
         raster_version: version of raster stac extension; on off [v1, v2].

     Returns:
         Dataset where offset, scale, and filling nodata values are applied.

    See Also:
        `get_stac_extension` for getting the version of raster stac extension.
    """
    if raster_version == "v1":
        scale, offset = _get_scaling_v1(asset)
        nodata = _get_nodata_v1(asset)
    elif raster_version == "v2":
        scale, offset = _get_scaling_v2(asset)
        nodata = _get_nodata_v2(asset)
    else:
        LOG.warning(
            f"Stac extension raster exists only for version 'v1' and 'v2', not "
            f"for {raster_version!r}. No scaling is applied."
        )
        return ds

    if nodata is not None:
        ds = ds.where(ds != nodata)

    ds *= scale
    ds += offset
    return ds


def _get_scaling_v1(asset: pystac.Asset) -> tuple[int | float, int | float]:
    raster_bands = asset.extra_fields.get("raster:bands")
    scale = raster_bands[0].get("scale", 1)
    offset = raster_bands[0].get("offset", 0)
    return scale, offset


def _get_nodata_v1(asset: pystac.Asset) -> None | int | float:
    raster_bands = asset.extra_fields.get("raster:bands")
    return raster_bands[0].get("nodata", None)


def _get_scaling_v2(asset: pystac.Asset) -> tuple[int | float, int | float]:
    scale = asset.extra_fields.get("raster:scale", 1)
    offset = asset.extra_fields.get("raster:offset", 0)
    return scale, offset


def _get_nodata_v2(asset: pystac.Asset) -> None | int | float:
    return asset.extra_fields.get("nodata", None)
