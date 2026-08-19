# The MIT License (MIT)
# Copyright (c) 2024-2026 by the xcube development team and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
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

from collections.abc import Sequence
from typing import Any

import numpy as np
import pystac
import xarray as xr
from xcube.util.jsonschema import JsonArraySchema, JsonStringSchema
from xcube_resampling import resample_in_space
from xcube_resampling.gridmapping import GridMapping
from xcube_resampling.utils import reproject_bbox, resolution_meters_to_degrees

from xcube_stac.stac_extension.raster import apply_offset_scaling
from xcube_stac.utils import (
    _remove_fill_value_encoding,
    add_attributes,
    add_nominal_datetime,
    rename_dataset,
)
from xcube_stac.version import version

from .hls import Sen2HlsStacArdcAccessor, Sen2HlsStacItemAccessor

_LANDSAT_BANDS = [
    "qa",
    "red",
    "blue",
    "drad",
    "emis",
    "emsd",
    "trad",
    "urad",
    "atran",
    "cdist",
    "green",
    "nir08",
    "lwir11",
    "swir16",
    "swir22",
    "coastal",
    "qa_pixel",
    "qa_radsat",
    "qa_aerosol",
]
_CHUNK_SIZE = {"x": 2048, "y": 2048}


class LandsatC2L2StacItemAccessor(Sen2HlsStacItemAccessor):
    """Provides methods for accessing a Planetary Computer Landsat Collection 2
    Level-2 STAC Item.
    """

    def __init__(self, catalog: pystac.Catalog, **storage_options_s3):
        super().__init__(catalog, **storage_options_s3)
        self._asset_names_default = _LANDSAT_BANDS
        self._schema_asset_names = JsonArraySchema(
            items=(JsonStringSchema(min_length=1, enum=_LANDSAT_BANDS)),
            unique_items=True,
            title="Names of assets (spectral and ancillary bands)",
        )
        self._fill_values = {
            "qa_pixel": 1,
            "qa_radsat": 65535,
            "qa_aerosol": 1,
        }

    def _combiner_function(
        self,
        dss: Sequence[xr.Dataset],
        item: pystac.Item,
        catalog: pystac.Catalog,
        assets: Sequence[pystac.Asset] | None = None,
        apply_scaling: bool = True,
        **open_params,
    ) -> xr.Dataset:
        dss = [
            rename_dataset(ds, asset.extra_fields["asset_name"])
            for (ds, asset) in zip(dss, assets)
        ]
        ds = dss[0].copy()
        for ds_asset in dss[1:]:
            ds.update(ds_asset)
        for name, asset in item.assets.items():
            if name in ds:
                array = add_stac_asset_attributes(ds[name], asset)
                if name == "lwir11":
                    array.attrs["units"] = "K"
                if apply_scaling:
                    if "qa" not in name:
                        array = apply_offset_scaling(array, asset, "v1")
                ds[name] = array
        ds.attrs.update(
            stac_url=catalog.get_self_href(),
            stac_item_id=item.id,
            xcube_stac_version=version,
        )

        # remove _FillValue from encoding and attrs for integer valued arrays
        ds = _remove_fill_value_encoding(ds)

        # resample dataset if requested
        crs = open_params.get("crs")
        bbox = open_params.get("bbox")
        spatial_res = open_params.get("spatial_res")
        tile_size = open_params.get("tile_size", _CHUNK_SIZE.values())
        if crs is None and bbox is None and spatial_res is None:
            return ds

        source_gm = GridMapping.from_dataset(ds)
        if bbox is None:
            if crs:
                bbox = reproject_bbox(source_gm.xy_bbox, source_gm.crs, crs)
            else:
                bbox = source_gm.xy_bbox
        if spatial_res is None:
            if crs and crs.is_geographic:
                center_lat = (source_gm.xy_bbox[1] + source_gm.xy_bbox[3]) / 2
                spatial_res = resolution_meters_to_degrees(source_gm.xy_res, center_lat)
            else:
                spatial_res = source_gm.xy_res
        if crs is None:
            crs = source_gm.crs
        target_gm = GridMapping.regular_from_bbox(
            bbox=bbox, xy_res=spatial_res, crs=crs, tile_size=tile_size
        )
        ds = resample_in_space(
            ds,
            source_gm=source_gm,
            target_gm=target_gm,
            prevent_nan_propagations=True,
            fill_values={"Fmask": 255},
        )
        return ds


class LandsatC2L2StacArdcAccessor(LandsatC2L2StacItemAccessor, Sen2HlsStacArdcAccessor):
    """Provides utilities to retrieve multiple Landsat STAC items from the
    Planetary Computer STAC API and assemble them into a data cube.
    """

    def open_ardc(
        self,
        data_id: str,
        items: Sequence[pystac.Item],
        **open_params,
    ) -> xr.Dataset:
        grouped_items = self._group_items(items)
        ds = self._generate_cube(grouped_items, **open_params)
        if "qa_pixel" in ds:
            ds["qa_pixel"] = ds["qa_pixel"].fillna(1).astype(np.uint16)
        ds.attrs.pop("stac_item_id", None)
        ds = add_attributes(
            data_id, self._catalog.get_self_href(), ds, grouped_items, **open_params
        )
        return ds

    @staticmethod
    def _group_items(items: Sequence[pystac.Item]):
        """Group Landsat STAC items by solar day and path/row."""
        items = add_nominal_datetime(items)

        dates = []
        tile_ids = []
        for item in items:
            dates.append(item.properties["datetime_nominal"].date())
            tile_ids.append(
                f"{int(item.properties['landsat:wrs_path']):03d}"
                f"{int(item.properties['landsat:wrs_row']):03d}"
            )
        dates = np.unique(dates)
        tile_ids = np.unique(tile_ids)

        grouped_items = np.full((len(dates), len(tile_ids)), None, dtype=object)
        for item in items:
            date = item.properties["datetime_nominal"].date()
            tile_id = (
                f"{int(item.properties['landsat:wrs_path']):03d}"
                f"{int(item.properties['landsat:wrs_row']):03d}"
            )
            idx_date = np.where(dates == date)[0][0]
            idx_tile_id = np.where(tile_ids == tile_id)[0][0]
            if grouped_items[idx_date, idx_tile_id] is None:
                grouped_items[idx_date, idx_tile_id] = [item]
            else:
                grouped_items[idx_date, idx_tile_id].append(item)

        for idx_date in range(grouped_items.shape[0]):
            for idx_tile_id in range(grouped_items.shape[1]):
                if grouped_items[idx_date, idx_tile_id] is None:
                    grouped_items[idx_date, idx_tile_id] = []

        grouped_items = xr.DataArray(
            grouped_items,
            dims=("time", "tile_id"),
            coords={"time": dates, "tile_id": tile_ids},
        )

        dts = []
        for date in grouped_items.time.values:
            item = np.sum(grouped_items.sel(time=date).values)[0]
            dts.append(item.datetime.replace(tzinfo=None))
        grouped_items = grouped_items.assign_coords(
            time=np.array(dts, dtype="datetime64[ns]")
        )

        return grouped_items


def add_stac_asset_attributes(
    da: xr.DataArray,
    asset: pystac.Asset,
) -> xr.DataArray:
    """Add CF-style metadata from a STAC asset to an xarray DataArray."""
    attrs = da.attrs

    # Asset title -> long_name
    title = asset.title
    if title:
        attrs["long_name"] = title

    # EO Bands extension
    _add_eo_band_attributes(attrs, asset)

    # Classification extension
    _add_classification_bitfields(attrs, asset)

    return da


def _add_classification_bitfields(
    attrs: dict[str, Any],
    asset: pystac.Asset,
) -> None:
    """Add CF flag metadata from STAC classification bitfields."""
    bitfields = asset.extra_fields.get("classification:bitfields")

    if not bitfields:
        return

    flag_masks: list[int] = []
    flag_values: list[int] = []
    flag_meanings: list[str] = []

    for bitfield in bitfields:
        name = bitfield["name"]
        offset = bitfield["offset"]
        length = bitfield["length"]

        # Mask covering the complete bit field.
        mask = ((1 << length) - 1) << offset

        classes = bitfield.get("classes", [])

        for classification in classes:
            value = classification["value"]

            # Shift the class value into its actual bit position.
            encoded_value = value << offset

            meaning = classification["name"]

            flag_masks.append(mask)
            flag_values.append(encoded_value)
            flag_meanings.append(f"{name}_{meaning}")

    if flag_masks:
        attrs["flag_masks"] = flag_masks
        attrs["flag_values"] = flag_values
        attrs["flag_meanings"] = " ".join(flag_meanings)


def _add_eo_band_attributes(
    attrs: dict[str, Any],
    asset: pystac.Asset,
) -> None:
    """Add attributes from the STAC EO Band extension."""
    eo_bands = asset.extra_fields.get("eo:bands")

    if not eo_bands:
        return

    band = eo_bands[0]

    if "name" in band:
        attrs["band_name"] = band["name"]

    if "common_name" in band:
        attrs["common_name"] = band["common_name"]

    if "description" in band:
        attrs["description"] = band["description"]

    if "center_wavelength" in band:
        attrs["center_wavelength"] = band["center_wavelength"]

    if "full_width_half_max" in band:
        attrs["full_width_half_max"] = band["full_width_half_max"]
