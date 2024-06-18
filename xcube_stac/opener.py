# The MIT License (MIT)
# Copyright (c) 2024 by the xcube development team and contributors
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

import rioxarray
import xarray as xr
from xcube.util.jsonschema import (
    JsonArraySchema,
    JsonBooleanSchema,
    JsonNumberSchema,
    JsonObjectSchema,
    JsonStringSchema,
)
from xcube.core.store import DataOpener
from xcube.core.store.fs.impl.dataset import GEOTIFF_OPEN_DATA_PARAMS_SCHEMA


class HttpsNetcdfDataOpener(DataOpener):
    """Implementation of the data opener supporting
    the netcdf format via the https protocol.
    """

    def get_open_data_params_schema(self, data_id: str = None) -> JsonObjectSchema:
        open_parms = dict(
            tile_size=JsonArraySchema(
                items=(
                    JsonNumberSchema(minimum=256, default=512),
                    JsonNumberSchema(minimum=256, default=512),
                ),
                title="Tile size [y, x] for chunking",
                default=[512, 512],
            ),
        )
        return JsonObjectSchema(
            properties=dict(**open_parms),
            required=[],
            additional_properties=False,
        )

    def open_data(self, data_id: str, **open_params) -> xr.Dataset:
        stac_schema = self.get_open_data_params_schema()
        stac_schema.validate_instance(open_params)
        tile_size = open_params.get("tile_size", (512, 512))
        return rioxarray.open_rasterio(data_id, chunks=dict(zip(("x", "y"), tile_size)))


class HttpsTiffDataOpener(DataOpener):
    """Implementation of the data opener supporting
    the tiff and geotiff format via the https protocol.
    """

    def get_open_data_params_schema(self, data_id: str = None) -> JsonObjectSchema:
        return GEOTIFF_OPEN_DATA_PARAMS_SCHEMA

    def open_data(self, data_id: str, **open_params) -> xr.Dataset:
        stac_schema = self.get_open_data_params_schema()
        stac_schema.validate_instance(open_params)
        tile_size = open_params.get("tile_size", (512, 512))
        overview_level = open_params.get("overview_level", None)
        return rioxarray.open_rasterio(
            data_id,
            overview_level=overview_level,
            chunks=dict(zip(("x", "y"), tile_size)),
        )


class HttpsZarrDataOpener(DataOpener):
    """Implementation of the data opener supporting
    the zarr format via the https protocol.
    """

    def get_open_data_params_schema(self, data_id: str = None) -> JsonObjectSchema:
        open_parms = dict(
            group=JsonStringSchema(
                description="Group path." " (a.k.a. path in zarr terminology.).",
                min_length=1,
            ),
            chunks=JsonObjectSchema(
                description="Optional chunk sizes along each dimension."
                ' Chunk size values may be None, "auto"'
                " or an integer value.",
                examples=[
                    {"time": None, "lat": "auto", "lon": 90},
                    {"time": 1, "y": 512, "x": 512},
                ],
                additional_properties=True,
            ),
            decode_cf=JsonBooleanSchema(
                description="Whether to decode these variables,"
                " assuming they were saved according to"
                " CF conventions.",
                default=True,
            ),
            mask_and_scale=JsonBooleanSchema(
                description="If True, replace array values equal"
                ' to attribute "_FillValue" with NaN. '
                ' Use "scale_factor" and "add_offset"'
                " attributes to compute actual values.",
                default=True,
            ),
            decode_times=JsonBooleanSchema(
                description="If True, decode times encoded in the"
                " standard NetCDF datetime format "
                "into datetime objects. Otherwise,"
                " leave them encoded as numbers.",
                default=True,
            ),
            decode_coords=JsonBooleanSchema(
                description='If True, decode the "coordinates"'
                " attribute to identify coordinates in "
                "the resulting dataset.",
                default=True,
            ),
            drop_variables=JsonArraySchema(
                items=JsonStringSchema(min_length=1),
            ),
        )
        return JsonObjectSchema(
            properties=dict(**open_parms),
            required=[],
            additional_properties=False,
        )

    def open_data(self, data_id: str, **open_params) -> xr.Dataset:
        stac_schema = self.get_open_data_params_schema()
        stac_schema.validate_instance(open_params)
        return xr.open_zarr(data_id, **open_params)
