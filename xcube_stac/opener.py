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

import xarray as xr
from xcube.util.jsonschema import JsonObjectSchema
from xcube.core.store import DataOpener, new_data_store


class HttpsDataOpener(DataOpener):
    """Implementation of the data opener supporting
    the zarr, geotiff and netcdf format via the https protocol.
    """

    def __init__(self, root: str, opener_id: str):
        self._root = root
        self._opener_id = opener_id
        if opener_id.split(":")[1] == "netcdf":
            self._https_accessor = HttpsNetcdfDataAccessor(root=root)
        else:
            self._https_accessor = new_data_store("https", root=root, read_only=True)

    @property
    def root(self):
        return self._root

    def get_open_data_params_schema(self, data_id: str = None) -> JsonObjectSchema:
        return self._https_accessor.get_open_data_params_schema(
            data_id=data_id, opener_id=self._opener_id
        )

    def open_data(self, data_id: str, **open_params) -> xr.Dataset:
        stac_schema = self.get_open_data_params_schema()
        stac_schema.validate_instance(open_params)
        return self._https_accessor.open_data(
            data_id=data_id, opener_id=self._opener_id, **open_params
        )


class HttpsNetcdfDataAccessor:
    """Implementation of the data accessor supporting
    the netcdf format via the https protocol.
    """

    def __init__(self, root: str):
        self._root = root

    def get_open_data_params_schema(
        self, data_id: str = None, opener_id: str = None
    ) -> JsonObjectSchema:
        open_parms = {}
        return JsonObjectSchema(
            properties=dict(**open_parms),
            required=[],
            additional_properties=False,
        )

    def open_data(
        self, data_id: str, opener_id: str = None, **open_params
    ) -> xr.Dataset:
        stac_schema = self.get_open_data_params_schema()
        stac_schema.validate_instance(open_params)
        tile_size = open_params.get("tile_size", (512, 512))
        fs_path = "https://" + self._root + "/" + data_id + "#mode=bytes"
        return xr.open_dataset(fs_path, chunks={})


class S3DataOpener(DataOpener):
    """Implementation of the data opener supporting
    the zarr, geotiff and netcdf format via the https protocol.
    """

    def __init__(
        self,
        root: str,
        opener_id: str,
        storage_options: dict,
    ):
        self._root = root
        self._opener_id = opener_id
        self._storage_options = storage_options
        if "anon" not in storage_options:
            storage_options["anon"] = True
        self._s3_accessor = new_data_store(
            "s3",
            root=root,
            read_only=True,
            storage_options=storage_options,
        )

    @property
    def root(self):
        return self._root

    def get_open_data_params_schema(
        self, data_id: str = None, opener_id: str = None
    ) -> JsonObjectSchema:
        return self._s3_accessor.get_open_data_params_schema(
            data_id=data_id, opener_id=opener_id
        )

    def open_data(self, data_id: str, **open_params) -> xr.Dataset:
        stac_schema = self.get_open_data_params_schema()
        stac_schema.validate_instance(open_params)
        return self._s3_accessor.open_data(
            data_id=data_id, opener_id=self._opener_id, **open_params
        )
