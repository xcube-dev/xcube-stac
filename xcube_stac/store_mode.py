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

import collections
import json
from typing import Iterator, Union

import dask.array as da
import numpy as np
import pyproj
import pystac
import pystac_client.client
import rioxarray
import requests
import xarray as xr
from xcube.core.mldataset import MultiLevelDataset
from xcube.core.store import DataStoreError, DataTypeLike, new_data_store
from xcube.core.gridmapping import GridMapping
from xcube.core.resampling import resample_in_space
from xcube.util.jsonschema import JsonObjectSchema
import xmltodict

from .accessor import (
    HttpsDataAccessor,
    S3DataAccessor,
)
from .constants import (
    FloatInt,
    LOG,
    STAC_SEARCH_PARAMETERS_STACK_MODE,
)
from .constants import COLLECTION_PREFIX
from .constants import TILE_SIZE
from .mldataset import SingleItemMultiLevelDataset
from .mldataset import StackModeMultiLevelDataset
from .util import Util
from .util import CdseUtil
from .util import XcubeUtil
from ._utils import (
    add_nominal_datetime,
    convert_datetime2str,
    get_data_id_from_pystac_object,
    get_url_from_pystac_object,
    is_valid_ml_data_type,
    list_assets_from_item,
    reproject_bbox,
    search_collections,
    update_dict,
    xarray_rename_vars,
)

_HTTPS_STORE = new_data_store("https")
_OPEN_DATA_PARAMETERS = {
    "open_params_dataset_netcdf": _HTTPS_STORE.get_open_data_params_schema(
        opener_id="dataset:netcdf:https"
    ),
    "open_params_dataset_zarr": _HTTPS_STORE.get_open_data_params_schema(
        opener_id="dataset:zarr:https"
    ),
    "open_params_dataset_geotiff": _HTTPS_STORE.get_open_data_params_schema(
        opener_id="dataset:geotiff:https"
    ),
    "open_params_mldataset_geotiff": _HTTPS_STORE.get_open_data_params_schema(
        opener_id="mldataset:geotiff:https"
    ),
    "open_params_dataset_jp2": _HTTPS_STORE.get_open_data_params_schema(
        opener_id="dataset:geotiff:https"
    ),
    "open_params_mldataset_jp2": _HTTPS_STORE.get_open_data_params_schema(
        opener_id="mldataset:geotiff:https"
    ),
    "open_params_dataset_levels": _HTTPS_STORE.get_open_data_params_schema(
        opener_id="dataset:levels:https"
    ),
    "open_params_mldataset_levels": _HTTPS_STORE.get_open_data_params_schema(
        opener_id="mldataset:levels:https"
    ),
}


class SingleStoreMode:
    """Implementations to access single STAC items"""

    def __init__(
        self,
        catalog: Union[pystac.Catalog, pystac_client.client.Client],
        url_mod: str,
        searchable: bool,
        storage_options_s3: dict,
        util: Union[Util, CdseUtil, XcubeUtil],
    ):
        self._catalog = catalog
        self._url_mod = url_mod
        self._searchable = searchable
        self._storage_options_s3 = storage_options_s3
        self._util = util
        self._https_accessor = None
        self._s3_accessor = None

    def access_item(self, data_id: str) -> pystac.Item:
        """Access item for a given data ID.

        Args:
            data_id: An identifier of data that is provided by this store.

        Returns:
            item object

        Raises:
            DataStoreError: Error, if the item json cannot be accessed.
        """
        response = requests.request(method="GET", url=f"{self._url_mod}{data_id}")
        if response.ok:
            return pystac.Item.from_dict(
                json.loads(response.text),
                href=f"{self._url_mod}{data_id}",
                root=self._catalog,
                preserve_dict=False,
            )
        else:
            raise DataStoreError(response.raise_for_status())

    def get_data_ids(
        self, data_type: DataTypeLike = None
    ) -> Iterator[tuple[str, pystac.Item]]:
        for item in self._catalog.get_items(recursive=True):
            if is_valid_ml_data_type(data_type):
                if not self._util.is_mldataset_available(item):
                    continue
            data_id = get_data_id_from_pystac_object(item, catalog_url=self._url_mod)
            yield data_id, item

    def get_open_data_params_schema(
        self,
        data_id: str = None,
        opener_id: str = None,
    ) -> JsonObjectSchema:
        properties = {}
        if opener_id is not None:
            key = "_".join(opener_id.split(":")[:2])
            key = f"open_params_{key}"
            properties[key] = _OPEN_DATA_PARAMETERS[key]
        if data_id is not None:
            item = self.access_item(data_id)
            for format_id in self._util.get_format_ids(item):
                for key in _OPEN_DATA_PARAMETERS.keys():
                    if format_id == key.split("_")[-1]:
                        properties[key] = _OPEN_DATA_PARAMETERS[key]
        if not properties:
            properties = _OPEN_DATA_PARAMETERS

        return JsonObjectSchema(
            properties=update_dict(
                self._util.schema_open_params, properties, inplace=False
            ),
            required=[],
            additional_properties=False,
        )

    def open_data(
        self,
        data_id: str,
        opener_id: str = None,
        data_type: DataTypeLike = None,
        **open_params,
    ) -> Union[xr.Dataset, MultiLevelDataset]:
        schema = self.get_open_data_params_schema(data_id=data_id, opener_id=opener_id)
        schema.validate_instance(open_params)
        item = self.access_item(data_id)
        ds = self.build_dataset(
            item, opener_id=opener_id, data_type=data_type, **open_params
        )
        return ds

    def get_extent(self, data_id: str) -> dict:
        item = self.access_item(data_id)

        # prepare metadata
        time_range = (None, None)
        if "start_datetime" in item.properties and "end_datetime" in item.properties:
            time_range = (
                item.properties["start_datetime"],
                item.properties["end_datetime"],
            )
        elif "datetime" in item.properties:
            time_range = (item.properties["datetime"], None)

        return dict(bbox=item.bbox, time_range=time_range)

    def search_data(self, **search_params) -> Iterator[pystac.Item]:
        schema = self.get_search_params_schema()
        schema.validate_instance(search_params)
        items = self._util.search_data(self._catalog, self._searchable, **search_params)
        return items

    def get_search_params_schema(self) -> JsonObjectSchema:
        return JsonObjectSchema(
            properties=self._util.schema_search_params,
            required=[],
            additional_properties=False,
        )

    def build_dataset(
        self,
        item: pystac.Item,
        opener_id: str = None,
        data_type: DataTypeLike = None,
        **open_params,
    ) -> Union[xr.Dataset, MultiLevelDataset]:
        """Builds a dataset where the data variable names correspond
        to the asset keys. If the loaded data consists of multiple
        data variables, the variable name follows the structure
        '<asset_key>_<data_variable_name>'

        Args:
            xitem: internal item object
            opener_id: Data opener identifier. Defaults to None.
            data_type: Data type assigning the return value data type.
                May be given as type alias name, as a type, or as a
                :class:`xcube.core.store.DataType` instance.

        Returns:
            Dataset representation of the data resources identified
            by *data_id* and *open_params*.
        """
        parsed_item = self._util.parse_item(item, **open_params)
        access_params = self._util.get_data_access_params(
            parsed_item, opener_id=opener_id, data_type=data_type, **open_params
        )
        list_ds_asset = []
        for asset_key, params in access_params.items():
            if opener_id is not None:
                key = "_".join(opener_id.split(":")[:2])
                open_params_asset = open_params.get(f"open_params_{key}", {})
            elif data_type is not None:
                open_params_asset = open_params.get(
                    f"open_params_{data_type}_{params["format_id"]}", {}
                )
            else:
                open_params_asset = open_params.get(
                    f"open_params_dataset_{params["format_id"]}", {}
                )

            if params["protocol"] == "https":
                opener = self._get_https_accessor(params)
                ds_asset = opener.open_data(
                    params,
                    opener_id=opener_id,
                    data_type=data_type,
                    **open_params_asset,
                )
            elif params["protocol"] == "s3":
                opener = self._get_s3_accessor(params)
                ds_asset = opener.open_data(
                    params,
                    opener_id=opener_id,
                    data_type=data_type,
                    **open_params_asset,
                )
            else:
                url = get_url_from_pystac_object(item)
                raise DataStoreError(
                    f"Only 's3' and 'https' protocols are supported, not "
                    f"{params["protocol"]!r}. The asset {asset_key!r} has a href "
                    f"{params["href"]!r}. The item's url is given by {url!r}."
                )

            if isinstance(ds_asset, MultiLevelDataset):
                var_names = list(ds_asset.base_dataset.keys())
            else:
                var_names = list(ds_asset.keys())
            if len(var_names) == 1:
                name_dict = {var_names[0]: asset_key}
            else:
                name_dict = {
                    var_name: f"{asset_key}_{var_name}" for var_name in var_names
                }
            if isinstance(ds_asset, MultiLevelDataset):
                ds_asset = ds_asset.apply(xarray_rename_vars, dict(name_dict=name_dict))
            else:
                ds_asset = ds_asset.rename_vars(name_dict=name_dict)
            list_ds_asset.append(ds_asset)

        if len(list_ds_asset) == 1:
            ds = list_ds_asset[0]
        else:
            if all(isinstance(ds, MultiLevelDataset) for ds in list_ds_asset):
                ds = SingleItemMultiLevelDataset(list_ds_asset, parsed_item)
            else:
                ds = list_ds_asset[0].copy()
                for ds_asset in list_ds_asset[1:]:
                    ds.update(ds_asset)
                ds = self._util.apply_scaling_nodata(ds, parsed_item)
        return ds

    ##########################################################################
    # Implementation helpers

    def _get_s3_accessor(self, access_params: dict) -> S3DataAccessor:
        """This function returns the S3 data accessor associated with the
        bucket *root*. It creates the S3 data accessor only if it is not
        created yet or if *root* changes.

        Args:
            access_params: dictionary containing access parameter for one asset

        Returns:
            S3 data opener
        """

        if self._s3_accessor is None:
            self._s3_accessor = self._util.s3_accessor(
                access_params["root"],
                fs=self._util.fs,
                storage_options=update_dict(
                    self._storage_options_s3,
                    access_params["storage_options"],
                    inplace=False,
                ),
            )
        elif not self._s3_accessor.root == access_params["root"]:
            LOG.debug(
                f"The bucket {self._s3_accessor.root!r} of the "
                f"S3 object storage changed to {access_params["root"]!r}. "
                "A new s3 data opener will be initialized."
            )
            self._s3_accessor = self._util.s3_accessor(
                access_params["root"],
                fs=self._util.fs,
                storage_options=update_dict(
                    self._storage_options_s3,
                    access_params["storage_options"],
                    inplace=False,
                ),
            )

        return self._s3_accessor

    def _get_https_accessor(self, access_params: dict) -> HttpsDataAccessor:
        """This function returns the HTTPS data opener associated with the
        *root* url and the *opener_id*. It creates the HTTPS data opener
        only if it is not created yet or if *root* or *opener_id* changes.

        Args:
            access_params: dictionary containing asset parameters for one asset

        Returns:
            HTTPS data opener
        """
        if self._https_accessor is None:
            self._https_accessor = HttpsDataAccessor(access_params["root"])
        elif not self._https_accessor.root == access_params["root"]:
            LOG.debug(
                f"The root {self._https_accessor.root!r} of the "
                f"https data opener changed to {access_params["root"]!r}. "
                "A new https data opener will be initialized."
            )
            self._https_accessor = HttpsDataAccessor(access_params["root"])
        return self._https_accessor


class StackStoreMode:
    """Implementations to access stacked STAC items within one collection"""

    def __init__(
        self,
        catalog: Union[pystac.Catalog, pystac_client.client.Client],
        url_mod: str,
        searchable: bool,
        storage_options_s3: dict,
        util: Util,
    ):
        self._catalog = catalog
        self._url_mod = url_mod
        self._searchable = searchable
        self._storage_options_s3 = storage_options_s3
        self._util = util

    def access_collection(self, data_id: str) -> pystac.Collection:
        """Access collection for a given data ID.

        Args:
            data_id: An identifier of data that is provided by this store.

        Returns:
            collection object.

        Raises:
            DataStoreError: Error, if the item json cannot be accessed.
        """
        if COLLECTION_PREFIX in data_id:
            data_id = data_id.replace(COLLECTION_PREFIX, "")
        collections = self._catalog.get_collections()
        return next((c for c in collections if c.id == data_id), None)

    def access_item(self, data_id: str) -> pystac.Item:
        """Access the first item of a collection for a given data ID.

        Args:
            data_id: An identifier of data that is provided by this store.

        Returns:
            item object.

        Raises:
            DataStoreError: Error, if the item json cannot be accessed.
        """
        collection = self.access_collection(data_id)
        return next(collection.get_items())

    def get_data_ids(
        self, data_type: DataTypeLike = None
    ) -> Iterator[tuple[str, pystac.Collection]]:
        for collection in self._catalog.get_collections():
            if is_valid_ml_data_type(data_type):
                item = next(collection.get_items())
                if not self._util.is_mldataset_available(item):
                    continue
            yield collection.id, collection

    def get_open_data_params_schema(
        self,
        data_id: str = None,
        opener_id: str = None,
    ):
        return JsonObjectSchema(
            title="Open data parameters",
            description=(
                "All keyword arguments of the supported stacking library are "
                "supported, but not listed specifically."
            ),
            properties=self._util.schema_open_params_stack,
            required=[],
            additional_properties=True,
        )

    def open_data(
        self,
        data_id: str,
        opener_id: str = None,
        data_type: DataTypeLike = None,
        bbox: [FloatInt, FloatInt, FloatInt, FloatInt] = None,
        spatial_res: float = None,
        time_range: [str, str] = None,
        **open_params,
    ) -> Union[xr.Dataset, MultiLevelDataset]:
        schema = self.get_open_data_params_schema(data_id=data_id, opener_id=opener_id)
        schema.validate_instance(open_params)
        crs = open_params.get("crs", "EPSG:4326")
        bbox_wgs84 = reproject_bbox(bbox, crs, "EPSG:4326")

        items = list(
            self._util.search_data(
                self._catalog,
                self._searchable,
                collections=[data_id],
                bbox=bbox_wgs84,
                time_range=time_range,
                query=open_params.get("query"),
            )
        )
        if len(items) == 0:
            LOG.warn(
                f"No items found in collection {data_id!r} for the "
                f"parameters bbox {bbox!r}, time_range {time_range!r} and "
                f"query {open_params.get("query", "None")!r}"
            )
        sorted(items, key=lambda item: item.properties.get("datetime"))

        grouped = self._groupby_solar_day(items)
        grouped = self._get_mosaic_timestamps(grouped, items)
        parsed_items = self._util.parse_items_stack(items, grouped, **open_params)

        target_gm = self._get_gridmapping(
            bbox, spatial_res, crs, open_params.get("tile_size", TILE_SIZE)
        )

        if opener_id is None:
            opener_id = ""
        if "bands" not in open_params:

            assets = list_assets_from_item(
                parsed_items[next(iter(parsed_items))][0],
                supported_format_ids=self._util.supported_format_ids,
            )
            open_params["bands"] = [asset.extra_fields["id"] for asset in assets]

        if is_valid_ml_data_type(data_type) or opener_id.split(":")[0] == "mldataset":
            raise NotImplementedError("mldataset not supported in stacking mode")
        else:
            ds_dates = []
            np_datetimes = []
            for datetime, items_for_date in parsed_items.items():
                print(datetime)
                np_datetimes.append(np.datetime64(datetime).astype("datetime64[ns]"))
                list_ds_items = []
                for item in items_for_date:
                    list_ds_asset = []
                    angles = self._get_angles_from_item(item)
                    for band in open_params["bands"]:
                        ds = rioxarray.open_rasterio(
                            item.assets[band].href,
                            chunks=dict(x=1024, y=1024),
                            band_as_variable=True,
                        )
                        ds = ds.rename(dict(band_1=band))
                        ds = ds.where(ds != 0)
                        ds = _resample_in_space(ds, target_gm)
                        list_ds_asset.append(ds)
                    ds = list_ds_asset[0].copy()
                    for ds_asset in list_ds_asset[1:]:
                        ds.update(ds_asset)
                    list_ds_items.append(ds)
                ds_mosaic = _mosaic_first_non_nan(list_ds_items)
                ds_dates.append(ds_mosaic)
            ds = xr.concat(ds_dates, dim="time")
            ds = ds.assign_coords(coords=dict(time=np_datetimes))
            ds = ds.drop_vars("crs")
            ds["crs"] = ds_dates[0].crs
            ds = self._util.apply_offset_scaling(ds, parsed_items)
        return ds

    def get_extent(self, data_id: str) -> dict:
        collection = self.access_collection(data_id)
        temp_extent = collection.extent.temporal.intervals[0]
        temp_extent_str = [None, None]
        if temp_extent[0] is not None:
            temp_extent_str[0] = convert_datetime2str(temp_extent[0])
        if temp_extent[1] is not None:
            temp_extent_str[1] = convert_datetime2str(temp_extent[1])
        return dict(
            bbox=collection.extent.spatial.bboxes[0],
            time_range=temp_extent_str,
        )

    def search_data(self, **search_params) -> Iterator[pystac.Item]:
        schema = self.get_search_params_schema()
        schema.validate_instance(search_params)
        return search_collections(self._catalog, **search_params)

    def get_search_params_schema(self) -> JsonObjectSchema:
        return JsonObjectSchema(
            properties=dict(**STAC_SEARCH_PARAMETERS_STACK_MODE),
            required=[],
            additional_properties=False,
        )

    def _get_gridmapping(
        self,
        bbox: list[float],
        spatial_res: float,
        crs: Union[str, pyproj.crs.CRS],
        tile_size: Union[int, tuple[int, int]] = TILE_SIZE,
    ) -> GridMapping:
        x_size = int((bbox[2] - bbox[0]) / spatial_res) + 1
        y_size = int(abs(bbox[3] - bbox[1]) / spatial_res) + 1
        return GridMapping.regular(
            size=(x_size, y_size),
            xy_min=(bbox[0] - spatial_res / 2, bbox[1] - spatial_res / 2),
            xy_res=spatial_res,
            crs=crs,
            tile_size=tile_size,
        )

    def _groupby_solar_day(self, items: list[pystac.Item]) -> dict:
        items = add_nominal_datetime(items)
        nested_dict = collections.defaultdict(lambda: collections.defaultdict(list))
        # group by date and processing baseline
        for idx, item in enumerate(items):
            date = item.properties["datetime_nominal"].date()
            proc_version = float(item.properties["processorVersion"])
            nested_dict[date][proc_version].append(idx)
        # if two processing baselines are available, take most recent one
        grouped = {}
        for date, proc_version_dic in nested_dict.items():
            proc_version = max(list(proc_version_dic.keys()))
            grouped[date] = nested_dict[date][proc_version]
        return grouped

    def _get_mosaic_timestamps(self, grouped: dict, items: list[pystac.Item]):
        grouped_new = {}
        for old_key, value in grouped.items():
            new_key = (
                items[value[0]].properties["datetime_nominal"].replace(tzinfo=None)
            )
            grouped_new[new_key] = value
        return grouped

    def _get_angles_from_item(self, item):
        bucket_name = "eodata"
        href = item.assets["granule_metadata"].href.replace("s3://eodata/", "")
        response = self._util.s3_boto.get_object(Bucket=bucket_name, Key=href)
        xml_content = response["Body"].read().decode("utf-8")
        xml_dict = xmltodict.parse(xml_content)
        return xml_dict


def _mosaic_first_non_nan(list_ds):
    dim = "dummy"
    ds = xr.concat(list_ds, dim=dim)
    ds = ds.drop_vars("crs")
    ds_mosaic = xr.Dataset()
    for key in ds:
        axis = ds[key].dims.index(dim)
        da_arr = ds[key].data
        nan_mask = da.isnan(da_arr)
        first_non_nan_index = (~nan_mask).argmax(axis=axis)
        da_arr_select = da.choose(first_non_nan_index, da_arr)
        ds_mosaic[key] = xr.DataArray(
            da_arr_select,
            dims=("y", "x"),
            coords={"y": ds.y, "x": ds.x},
        )
    ds_mosaic["crs"] = list_ds[0].crs
    return ds_mosaic


def _resample_in_space(ds, target_gm):
    ds = resample_in_space(ds, target_gm=target_gm, encode_cf=True)
    if "spatial_ref" in ds.coords:
        ds = ds.drop_vars("spatial_ref")
    if "x_bnds" in ds.coords:
        ds = ds.drop_vars("x_bnds")
    if "y_bnds" in ds.coords:
        ds = ds.drop_vars("y_bnds")
    if "transformed_x" in ds.data_vars:
        ds = ds.drop_vars("transformed_x")
    if "transformed_y" in ds.data_vars:
        ds = ds.drop_vars("transformed_y")
    return ds