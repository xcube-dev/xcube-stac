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

import json
from typing import Iterator, Union

import numpy as np
import pystac
import pystac_client.client
import requests
import tqdm
import xarray as xr
from xcube.core.mldataset import MultiLevelDataset
from xcube.core.store import DataStoreError, DataTypeLike, new_data_store
from xcube.util.jsonschema import JsonObjectSchema
from xcube.core.gridmapping import GridMapping

from .accessor import (
    HttpsDataAccessor,
    S3DataAccessor,
)
from .constants import LOG
from .constants import STAC_SEARCH_PARAMETERS_STACK_MODE
from .helper import Helper
from .constants import COLLECTION_PREFIX
from .constants import TILE_SIZE
from .mldataset import SingleItemMultiLevelDataset
from .stac_extension.raster import apply_offset_scaling
from ._utils import (
    merge_datasets,
    rename_dataset,
    convert_datetime2str,
    get_data_id_from_pystac_object,
    get_url_from_pystac_object,
    is_valid_ml_data_type,
    list_assets_from_item,
    reproject_bbox,
    search_collections,
    update_dict,
    get_gridmapping,
)
from .stack import groupby_solar_day
from .stack import mosaic_take_first

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
        helper: Helper,
    ):
        self._catalog = catalog
        self._url_mod = url_mod
        self._searchable = searchable
        self._storage_options_s3 = storage_options_s3
        self._helper = helper
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
                if not self._helper.is_mldataset_available(item):
                    continue
            data_id = get_data_id_from_pystac_object(item, catalog_url=self._url_mod)
            yield data_id, item

    def get_open_data_params_schema(
        self,
        data_id: str = None,
        opener_id: str = None,
    ) -> JsonObjectSchema:
        properties = self._get_open_params_data_opener(
            data_id=data_id, opener_id=opener_id
        )
        return JsonObjectSchema(
            properties=update_dict(
                self._helper.schema_open_params, properties, inplace=False
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
        ds = self.build_dataset_from_item(
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
        items = self._helper.search_items(
            self._catalog, self._searchable, **search_params
        )
        return items

    def get_search_params_schema(self) -> JsonObjectSchema:
        return JsonObjectSchema(
            properties=self._helper.schema_search_params,
            required=[],
            additional_properties=False,
        )

    def build_dataset_from_item(
        self,
        item: pystac.Item,
        opener_id: str = None,
        data_type: DataTypeLike = None,
        target_gm: GridMapping = None,
        **open_params,
    ) -> Union[xr.Dataset, MultiLevelDataset]:
        """Builds a dataset where the data variable names correspond
        to the asset keys. If the loaded data consists of multiple
        data variables, the variable name follows the structure
        '<asset_key>_<data_variable_name>'

        Args:
            item: item object
            opener_id: Data opener identifier. Defaults to None.
            data_type: Data type assigning the return value data type.
                May be given as type alias name, as a type, or as a
                :class:`xcube.core.store.DataType` instance.

        Returns:
            Dataset representation of the data resources identified
            by *data_id* and *open_params*.
        """
        parsed_item = self._helper.parse_item(item, **open_params)
        access_params = self._helper.get_data_access_params(
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

            # open data with respective xcube data opener
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

            if isinstance(ds_asset, xr.Dataset):
                ds_asset = rename_dataset(ds_asset, asset_key)
                if open_params.get("apply_scaling", False):
                    ds_asset = apply_offset_scaling(ds_asset, parsed_item, asset_key)
            list_ds_asset.append(ds_asset)

        attrs = dict(
            stac_catalog_url=self._catalog.get_self_href(), stac_item_id=item.id
        )

        if all(isinstance(ds, MultiLevelDataset) for ds in list_ds_asset):
            ds = SingleItemMultiLevelDataset(
                list_ds_asset,
                parsed_item,
                list(access_params.keys()),
                target_gm=target_gm,
                open_params=open_params,
                attrs=attrs,
            )
        else:
            ds = merge_datasets(list_ds_asset, target_gm=target_gm)
            ds.attrs = attrs

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
            self._s3_accessor = self._helper.s3_accessor(
                access_params["root"],
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
            self._s3_accessor = self._helper.s3_accessor(
                access_params["root"],
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

    def _get_open_params_data_opener(
        self,
        data_id: str = None,
        opener_id: str = None,
    ):
        properties = {}
        if opener_id is not None:
            key = "_".join(opener_id.split(":")[:2])
            key = f"open_params_{key}"
            properties[key] = _OPEN_DATA_PARAMETERS[key]
        if data_id is not None:
            item = self.access_item(data_id)
            for format_id in self._helper.get_format_ids(item):
                for key in _OPEN_DATA_PARAMETERS.keys():
                    if format_id == key.split("_")[-1]:
                        properties[key] = _OPEN_DATA_PARAMETERS[key]
        if not properties:
            properties = _OPEN_DATA_PARAMETERS
        return properties


class StackStoreMode(SingleStoreMode):
    """Implementations to access stacked STAC items within one collection"""

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
                if not self._helper.is_mldataset_available(item):
                    continue
            yield collection.id, collection

    def get_open_data_params_schema(
        self,
        data_id: str = None,
        opener_id: str = None,
    ) -> JsonObjectSchema:
        properties = self._get_open_params_data_opener(
            data_id=data_id, opener_id=opener_id
        )
        return JsonObjectSchema(
            properties=update_dict(
                self._helper.schema_open_params_stack, properties, inplace=False
            ),
            required=["time_range", "bbox", "crs", "spatial_res"],
            additional_properties=False,
        )

    def open_data(
        self,
        data_id: str,
        opener_id: str = None,
        data_type: DataTypeLike = None,
        **open_params,
    ) -> Union[xr.Dataset, MultiLevelDataset, None]:
        schema = self.get_open_data_params_schema(data_id=data_id, opener_id=opener_id)
        schema.validate_instance(open_params)

        # search for items
        bbox_wgs84 = reproject_bbox(
            open_params["bbox"], open_params["crs"], "EPSG:4326"
        )
        items = list(
            self._helper.search_items(
                self._catalog,
                self._searchable,
                collections=[data_id],
                bbox=bbox_wgs84,
                time_range=open_params["time_range"],
                query=open_params.get("query"),
            )
        )
        if len(items) == 0:
            LOG.warn(
                f"No items found in collection {data_id!r} for the "
                f"parameters bbox {bbox_wgs84!r}, time_range"
                f"{open_params["time_range"]!r} and "
                f"query {open_params.get("query", "None")!r}"
            )
            return None
        sorted(items, key=lambda item: item.properties.get("datetime"))

        # group items by date
        grouped_items = groupby_solar_day(items)

        if opener_id is None:
            opener_id = ""
        if "asset_names" not in open_params:
            assets = list_assets_from_item(
                next(iter(grouped_items.values())),
                supported_format_ids=self._helper.supported_format_ids,
            )
            open_params["asset_names"] = [asset.extra_fields["id"] for asset in assets]

        if is_valid_ml_data_type(data_type) or opener_id.split(":")[0] == "mldataset":
            raise NotImplementedError("mldataset not supported in stacking mode")
        else:
            ds = self.stack_items(grouped_items, **open_params)
            ds.attrs["stac_catalog_url"] = self._catalog.get_self_href()
            ds.attrs["stac_item_ids"] = dict(
                {
                    date.isoformat(): [item.id for item in items]
                    for date, items in grouped_items.items()
                }
            )
        return ds

    def stack_items(
        self,
        grouped_items: dict,
        opener_id: str = None,
        data_type: DataTypeLike = None,
        **open_params,
    ) -> xr.Dataset:
        target_gm = get_gridmapping(
            open_params["bbox"],
            open_params["spatial_res"],
            open_params["crs"],
            open_params.get("tile_size", TILE_SIZE),
        )
        ds_dates = []
        np_datetimes = []
        desc = "Stack tiles along time axis."
        for datetime, items_for_date in tqdm.tqdm(
            grouped_items.items(), total=len(grouped_items), desc=desc
        ):
            np_datetimes.append(np.datetime64(datetime).astype("datetime64[ns]"))
            list_ds_items = []
            for item in items_for_date:
                ds = self.build_dataset_from_item(
                    item,
                    opener_id=opener_id,
                    data_type=data_type,
                    target_gm=target_gm,
                    **open_params,
                )
                list_ds_items.append(ds)
            ds_mosaic = mosaic_take_first(list_ds_items)
            ds_dates.append(ds_mosaic)
        ds = xr.concat(ds_dates, dim="time")
        ds = ds.assign_coords(coords=dict(time=np_datetimes))
        if "crs" in ds:
            ds = ds.drop_vars("crs")
            ds["crs"] = ds_dates[0].crs
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
