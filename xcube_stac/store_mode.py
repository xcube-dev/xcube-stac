# The MIT License (MIT)
# Copyright (c) 2024-2025 by the xcube development team and contributors
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
from collections.abc import Iterator

import pystac
import pystac_client.client
import requests
import xarray as xr
from xcube.core.gridmapping import GridMapping
from xcube.core.mldataset import MultiLevelDataset
from xcube.core.store import DataStoreError, DataTypeLike, new_data_store
from xcube.util.jsonschema import JsonObjectSchema

from .accessor.https import HttpsDataAccessor
from .accessor.s3 import S3DataAccessor
from .accessor.sen2 import S3Sentinel2DataAccessor
from .constants import (
    CDSE_STAC_URL,
    COLLECTION_PREFIX,
    LOG,
    STAC_OPEN_PARAMETERS,
    STAC_OPEN_PARAMETERS_STACK_MODE,
    STAC_SEARCH_PARAMETERS,
    STAC_SEARCH_PARAMETERS_STACK_MODE,
)
from .helper import Helper
from .mldataset.single_item import SingleItemMultiLevelDataset
from .stac_extension.raster import apply_offset_scaling
from .utils import (
    convert_datetime2str,
    get_data_id_from_pystac_object,
    is_valid_ml_data_type,
    merge_datasets,
    normalize_grid_mapping,
    rename_dataset,
    reproject_bbox,
    search_collections,
    search_items,
    update_dict,
)

_HTTPS_STORE = new_data_store("https")
_OPEN_DATA_PARAMETERS = {
    "dataset:netcdf": _HTTPS_STORE.get_open_data_params_schema(
        opener_id="dataset:netcdf:https"
    ),
    "dataset:zarr": _HTTPS_STORE.get_open_data_params_schema(
        opener_id="dataset:zarr:https"
    ),
    "dataset:geotiff": _HTTPS_STORE.get_open_data_params_schema(
        opener_id="dataset:geotiff:https"
    ),
    "mldataset:geotiff": _HTTPS_STORE.get_open_data_params_schema(
        opener_id="mldataset:geotiff:https"
    ),
    "dataset:jp2": _HTTPS_STORE.get_open_data_params_schema(
        opener_id="dataset:geotiff:https"
    ),
    "mldataset:jp2": _HTTPS_STORE.get_open_data_params_schema(
        opener_id="mldataset:geotiff:https"
    ),
    "dataset:levels": _HTTPS_STORE.get_open_data_params_schema(
        opener_id="dataset:levels:https"
    ),
    "mldataset:levels": _HTTPS_STORE.get_open_data_params_schema(
        opener_id="mldataset:levels:https"
    ),
}


class SingleStoreMode:
    """Implementations to access single STAC items"""

    def __init__(
        self,
        catalog: pystac.Catalog | pystac_client.client.Client,
        url_mod: str,
        searchable: bool,
        storage_options_s3: dict,
        helper: Helper,
    ):
        self._catalog = catalog
        self._url_mod = url_mod
        self._searchable = searchable
        self._storage_options_s3 = storage_options_s3
        self._https_accessor: HttpsDataAccessor | None = None
        self._s3_accessor: S3Sentinel2DataAccessor | S3DataAccessor | None = None
        self._helper = helper

    def access_item(self, data_id: str) -> pystac.Item:
        """Retrieves and parses a STAC item associated with the given data ID.

        Args:
            data_id: The identifier of the data item provided by this store.

        Returns:
            A `pystac.Item` object representing the STAC item.

        Raises:
            DataStoreError: If the item cannot be retrieved from the catalog or
            if the response cannot be parsed into a valid STAC item.
        """
        url = f"{self._url_mod}{data_id}"
        try:
            response = requests.get(url)
            response.raise_for_status()
        except requests.RequestException as e:
            raise DataStoreError(f"Failed to access item at {url}: {e}")

        try:
            return pystac.Item.from_dict(
                json.loads(response.text),
                href=url,
                root=self._catalog,
                preserve_dict=False,
            )
        except Exception as e:
            raise DataStoreError(f"Failed to parse item JSON at {url}: {e}")

    def get_data_ids(
        self, data_type: DataTypeLike = None
    ) -> Iterator[tuple[str, pystac.Item]]:
        """Yields tuples of data identifiers and corresponding STAC items from the
        catalog.

        Iterates over all items in the catalog recursively. If a specific data type
        is provided, it filters items based on dataset availability.

        Args:
            data_type: Optional data type to filter items.

        Yields:
            Tuples of (data_id, pystac.Item) where:
                - data_id: A string uniquely identifying the data item.
                - pystac.Item: The corresponding STAC item object.
        """
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
        data_opener_open_params = JsonObjectSchema(
            properties=self._get_open_params_data_opener(
                data_id=data_id, opener_id=opener_id
            ),
            required=[],
            additional_properties=False,
        )
        return JsonObjectSchema(
            properties={
                **STAC_OPEN_PARAMETERS,
                "data_opener_open_params": data_opener_open_params,
            },
            required=[],
            additional_properties=False,
        )

    def open_data(
        self,
        data_id: str,
        opener_id: str = None,
        data_type: DataTypeLike = None,
        **open_params,
    ) -> xr.Dataset | MultiLevelDataset:
        """Opens a dataset corresponding to the given data identifier corresponding to
        a single STAC item.

        Args:
            data_id: The unique identifier for the data to open.
            opener_id: Optional identifier specifying which data opener to use.
            data_type: Optional data type to influence how the dataset is opened.
            **open_params: Additional opening parameters.

        Returns:
            An xarray.Dataset or MultiLevelDataset constructed from the STAC item.

        Raises:
            ValidationError: If the open parameters do not conform to the schema.
            DataStoreError: If the data_id is invalid or the dataset cannot be opened.
        """
        schema = self.get_open_data_params_schema(data_id=data_id, opener_id=opener_id)
        schema.validate_instance(open_params)
        item = self.access_item(data_id)
        return self.build_dataset_from_item(
            item, opener_id=opener_id, data_type=data_type, **open_params
        )

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
        items = search_items(self._catalog, self._searchable, **search_params)
        return items

    def get_search_params_schema(self) -> JsonObjectSchema:
        return JsonObjectSchema(
            properties=STAC_SEARCH_PARAMETERS,
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
    ) -> xr.Dataset | MultiLevelDataset:
        """Builds one dataset from a given STAC item, handling asset opening,
        normalization, optional scaling, and merging.

        Args:
            item: The STAC item representing multiple datasets to be opened via
                their assets.
            opener_id: Optional identifier specifying which data opener to use.
            data_type: Optional data type to influence the returned dataset type.
            target_gm: Optional target GridMapping used for merging or aligning
                datasets.
            **open_params: Additional opening parameters.

        Returns:
            An xarray.Dataset or a MultiLevelDataset constructed from the STAC item.
        """
        access_params = self._helper.get_data_access_params(
            item, opener_id=opener_id, data_type=data_type, **open_params
        )
        list_ds_asset = []
        for asset_key, params in access_params.items():
            ds_asset = self.open_asset_dataset(
                params, opener_id, data_type, **open_params
            )
            if isinstance(ds_asset, xr.Dataset):
                ds_asset = normalize_grid_mapping(ds_asset)
                ds_asset = rename_dataset(ds_asset, params["name_origin"])
                if open_params.get("apply_scaling", False):
                    ds_asset[params["name_origin"]] = apply_offset_scaling(
                        ds_asset[params["name_origin"]], item, asset_key
                    )
            list_ds_asset.append(ds_asset)

        attrs = dict(
            stac_catalog_url=self._catalog.get_self_href(), stac_item_id=item.id
        )

        if all(isinstance(ds, MultiLevelDataset) for ds in list_ds_asset):
            ds = SingleItemMultiLevelDataset(
                list_ds_asset,
                access_params,
                target_gm=target_gm,
                open_params=open_params,
                attrs=attrs,
                item=item,
                s3_accessor=self._s3_accessor,
            )
        else:
            ds = merge_datasets(list_ds_asset, target_gm=target_gm)
            if open_params.get("angles_sentinel2", False):
                ds = self._s3_accessor.add_sen2_angles(item, ds, **open_params)
            ds.attrs = attrs
        return ds

    def open_asset_dataset(
        self,
        params: dict,
        opener_id: str = None,
        data_type: DataTypeLike = None,
        **open_params,
    ) -> xr.Dataset | MultiLevelDataset:
        """pens a dataset associated with a STAC asset.

        This method determines the appropriate data opener (based on `opener_id`,
        `data_type`, or asset format) and delegates to either an HTTPS or S3 accessor
        to open the dataset.

        Args:
            params: A dictionary containing access parameters for the asset. Expected
                to include keys like 'protocol', 'href', 'format_id', and 'name'.
            opener_id: Optional identifier for a specific data opener.
            data_type: Optional data type, used to determine the return type of
                the dataset.
            **open_params: Additional opening parameters passed to the underlying
                data opener.

        Returns:
            A dataset or a multi-level dataset.

        Raises:
            DataStoreError: If the protocol is not supported or an error occurs
            during dataset opening.
        """
        if opener_id is not None:
            key = ":".join(opener_id.split(":")[:2])
        elif data_type is not None:
            key = f"{data_type}:{params['format_id']}"
        else:
            key = f"dataset:{params['format_id']}"
        open_params_asset = open_params.get("data_opener_open_params", {}).get(key, {})

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
            raise DataStoreError(
                f"Only 's3' and 'https' protocols are supported, not "
                f"{params['protocol']!r}. The asset {params['name']!r} has a href "
                f"{params['href']!r}."
            )

        return ds_asset

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
                f"S3 object storage changed to {access_params['root']!r}. "
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
                f"https data opener changed to {access_params['root']!r}. "
                "A new https data opener will be initialized."
            )
            self._https_accessor = HttpsDataAccessor(access_params["root"])
        return self._https_accessor

    def _get_open_params_data_opener(
        self,
        data_id: str = None,
        opener_id: str = None,
    ) -> dict:
        """Retrieves applicable open parameters for data openers based on the given
        `opener_id` and/or `data_id`.

        The method returns a dictionary of data opener parameters relevant to a
        specific data type and format id. It prioritizes parameters for a specific
        opener if `opener_id` is provided, then attempts to match format IDs from the
        STAC item if `data_id` is provided. If neither match is found, it returns
        all known open parameters.

        Args:
            data_id: Optional identifier for a STAC item whose format(s) will be used
                to look up applicable open parameters.
            opener_id: Optional opener ID used to directly look up specific parameters.

        Returns:
            A dictionary of open parameters filtered according to `opener_id` and/or
            the formats present in the STAC item referenced by `data_id`.

        """
        properties = {}
        if opener_id is not None:
            key = ":".join(opener_id.split(":")[:2])
            properties[key] = _OPEN_DATA_PARAMETERS[key]
        if data_id is not None:
            item = self.access_item(data_id)
            for format_id in self._helper.list_format_ids(item):
                for key in _OPEN_DATA_PARAMETERS.keys():
                    if format_id == key.split(":")[-1]:
                        properties[key] = _OPEN_DATA_PARAMETERS[key]
        if not properties:
            properties = _OPEN_DATA_PARAMETERS
        return properties


class StackStoreMode(SingleStoreMode):
    """Provides functionality to access and combine stacked STAC items within a single
    collection. Multiple items are harmonized into a single dataset using mosaicking
    and stacking techniques.
    """

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
        """Yields data identifiers and their corresponding STAC collections.

        Args:
            data_type: Optional data type used to filter collections.

        Yields:
            Tuples containing the data ID (collection ID) and the corresponding STAC
            Collection object.
        """
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
        data_opener_open_params = JsonObjectSchema(
            properties=self._get_open_params_data_opener(
                data_id=data_id, opener_id=opener_id
            ),
            required=[],
            additional_properties=False,
        )
        return JsonObjectSchema(
            properties={
                **STAC_OPEN_PARAMETERS_STACK_MODE,
                "data_opener_open_params": data_opener_open_params,
            },
            required=["time_range", "bbox", "crs", "spatial_res"],
            additional_properties=False,
        )

    def open_data(
        self,
        data_id: str,
        opener_id: str = None,
        data_type: DataTypeLike = None,
        **open_params,
    ) -> xr.Dataset | MultiLevelDataset | None:
        """Open and return a dataset by searching, and harmonizing multiple STAC items
        from a specified collection.

        This method searches for STAC items in the given collection filtered by
        bounding box, time range, and query. The resulting items are then mosaicked and
        stacked into a single 3D dataset.

        Args:
            data_id (str): Identifier of the data collection.
            opener_id (str, optional): Identifier for the data opener to use.
                Defaults to None.
            data_type (DataTypeLike, optional): Data type hint influencing dataset handling.
                Defaults to None. "mldataset" is not implemented yet and raises a
                NotImplementedError.
            **open_params: Additional parameters for opening data. Must include at least:
                - bbox: bounding box coordinates in the specified CRS,
                - crs: coordinate reference system of the bbox,
                - time_range: temporal range filter,
                - query: optional additional query parameters.

        Returns:
            xr.Dataset or None: The combined dataset from matched STAC items,
            or None if no items were found.

        Raises:
            NotImplementedError: If `data_type` indicates a multilevel dataset (mldataset),
                                 which is not supported in stacking mode.

        Notes:
            - Bounding boxes of Sentinel-2 tiles crossing the antimeridian in the CDSE
              catalog are filtered out to exclude known catalog bugs.
        """
        schema = self.get_open_data_params_schema(data_id=data_id, opener_id=opener_id)
        schema.validate_instance(open_params)

        # search for items
        bbox_wgs84 = reproject_bbox(
            open_params["bbox"], open_params["crs"], "EPSG:4326"
        )
        items = list(
            search_items(
                self._catalog,
                self._searchable,
                collections=[data_id],
                bbox=bbox_wgs84,
                time_range=open_params["time_range"],
                query=open_params.get("query"),
            )
        )

        # Remove items with incorrect bounding boxes in the CDSE Sentinel-2 L2A catalog.
        # This issue primarily affects tiles that cross the antimeridian and has been
        # reported as a catalog bug. A single Sentinel-2 tile spans approximately
        # 110 km in width. Near the poles (up to 83°N), this corresponds to a bounding
        # box width of about 8°. To account for inaccuracies, we use a conservative
        # threshold of 20° to detect and exclude faulty items.
        if CDSE_STAC_URL in self._url_mod and data_id == "sentinel-2-l2a":
            items = [item for item in items if abs(item.bbox[2] - item.bbox[0]) < 20]

        if len(items) == 0:
            LOG.warn(
                f"No items found in collection {data_id!r} for the "
                f"parameters bbox {bbox_wgs84!r}, time_range "
                f"{open_params['time_range']!r} and "
                f"query {open_params.get('query', 'None')!r}."
            )
            return None

        if opener_id is None:
            opener_id = ""
        if is_valid_ml_data_type(data_type) or opener_id.split(":")[0] == "mldataset":
            raise NotImplementedError("mldataset not supported in stacking mode")
        else:
            item_params = self._helper.get_data_access_params(
                items[0], opener_id=opener_id, data_type=data_type, **open_params
            )
            asset_params = next(iter(item_params.values()))
            accessor = self._get_s3_accessor(asset_params)
            ds = self._helper.stack_items(items, accessor, **open_params)
            ds.attrs["stac_catalog_url"] = self._catalog.get_self_href()
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

    def search_data(self, **search_params) -> Iterator[pystac.Collection]:
        schema = self.get_search_params_schema()
        schema.validate_instance(search_params)
        return search_collections(self._catalog, **search_params)

    def get_search_params_schema(self) -> JsonObjectSchema:
        return JsonObjectSchema(
            properties=dict(**STAC_SEARCH_PARAMETERS_STACK_MODE),
            required=[],
            additional_properties=False,
        )
