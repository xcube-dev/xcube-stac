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
from .accessor.sen3 import S3Sentinel3DataAccessor
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


class StackMode:
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
