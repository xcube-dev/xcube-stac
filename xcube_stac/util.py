from typing import Iterator, Union

import numpy as np
import pystac
import pystac_client.client
import s3fs
from xcube.core.store import DataStoreError
from xcube.util.jsonschema import JsonObjectSchema

from .accessor import S3DataAccessor
from .accessor import S3Sentinel2DataAccessor
from .constants import CDSE_SENITNEL_2_BANDS
from .constants import CDSE_SENTINEL_2_LEVEL_BAND_RESOLUTIONS
from .constants import MAP_CDSE_COLLECTION_FORMAT
from .constants import MLDATASET_FORMATS
from .constants import STAC_SEARCH_PARAMETERS
from .constants import STAC_SEARCH_PARAMETERS_CDSE
from .constants import STAC_OPEN_PARAMETERS
from .constants import STAC_OPEN_PARAMETERS_STACK_MODE
from .constants import STAC_OPEN_PARAMETERS_CDSE
from .constants import STAC_OPEN_PARAMETERS_CDSE_STACK_MODE
from ._href_parse import decode_href
from ._utils import get_format_id
from ._utils import get_format_from_path
from ._utils import is_valid_ml_data_type
from ._utils import list_assets_from_item
from ._utils import search_nonsearchable_catalog


class Util:

    def __init__(self):
        self.supported_protocols = ["s3", "https"]
        self.supported_format_ids = ["netcdf", "zarr", "geotiff"]
        self.odc_stac_driver = None
        self.fs = None
        self.schema_open_params = STAC_OPEN_PARAMETERS
        self.schema_open_params_stack = STAC_OPEN_PARAMETERS_STACK_MODE
        self.schema_search_params = STAC_SEARCH_PARAMETERS
        self.s3_accessor = S3DataAccessor

    def parse_item(self, item: pystac.Item, **open_params) -> pystac.Item:
        return item

    def parse_item_stack(self, item: pystac.Item, **open_params) -> pystac.Item:
        return self.parse_item(item, **open_params)

    def parse_items_stack(
        self, items: list[pystac.Item], **open_params
    ) -> list[pystac.Item]:
        return items

    def get_data_access_params(self, item: pystac.Item, **open_params) -> dict:
        assets = list_assets_from_item(
            item,
            asset_names=open_params.get("asset_names"),
            supported_format_ids=self.supported_format_ids,
        )
        data_access_params = {}
        for asset in assets:
            protocol, root, fs_path, storage_options = decode_href(asset.href)
            format_id = get_format_id(asset)
            data_access_params[asset.extra_fields["id"]] = dict(
                name=asset.extra_fields["id"],
                protocol=protocol,
                root=root,
                fs_path=fs_path,
                storage_options=storage_options,
                format_id=format_id,
                href=asset.href,
            )
        return data_access_params

    def get_protocols(self, item: pystac.Item, **open_params) -> list[str]:
        params = self.get_data_access_params(item, **open_params)
        return list(np.unique([params[key]["protocol"] for key in params]))

    def get_format_ids(self, item: pystac.Item, **open_params) -> list[str]:
        params = self.get_data_access_params(item, **open_params)
        format_ids = list(np.unique([params[key]["format_id"] for key in params]))
        return [
            format_id
            for format_id in format_ids
            if format_id in self.supported_format_ids
        ]

    def is_mldataset_available(self, item: pystac.Item, **open_params) -> bool:
        format_ids = self.get_format_ids(item, **open_params)
        return all(format_id in MLDATASET_FORMATS for format_id in format_ids)

    def get_search_params_schema(self) -> JsonObjectSchema:
        return JsonObjectSchema(
            properties=dict(**STAC_SEARCH_PARAMETERS),
            required=[],
            additional_properties=False,
        )

    def get_open_data_params_schema(self) -> JsonObjectSchema:
        return STAC_OPEN_PARAMETERS

    def get_open_data_params_schema_stack(self) -> JsonObjectSchema:
        return STAC_OPEN_PARAMETERS_STACK_MODE

    def search_data(
        self,
        catalog: Union[pystac.Catalog, pystac_client.client.Client],
        searchable: bool,
        **search_params,
    ) -> Iterator[pystac.Item]:
        return search_data(catalog, searchable, **search_params)


class XcubeUtil:

    def __init__(self):
        self.supported_protocols = ["s3"]
        self.supported_format_ids = ["zarr", "levels"]
        self.odc_stac_driver = None
        self.fs = None
        self.schema_open_params = STAC_OPEN_PARAMETERS
        self.schema_open_params_stack = STAC_OPEN_PARAMETERS_STACK_MODE
        self.schema_search_params = STAC_SEARCH_PARAMETERS
        self.s3_accessor = S3DataAccessor

    def parse_item(self, item: pystac.Item, **open_params) -> pystac.Item:
        return item

    def parse_item_stack(self, item: pystac.Item, **open_params) -> pystac.Item:
        return self.parse_item(item, **open_params)

    def parse_items_stack(
        self, items: list[pystac.Item], **open_params
    ) -> list[pystac.Item]:
        return items

    def get_data_access_params(self, item: pystac.Item, **open_params) -> dict:
        asset_names = open_params.get("asset_names")
        assets = list_assets_from_item(
            item,
            asset_names=asset_names,
            supported_format_ids=self.supported_format_ids,
        )
        opener_id_data_type = open_params.get("opener_id")
        if opener_id_data_type is not None:
            opener_id_data_type = opener_id_data_type.split(":")[0]
        if asset_names is None:
            if is_valid_ml_data_type(open_params.get("data_type")):
                assets = [assets[1]]
            elif is_valid_ml_data_type(opener_id_data_type):
                assets = [assets[1]]
            else:
                assets = [assets[0]]
        elif "analytic_multires" in asset_names and "analytic" in asset_names:
            raise DataStoreError(
                "Xcube server publishes data resources as 'dataset' and "
                "'mldataset' under the asset names 'analytic' and "
                "'analytic_multires'. Please select only one asset in "
                "<asset_names> when opening the data."
            )
        data_access_params = {}
        for asset in assets:
            protocol = asset.extra_fields["xcube:data_store_id"]
            data_store_params = asset.extra_fields["xcube:data_store_params"]
            root = data_store_params["root"]
            storage_options = data_store_params["storage_options"]
            fs_path = asset.extra_fields["xcube:open_data_params"]["data_id"]
            format_id = get_format_from_path(fs_path)
            data_access_params[asset.extra_fields["id"]] = dict(
                name=asset.extra_fields["id"],
                protocol=protocol,
                root=root,
                fs_path=fs_path,
                storage_options=storage_options,
                format_id=format_id,
                href=asset.href,
            )
        return data_access_params

    def get_protocols(self, item: pystac.Item, **open_params) -> list[str]:
        return ["s3"]

    def get_format_ids(self, item: pystac.Item, **open_params) -> list[str]:
        return ["zarr", "levels"]

    def is_mldataset_available(self, item: pystac.Item, **open_params) -> bool:
        return True

    def get_search_params_schema(self) -> JsonObjectSchema:
        return JsonObjectSchema(
            properties=dict(**STAC_SEARCH_PARAMETERS),
            required=[],
            additional_properties=False,
        )

    def search_data(
        self,
        catalog: Union[pystac.Catalog, pystac_client.client.Client],
        searchable: bool,
        **search_params,
    ) -> Iterator[pystac.Item]:
        return search_data(catalog, searchable, **search_params)


class CdseUtil:

    def __init__(self, storage_options_s3: dict):
        self.supported_protocols = ["s3"]
        self.supported_format_ids = ["netcdf", "zarr", "geotiff", "jp2"]
        self.fs = s3fs.S3FileSystem(
            anon=False,
            key=storage_options_s3["key"],
            secret=storage_options_s3["secret"],
            endpoint_url=storage_options_s3["client_kwargs"]["endpoint_url"],
        )
        self.odc_stac_driver = None
        self.schema_open_params = STAC_OPEN_PARAMETERS_CDSE
        self.schema_open_params_stack = STAC_OPEN_PARAMETERS_CDSE_STACK_MODE
        self.schema_search_params = STAC_SEARCH_PARAMETERS_CDSE
        self.s3_accessor = S3Sentinel2DataAccessor

    def parse_item(self, item: pystac.Item, **open_params) -> pystac.Item:
        processing_level = open_params.pop("processing_level", "L2A")
        open_params["bands"] = open_params.get(
            "bands", CDSE_SENITNEL_2_BANDS[processing_level]
        )
        href_base = item.assets["PRODUCT"].extra_fields["alternate"]["s3"]["href"][1:]
        res_want = open_params.get("resolution", 20)
        for band in open_params["bands"]:
            res_avail = CDSE_SENTINEL_2_LEVEL_BAND_RESOLUTIONS[processing_level][band]
            res_select = res_avail[np.argmin(abs(np.array(res_avail) - res_want))]
            hrefs = self.fs.glob(f"{href_base}/**/*_{band}_{res_select}m.jp2")
            assert len(hrefs) == 1, "No unique jp2 file found"
            href_mod = f"s3://{hrefs[0]}"
            item.assets[band] = pystac.Asset(
                href_mod,
                band,
                media_type="image/jp2",
                roles=["data"],
                extra_fields=dict(cdse=True),
            )
        return item

    def parse_item_stack(self, item: pystac.Item, **open_params) -> pystac.Item | None:
        processing_level = open_params.pop("processing_level", "L2A")
        processing_baseline = open_params.pop("processing_baseline", "5.00")
        if not self._is_processing_level_baseline(
            item,
            processing_level=processing_level,
            processing_baseline=processing_baseline,
        ):
            return None
        return self.parse_item(item, **open_params)

    def parse_items_stack(
        self, items: list[pystac.Item], **open_params
    ) -> list[pystac.Item]:
        filtered_items = []
        for item in items:
            parsed_item = self.parse_item_stack(item, **open_params)
            if parsed_item is not None:
                filtered_items.append(parsed_item)
        return filtered_items

    def get_data_access_params(self, item: pystac.Item, **open_params) -> dict:
        processing_level = open_params.pop("processing_level", "L2A")
        bands = open_params.get("bands", CDSE_SENITNEL_2_BANDS[processing_level])
        data_access_params = {}
        for band in bands:
            protocol = "s3"
            href_components = item.assets[band].href.split("/")
            root = href_components[0]
            instrument = href_components[3]
            format_id = MAP_CDSE_COLLECTION_FORMAT[instrument]
            fs_path = "/".join(href_components[1:])
            storage_options = {}
            data_access_params[band] = dict(
                name=band,
                protocol=protocol,
                root=root,
                fs_path=fs_path,
                storage_options=storage_options,
                format_id=format_id,
                href=item.assets[band].href,
            )
        return data_access_params

    def get_protocols(self, item: pystac.Item, **open_params) -> list[str]:
        return ["s3"]

    def get_format_ids(self, item: pystac.Item, **open_params) -> list[str]:
        return ["jp2"]

    def is_mldataset_available(self, item: pystac.Item, **open_params) -> bool:
        return True

    def _is_processing_level_baseline(
        self, item: pystac.Item, processing_level="L2A", processing_baseline="5.00"
    ):
        return (
            processing_level[1:] in item.properties["processingLevel"]
            and processing_baseline in item.properties["processorVersion"]
        )

    def search_data(
        self,
        catalog: Union[pystac.Catalog, pystac_client.client.Client],
        searchable: bool,
        **search_params,
    ) -> Iterator[pystac.Item]:
        processing_level = search_params.pop("processing_level", "L2A")
        processing_baseline = search_params.pop("processing_baseline", "5.00")
        items = search_data(catalog, searchable, **search_params)
        for item in items:
            if not self._is_processing_level_baseline(
                item,
                processing_level=processing_level,
                processing_baseline=processing_baseline,
            ):
                continue
            yield item


def search_data(
    catalog: Union[pystac.Catalog, pystac_client.client.Client],
    searchable: bool,
    **search_params,
) -> Iterator[pystac.Item]:
    if searchable:
        # rewrite to "datetime"
        search_params["datetime"] = search_params.pop("time_range", None)
        items = catalog.search(**search_params).items()
    else:
        items = search_nonsearchable_catalog(catalog, **search_params)
    return items
