import collections
from typing import Iterator, Union

import numpy as np
import pystac
import pystac_client.client
import rasterio.session
import s3fs
import xarray as xr
from xcube.core.store import DataStoreError
from xcube.util.jsonschema import JsonObjectSchema

from .accessor import S3DataAccessor
from .accessor import S3Sentinel2DataAccessor

from .constants import MAP_CDSE_COLLECTION_FORMAT
from .constants import MLDATASET_FORMATS
from .constants import STAC_SEARCH_PARAMETERS
from .constants import STAC_SEARCH_PARAMETERS_STACK_MODE
from .constants import STAC_OPEN_PARAMETERS
from .constants import STAC_OPEN_PARAMETERS_STACK_MODE
from .constants import SCHEMA_SPATIAL_RES
from .constants import SCHEMA_PROCESSING_LEVEL
from .constants import SCHEMA_COLLECTIONS
from .sen2.constants import CDSE_SENITNEL_2_BANDS
from .sen2.constants import CDSE_SENTINEL_2_LEVEL_BAND_RESOLUTIONS
from .sen2.constants import CDSE_SENITNEL_2_SCALE
from .sen2.constants import CDSE_SENITNEL_2_OFFSET_400
from .sen2.constants import CDSE_SENITNEL_2_NO_DATA
from .constants import LOG
from ._href_parse import decode_href
from ._utils import get_format_id
from ._utils import get_format_from_path
from ._utils import is_valid_ml_data_type
from ._utils import list_assets_from_item
from ._utils import search_items
from ._utils import search_nonsearchable_catalog


class Helper:

    def __init__(self):
        self.supported_protocols = ["s3", "https"]
        self.supported_format_ids = ["netcdf", "zarr", "geotiff"]
        self.schema_open_params = STAC_OPEN_PARAMETERS
        self.schema_open_params_stack = STAC_OPEN_PARAMETERS_STACK_MODE
        self.schema_search_params = STAC_SEARCH_PARAMETERS
        self.schema_search_params_stack = STAC_SEARCH_PARAMETERS_STACK_MODE
        self.s3_accessor = S3DataAccessor

    def parse_item(self, item: pystac.Item, **open_params) -> pystac.Item:
        return item

    def parse_items_stack(self, items: dict[list[pystac.Item]], **open_params) -> dict:
        parsed_items = {}
        for key, items in items.items():
            parsed_items[key] = [self.parse_item(item, **open_params) for item in items]
        return dict(parsed_items)

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

    def search_items(
        self,
        catalog: Union[pystac.Catalog, pystac_client.client.Client],
        searchable: bool,
        **search_params,
    ) -> Iterator[pystac.Item]:
        return search_items(catalog, searchable, **search_params)

    def apply_offset_scaling(
        self,
        ds: xr.Dataset,
        items: dict,
    ) -> xr.Dataset:
        """This function applies scaling of the data and fills no-data pixel
        with np.nan.

        Args:
            ds: dataset
            items: item object or list of item objects (depending on stack-mode
                equal to False and True, respectively.)

        Returns:
            Dataset where scaling and filling nodata values are applied.
        """
        if isinstance(items, pystac.Item):
            items = [items]

        if items[0].ext.has("raster"):
            for data_varname in ds.data_vars.keys():
                scale = np.ones(len(items))
                offset = np.zeros(len(items))
                nodata_val = np.zeros(len(items))
                for i, item in enumerate(items):
                    raster_bands = item.assets[data_varname].extra_fields.get(
                        "raster:bands"
                    )
                    if not raster_bands:
                        break
                    nodata_val[i] = raster_bands[0].get("nodata", 0)
                    if "scale" in raster_bands[0]:
                        scale[i] = raster_bands[0]["scale"]
                    if "offset" in raster_bands[0]:
                        offset[i] = raster_bands[0]["offset"]

                nodata_val = np.unique(nodata_val)
                msg = (
                    "Items contain different values in the "
                    "asset's field 'raster:bands:nodata'"
                )
                assert len(nodata_val) == 1, msg
                nodata_val = nodata_val[0]
                ds[data_varname] = ds[data_varname].where(
                    ds[data_varname] != nodata_val
                )

                offset = np.unique(offset)
                msg = (
                    "Items contain different values in the "
                    "asset's field 'raster:bands:offset'"
                )
                assert len(offset) == 1, msg
                ds[data_varname] += offset[0]

                scale = np.unique(scale)
                msg = (
                    "Items contain different values in the "
                    "asset's field 'raster:bands:scale'"
                )
                assert len(scale) == 1, msg
                ds[data_varname] *= scale[0]

        return ds


class HelperXcube(Helper):

    def __init__(self):
        super().__init__()
        self.supported_protocols = ["s3"]
        self.supported_format_ids = ["zarr", "levels"]

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


class HelperCdse(Helper):

    def __init__(self, **storage_options_s3):
        super().__init__()
        self.supported_protocols = ["s3"]
        self.supported_format_ids = ["netcdf", "zarr", "geotiff", "jp2"]
        self.schema_open_params = dict(
            **STAC_OPEN_PARAMETERS, spatial_res=SCHEMA_SPATIAL_RES
        )
        self.schema_open_params_stack = dict(
            **STAC_OPEN_PARAMETERS_STACK_MODE, processing_level=SCHEMA_PROCESSING_LEVEL
        )
        self.schema_search_params = dict(
            **STAC_SEARCH_PARAMETERS_STACK_MODE,
            collections=SCHEMA_COLLECTIONS,
            processing_level=SCHEMA_PROCESSING_LEVEL,
        )
        self.accessor = S3Sentinel2DataAccessor(**storage_options_s3)

    def parse_item(self, item: pystac.Item, **open_params) -> pystac.Item:
        processing_level = open_params.pop("processing_level", "L2A")
        open_params["bands"] = open_params.get(
            "bands", CDSE_SENITNEL_2_BANDS[processing_level]
        )
        href_base = item.assets["PRODUCT"].extra_fields["alternate"]["s3"]["href"][1:]
        res_want = open_params.get("resolution")
        time_end = None
        for band in open_params["bands"]:
            res_avail = CDSE_SENTINEL_2_LEVEL_BAND_RESOLUTIONS[processing_level][band]
            res_select = res_avail[np.argmin(abs(np.array(res_avail) - res_want))]
            if time_end is None:
                hrefs = self.fs.glob(f"{href_base}/**/*_{band}_{res_select}m.jp2")
                assert len(hrefs) == 1, "No unique jp2 file found"
                href_mod = f"s3://{hrefs[0]}"
                time_end = hrefs[0].split("/IMG_DATA/")[0][-15:]
            else:
                id_parts = item.id.split("_")
                href_mod = (
                    f"s3://{href_base}/GRANULE/L2A_T{item.properties["tileId"]}_"
                    f"A{item.properties["orbitNumber"]:06}_{time_end}/IMG_DATA/"
                    f"R{res_select}m/T{item.properties["tileId"]}_"
                    f"{id_parts[2]}_{band}_{res_select}m.jp2"
                )
            item.assets[band] = pystac.Asset(
                href_mod,
                band,
                media_type="image/jp2",
                roles=["data"],
                extra_fields=dict(cdse=True),
            )
        item.assets["granule_metadata"] = pystac.Asset(
            f"s3://{href_base}/GRANULE/MTD_TL.xml",
            "granule_metadata",
            media_type="application/xml",
            roles=["metadata"],
            extra_fields=dict(cdse=True),
        )
        return item

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

    def search_items(
        self,
        catalog: Union[pystac.Catalog, pystac_client.client.Client],
        searchable: bool,
        **search_params,
    ) -> Iterator[pystac.Item]:
        processing_level = search_params.pop("processing_level", "L2A")
        if "sortby" not in search_params:
            search_params["sortby"] = "+datetime"
        items = search_items(catalog, searchable, **search_params)
        for item in items:
            if not processing_level[1:] in item.properties["processingLevel"]:
                continue
            yield item

    def apply_offset_scaling(self, ds: xr.Dataset, items: dict) -> xr.Dataset:
        """This function applies scaling of the data and fills no-data pixel with np.nan.

        Args:
            ds: dataset
            items: item object or list of item objects (depending on stack-mode equal to
                False and True, respectively.)

        Returns:
            Dataset where scaling and filling nodata values are applied.
        """
        if isinstance(items, pystac.Item):
            items = [items]

        for count, (date, items_for_date) in enumerate(items.items()):
            assert all(
                items_for_date[0].properties["processorVersion"]
                == items_for_date[idx].properties["processorVersion"]
                for idx in range(1, len(items_for_date))
            )
            for key in ds.data_vars.keys():
                if key == "SCL" or key == "crs":
                    continue
                if float(items_for_date[0].properties["processorVersion"]) >= 4.00:
                    ds[key] += CDSE_SENITNEL_2_OFFSET_400[key]
                ds[key] /= CDSE_SENITNEL_2_SCALE[key]
        return ds
