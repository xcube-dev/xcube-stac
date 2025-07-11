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

import itertools
import unittest
import urllib.request
from unittest.mock import patch

import pytest
import xarray as xr
from xcube.core.mldataset import MultiLevelDataset
from xcube.core.store import (
    DatasetDescriptor,
    DataStoreError,
    MultiLevelDatasetDescriptor,
    new_data_store,
)
from xcube.util.jsonschema import JsonObjectSchema

from xcube_stac.accessors.sen2 import SENITNEL2_L2A_BANDS
from xcube_stac.constants import (
    DATA_STORE_ID,
    DATA_STORE_ID_CDSE,
    DATA_STORE_ID_CDSE_ARDC,
    DATA_STORE_ID_XCUBE,
)
from xcube_stac.utils import reproject_bbox

from .sampledata import (
    sentinel_2_band_data_10m,
    sentinel_2_band_data_60m,
    sentinel_3_data,
    sentinel_3_geolocation_data,
)

SKIP_HELP = (
    "Skipped, because server is not running:"
    " $ xcube serve2 -vvv -c examples/serve/demo/config.yml"
)
SERVER_URL = "http://localhost:8080"
SERVER_ENDPOINT_URL = f"{SERVER_URL}/s3"
CDSE_CREDENTIALS = {
    "key": "xxx",
    "secret": "xxx",
}


def is_server_running() -> bool:
    # noinspection PyBroadException
    try:
        with urllib.request.urlopen(SERVER_URL, timeout=2.0) as response:
            response.read()
    except Exception:
        return False
    return 200 <= response.code < 400


XCUBE_SERVER_IS_RUNNING = is_server_running()


class StacDataStoreTest(unittest.TestCase):

    def setUp(self):
        self.url_nonsearchable = (
            "https://raw.githubusercontent.com/stac-extensions/"
            "label/main/examples/multidataset/catalog.json"
        )
        self.url_searchable = "https://earth-search.aws.element84.com/v1"
        self.url_time_range = (
            "https://s3.eu-central-1.wasabisys.com/stac/odse/catalog.json"
        )
        self.url_netcdf = "https://geoservice.dlr.de/eoc/ogc/stac/v1"
        self.data_id_nonsearchable = "zanzibar/znz001.json"
        self.data_id_searchable = (
            "collections/sentinel-1-grd/items/S1A_EW_GRDM_1SDH_20250709T083719_"
            "20250709T083754_060004_07745D"
        )
        self.data_id_time_range = (
            "lcv_blue_landsat.glad.ard/lcv_blue_landsat.glad.ard_1999.12.02"
            "..2000.03.20/lcv_blue_landsat.glad.ard_1999.12.02..2000.03.20.json"
        )
        self.data_id_netcdf = (
            "collections/S5P_TROPOMI_L3_P1D_CF/items/"
            "S5P_DLR_NRTI_01_040201_L3_CF_20240619?f=application%2Fgeo%2Bjson"
        )
        self.data_id_cdse_sen2 = (
            "collections/sentinel-2-l2a/items/"
            "S2A_MSIL2A_20241107T113311_N0511_R080_T32VKR_20241107T123948"
        )
        self.data_id_cdse_sen3 = (
            "collections/sentinel-3-syn-2-syn-ntc/items/S3B_SY_2_SYN____20250706T"
            "233058_20250706T233358_20250708T043306_0179_108_258_3420_ESA_O_NT_002"
        )

    @pytest.mark.vcr()
    def test_get_data_store_params_schema(self):
        store = new_data_store(DATA_STORE_ID, url=self.url_searchable)
        schema = store.get_data_store_params_schema()
        self.assertIsInstance(schema, JsonObjectSchema)
        self.assertIn("url", schema.properties)
        self.assertIn("anon", schema.properties)
        self.assertIn("key", schema.properties)
        self.assertIn("secret", schema.properties)
        self.assertIn("url", schema.required)

    @pytest.mark.vcr()
    def test_get_data_types(self):
        store = new_data_store(DATA_STORE_ID, url=self.url_searchable)
        self.assertEqual(("dataset", "mldataset"), store.get_data_types())
        # CDSE STAC API Sentinel-2
        store = new_data_store(
            DATA_STORE_ID_CDSE,
            key=CDSE_CREDENTIALS["key"],
            secret=CDSE_CREDENTIALS["secret"],
        )
        self.assertEqual(("dataset",), store.get_data_types())

    @pytest.mark.vcr()
    def test_get_data_types_for_data(self):
        store = new_data_store(DATA_STORE_ID, url=self.url_nonsearchable)
        self.assertEqual(
            ("dataset", "mldataset"),
            store.get_data_types_for_data(self.data_id_nonsearchable),
        )
        store = new_data_store(DATA_STORE_ID, url=self.url_netcdf)
        self.assertEqual(
            ("dataset",),
            store.get_data_types_for_data(self.data_id_netcdf),
        )
        # CDSE STAC API Sentinel-2
        store = new_data_store(
            DATA_STORE_ID_CDSE,
            key=CDSE_CREDENTIALS["key"],
            secret=CDSE_CREDENTIALS["secret"],
        )
        self.assertEqual(
            ("dataset",),
            store.get_data_types_for_data(self.data_id_cdse_sen2),
        )

    @unittest.skipUnless(XCUBE_SERVER_IS_RUNNING, SKIP_HELP)
    def test_get_data_types_for_data_xcube_server(self):
        store = new_data_store(DATA_STORE_ID_XCUBE, url="http://127.0.0.1:8080/ogc")
        self.assertEqual(
            ("dataset", "mldataset"),
            store.get_data_types_for_data("collections/datacubes/items/local_ts"),
        )

    @pytest.mark.vcr()
    def test_get_data_ids(self):
        store = new_data_store(DATA_STORE_ID, url=self.url_nonsearchable)
        data_ids = store.get_data_ids()
        data_ids_expected = [
            "zanzibar/znz001.json",
            "zanzibar/znz029.json",
            "spacenet-buildings/AOI_2_Vegas_img2636.json",
            "spacenet-buildings/AOI_3_Paris_img1648.json",
            "spacenet-buildings/AOI_4_Shanghai_img3344.json",
        ]
        self.assertCountEqual(data_ids_expected, data_ids)

    @pytest.mark.vcr()
    def test_get_data_ids_data_type(self):
        store = new_data_store(DATA_STORE_ID, url=self.url_netcdf)
        data_ids = store.get_data_ids(data_type="mldataset")
        data_ids = list(itertools.islice(data_ids, 1))
        self.assertEqual(1, len(data_ids))
        self.assertEqual(
            [
                "collections/ENMAP_HSI_L2A/items/ENMAP01-____L2A-DT0000140097_20250708"
                "T104407Z_001_V010502_20250709T035921Z?f=application%2Fgeo%2Bjson"
            ],
            data_ids,
        )

    @pytest.mark.vcr()
    def test_get_data_ids_cdse_ardc(self):
        store = new_data_store(
            DATA_STORE_ID_CDSE_ARDC,
            key=CDSE_CREDENTIALS["key"],
            secret=CDSE_CREDENTIALS["secret"],
        )
        data_ids = store.list_data_ids()
        self.assertCountEqual(
            ["sentinel-2-l2a", "sentinel-2-l1c", "sentinel-3-syn-2-syn-ntc"], data_ids
        )

    @pytest.mark.vcr()
    def test_get_data_ids_include_attrs(self):
        store = new_data_store(DATA_STORE_ID, url=self.url_searchable)
        include_attrs = ["id", "bbox", "links"]
        data_id, attrs = next(store.get_data_ids(include_attrs=include_attrs))
        self.assertEqual(self.data_id_searchable, data_id)
        self.assertCountEqual(include_attrs, list(attrs.keys()))

        data_id, attrs = next(store.get_data_ids(include_attrs=True))
        self.assertEqual(self.data_id_searchable, data_id)
        all_attrs = [
            "id",
            "bbox",
            "geometry",
            "properties",
            "links",
            "assets",
            "stac_extensions",
        ]
        self.assertCountEqual(all_attrs, list(attrs.keys()))

    @pytest.mark.vcr()
    def test_get_data_ids_optional_args_empty_args(self):
        store = new_data_store(DATA_STORE_ID, url=self.url_nonsearchable)
        data_id, attrs = next(store.get_data_ids(include_attrs=["dtype"]))
        self.assertEqual("zanzibar/znz001.json", data_id)
        self.assertFalse(attrs)

    @pytest.mark.vcr()
    def test_has_data(self):
        store = new_data_store(DATA_STORE_ID, url=self.url_netcdf)
        data_id = (
            "collections/ENMAP_HSI_L2A/items/ENMAP01-____L2A-DT0000080454_202406"
            "30T082045Z_001_V010402_20240701T122237Z?f=application%2Fgeo%2Bjson"
        )
        self.assertTrue(store.has_data(data_id))
        self.assertFalse(store.has_data(data_id, data_type=str))
        self.assertTrue(store.has_data(data_id, data_type="mldataset"))

    @pytest.mark.vcr()
    def test_get_data_opener_ids(self):
        store = new_data_store(DATA_STORE_ID, url=self.url_nonsearchable)
        opener_ids = (
            "dataset:netcdf:https",
            "dataset:zarr:https",
            "dataset:geotiff:https",
            "mldataset:geotiff:https",
            "dataset:levels:https",
            "mldataset:levels:https",
            "dataset:netcdf:s3",
            "dataset:zarr:s3",
            "dataset:geotiff:s3",
            "mldataset:geotiff:s3",
            "dataset:levels:s3",
            "mldataset:levels:s3",
        )
        self.assertCountEqual(opener_ids, store.get_data_opener_ids())
        opener_ids = (
            "mldataset:levels:https",
            "mldataset:geotiff:https",
            "mldataset:levels:s3",
            "mldataset:geotiff:s3",
        )
        self.assertCountEqual(
            opener_ids, store.get_data_opener_ids(data_type="mldataset")
        )
        opener_ids = (
            "mldataset:geotiff:s3",
            "dataset:geotiff:s3",
        )
        self.assertCountEqual(
            opener_ids, store.get_data_opener_ids(data_id=self.data_id_nonsearchable)
        )
        opener_ids = ("dataset:geotiff:s3",)
        self.assertCountEqual(
            opener_ids,
            store.get_data_opener_ids(
                data_id=self.data_id_nonsearchable, data_type="dataset"
            ),
        )

    @pytest.mark.vcr()
    def test_get_data_opener_ids_optional_args(self):
        store = new_data_store(DATA_STORE_ID, url=self.url_nonsearchable)
        with self.assertRaises(DataStoreError) as cm:
            store.get_data_opener_ids(data_id="wrong_data_id")
        self.assertEqual(
            "Failed to access STAC item at https://raw.githubusercontent.com/"
            "stac-extensions/label/main/examples/multidataset/wrong_data_id: "
            "404 Client Error: Not Found for url: https://raw.githubusercontent.com/"
            "stac-extensions/label/main/examples/multidataset/wrong_data_id",
            f"{cm.exception}",
        )
        with self.assertRaises(DataStoreError) as cm:
            store.get_data_opener_ids(data_type=str)
        self.assertEqual(
            (
                "Data type must be one of ('dataset', 'mldataset'), "
                "but got <class 'str'>."
            ),
            f"{cm.exception}",
        )

    @unittest.skipUnless(XCUBE_SERVER_IS_RUNNING, SKIP_HELP)
    def test_get_data_opener_ids_xcube_server(self):
        store = new_data_store(DATA_STORE_ID_XCUBE, url="http://127.0.0.1:8080/ogc")
        self.assertCountEqual(
            ("dataset:zarr:s3", "dataset:levels:s3", "mldataset:levels:s3"),
            store.get_data_opener_ids("collections/datacubes/items/local_ts"),
        )

    @pytest.mark.vcr()
    def test_get_open_data_params_schema(self):
        store = new_data_store(DATA_STORE_ID, url=self.url_nonsearchable)
        schema = store.get_open_data_params_schema()
        # no optional arguments
        self.assertIsInstance(schema, JsonObjectSchema)
        self.assertIn("asset_names", schema.properties)
        self.assertIn("apply_scaling", schema.properties)

        # test opener_id argument
        schema = store.get_open_data_params_schema(opener_id="dataset:zarr:https")
        self.assertIsInstance(schema, JsonObjectSchema)
        self.assertIn("asset_names", schema.properties)
        self.assertIn("apply_scaling", schema.properties)
        self.assertIn("group", schema.properties)
        self.assertIn("chunks", schema.properties)
        self.assertNotIn("overview_level", schema.properties)

        # test data_id argument
        schema = store.get_open_data_params_schema(data_id=self.data_id_nonsearchable)
        self.assertIn("asset_names", schema.properties)
        self.assertIn("apply_scaling", schema.properties)
        self.assertNotIn("group", schema.properties)
        self.assertIn("overview_level", schema.properties)

        # CDSE STAC API Sentinel-2
        store = new_data_store(
            DATA_STORE_ID_CDSE,
            key=CDSE_CREDENTIALS["key"],
            secret=CDSE_CREDENTIALS["secret"],
        )
        schema = store.get_open_data_params_schema(data_id=self.data_id_cdse_sen2)
        self.assertIsInstance(schema, JsonObjectSchema)
        self.assertIn("asset_names", schema.properties)
        self.assertIn("spatial_res", schema.properties)
        self.assertIn("apply_scaling", schema.properties)
        self.assertIn("add_angles", schema.properties)

        schema = store.get_open_data_params_schema(data_id=self.data_id_cdse_sen3)
        self.assertIsInstance(schema, JsonObjectSchema)
        self.assertIn("asset_names", schema.properties)
        self.assertIn("apply_rectification", schema.properties)

    def test_get_open_data_params_schema_cdse_ardc(self):
        store = new_data_store(
            DATA_STORE_ID_CDSE_ARDC,
            key=CDSE_CREDENTIALS["key"],
            secret=CDSE_CREDENTIALS["secret"],
        )
        data_ids = ("sentinel-2-l2a", "sentinel-2-l1c", "sentinel-3-syn-2-syn-ntc")
        for data_id in data_ids:
            schema = store.get_open_data_params_schema(data_id=data_id)
            self.assertIsInstance(schema, JsonObjectSchema)
            self.assertIn("asset_names", schema.properties)
            self.assertIn("time_range", schema.properties)
            self.assertIn("bbox", schema.properties)
            self.assertIn("crs", schema.properties)
            self.assertIn("spatial_res", schema.properties)
            self.assertIn("query", schema.properties)

    @pytest.mark.vcr()
    def test_open_data_tiff(self):
        store = new_data_store(DATA_STORE_ID, url=self.url_time_range)

        # open data without open_params
        ds = store.open_data(
            self.data_id_time_range,
            apply_scaling=True,
        )
        self.assertIsInstance(ds, xr.Dataset)
        self.assertCountEqual(
            ["blue_p50", "blue_p25", "blue_p75", "qa_f"], list(ds.data_vars)
        )
        self.assertCountEqual([151000, 188000], [ds.sizes["y"], ds.sizes["x"]])
        self.assertCountEqual(
            [512, 512], [ds.chunksizes["x"][0], ds.chunksizes["y"][0]]
        )
        self.assertDictEqual(
            dict(
                AREA_OR_POINT="Area",
                _FillValue=0,
                scale_factor=1.0,
                add_offset=0.0,
                grid_mapping="spatial_ref",
            ),
            ds.blue_p25.attrs,
        )

        # open data where multiple assets are stored in one mldataset
        mlds = store.open_data(
            self.data_id_time_range,
            asset_names=["blue_p25", "blue_p75"],
            data_type="mldataset",
        )
        self.assertIsInstance(mlds, MultiLevelDataset)
        ds = mlds.base_dataset
        self.assertCountEqual(["blue_p25", "blue_p75"], list(ds.data_vars))
        self.assertCountEqual([151000, 188000], [ds.sizes["y"], ds.sizes["x"]])
        self.assertCountEqual(
            [512, 512], [ds.chunksizes["x"][0], ds.chunksizes["y"][0]]
        )

    @pytest.mark.vcr()
    def test_open_data_netcdf(self):
        store = new_data_store(DATA_STORE_ID, url=self.url_netcdf)

        # open data without open_params
        ds = store.open_data(self.data_id_netcdf, asset_names=["data"])
        self.assertIsInstance(ds, xr.Dataset)
        self.assertCountEqual(
            [
                "radiometric_cloud_fraction",
                "radiometric_cloud_fraction_precision",
                "number_of_observations",
                "quality_flag",
            ],
            list(ds.data_vars),
        )
        self.assertCountEqual([1800, 3600], [ds.sizes["lat"], ds.sizes["lon"]])

    @pytest.mark.vcr()
    def test_open_data_failed(self):
        store = new_data_store(DATA_STORE_ID, url=self.url_nonsearchable)
        with self.assertRaises(DataStoreError) as cm:
            store.open_data(self.data_id_nonsearchable.replace("z", "s"))
        self.assertEqual(
            (
                "Failed to access STAC item at https://raw.githubusercontent.com/"
                "stac-extensions/label/main/examples/multidataset/sansibar/"
                "sns001.json: 404 Client Error: Not Found for url: https://raw."
                "githubusercontent.com/stac-extensions/label/main/examples/"
                "multidataset/sansibar/sns001.json"
            ),
            f"{cm.exception}",
        )

    @pytest.mark.vcr()
    def test_open_data_abfs(self):
        store = new_data_store(
            DATA_STORE_ID, url="https://planetarycomputer.microsoft.com/api/stac/v1"
        )

        # open data without open_params
        data_id = "collections/era5-pds/items/era5-pds-2020-12-an"
        with self.assertRaises(DataStoreError) as cm:
            _ = store.open_data(data_id, asset_names=["surface_air_pressure"])
        self.assertEqual(
            (
                "Neither 's3' nor 'https' could be derived from href "
                "'abfs://era5/ERA5/2020/12/surface_air_pressure.zarr'."
            ),
            f"{cm.exception}",
        )

    # run server demo in xcube/examples/serve/demo by running
    # "xcube serve --verbose -c examples/serve/demo/config.yml" in the terminal
    @unittest.skipUnless(XCUBE_SERVER_IS_RUNNING, SKIP_HELP)
    def test_open_data_xcube_server(self):
        store = new_data_store(DATA_STORE_ID_XCUBE, url="http://127.0.0.1:8080/ogc")

        # open data in zarr format
        ds = store.open_data("collections/datacubes/items/local_ts")
        self.assertIsInstance(ds, xr.Dataset)
        self.assertCountEqual(
            [
                "c2rcc_flags",
                "conc_chl",
                "conc_tsm",
                "kd489",
                "lat_bnds",
                "lon_bnds",
                "quality_flags",
                "time_bnds",
            ],
            list(ds.data_vars),
        )
        self.assertCountEqual(
            [1000, 2000, 5], [ds.sizes["lat"], ds.sizes["lon"], ds.sizes["time"]]
        )
        # open data in zarr format with open_params
        ds = store.open_data(
            "collections/datacubes/items/local_ts",
            chunks={"time": 5, "lat": 128, "lon": 128},
        )
        self.assertIsInstance(ds, xr.Dataset)
        self.assertCountEqual(
            [
                "c2rcc_flags",
                "conc_chl",
                "conc_tsm",
                "kd489",
                "lat_bnds",
                "lon_bnds",
                "quality_flags",
                "time_bnds",
            ],
            list(ds.data_vars),
        )
        self.assertCountEqual(
            [1000, 2000, 5], [ds.sizes["lat"], ds.sizes["lon"], ds.sizes["time"]]
        )
        self.assertCountEqual(
            [128, 128, 5],
            [
                ds.chunksizes["lat"][0],
                ds.chunksizes["lon"][0],
                ds.chunksizes["time"][0],
            ],
        )

        # open data as ml dataset
        mldss = [
            store.open_data(
                "collections/datacubes/items/cog_local", data_type="mldataset"
            ),
            store.open_data(
                "collections/datacubes/items/cog_local", opener_id="mldataset:levels:s3"
            ),
        ]
        for mlds in mldss:
            ds = mlds.base_dataset
            self.assertIsInstance(mlds, MultiLevelDataset)
            self.assertEqual(3, mlds.num_levels)
            self.assertIsInstance(ds, xr.Dataset)
            self.assertCountEqual(
                ["band_1", "band_2", "band_3", "spatial_ref"],
                list(ds.data_vars),
            )
            self.assertCountEqual([343, 343], [ds.sizes["y"], ds.sizes["x"]])

        # raise error when selecting "analytic" (asset linking to the dataset) and
        # "analytic_multires" (asset linking to the mldataset)
        with self.assertRaises(DataStoreError) as cm:
            store.open_data(
                "collections/datacubes/items/cog_local",
                asset_names=["analytic", "analytic_multires"],
            )
        self.assertEqual(
            "Xcube server publishes data resources as 'dataset' and "
            "'mldataset' under the asset names 'analytic' and "
            "'analytic_multires'. Please select only one asset in "
            "<asset_names> when opening the data.",
            f"{cm.exception}",
        )

    @pytest.mark.vcr()
    @patch("rioxarray.open_rasterio")
    def test_open_data_cdse_sen2(self, mock_rioxarray_open):
        mock_rioxarray_open.return_value = sentinel_2_band_data_10m()

        store = new_data_store(
            DATA_STORE_ID_CDSE,
            key=CDSE_CREDENTIALS["key"],
            secret=CDSE_CREDENTIALS["secret"],
        )

        data_id = (
            "collections/sentinel-2-l2a/items/S2A_MSIL2A_20200301T090901"
            "_N0500_R050_T35UPU_20230630T033416"
        )

        # open data as dataset
        ds = store.open_data(
            data_id=data_id,
            apply_scaling=True,
            add_angles=True,
        )
        self.assertIsInstance(ds, xr.Dataset)
        self.assertCountEqual(
            SENITNEL2_L2A_BANDS + ["solar_angle", "viewing_angle"],
            list(ds.data_vars),
        )
        self.assertCountEqual(
            [10980, 10980, 23, 23, 2, 12],
            [
                ds.sizes["y"],
                ds.sizes["x"],
                ds.sizes["angle_y"],
                ds.sizes["angle_x"],
                ds.sizes["angle"],
                ds.sizes["band"],
            ],
        )

    @pytest.mark.vcr()
    @patch("rioxarray.open_rasterio")
    def test_open_data_cdse_sen3(self, mock_rioxarray_open):
        mock_rioxarray_open.side_effect = [
            sentinel_3_geolocation_data(),
            sentinel_3_data(),
        ]

        store = new_data_store(
            DATA_STORE_ID_CDSE,
            key=CDSE_CREDENTIALS["key"],
            secret=CDSE_CREDENTIALS["secret"],
        )

        data_id = (
            "collections/sentinel-3-syn-2-syn-ntc/items/S3B_SY_2_SYN____20250706T233058_"
            "20250706T233358_20250708T043306_0179_108_258_3420_ESA_O_NT_002"
        )

        # open data as dataset without rectification
        ds = store.open_data(
            data_id=data_id,
            asset_names=["syn_Oa01_reflectance"],
            apply_rectification=False,
        )
        self.assertIsInstance(ds, xr.Dataset)
        self.assertCountEqual(["SDR_Oa01"], list(ds.data_vars))
        self.assertCountEqual([4091, 4865], [ds.sizes["y"], ds.sizes["x"]])
        self.assertEqual(2, ds.lat.ndim)
        self.assertEqual(2, ds.lon.ndim)

        # TODO add test with rectification
        # # open data as dataset with rectification
        # ds = store.open_data(
        #     data_id=data_id,
        #     asset_names=["syn_Oa01_reflectance"],
        #     apply_rectification=True,
        # )
        # self.assertIsInstance(ds, xr.Dataset)
        # self.assertCountEqual(["SDR_Oa01"], list(ds.data_vars))
        # self.assertCountEqual([4091, 4865], [ds.sizes["lat"], ds.sizes["lon"]])
        # self.assertEqual(1, ds.lat.ndim)
        # self.assertEqual(1, ds.lon.ndim)

    @pytest.mark.vcr()
    def test_open_data_cdse_ardc_no_items_found(self):
        store = new_data_store(
            DATA_STORE_ID_CDSE_ARDC,
            key=CDSE_CREDENTIALS["key"],
            secret=CDSE_CREDENTIALS["secret"],
        )

        # get warning, if no tiles are found
        with self.assertLogs("xcube.stac", level="WARNING") as cm:
            bbox_utm = [659574, 5892990, 659724, 5893140]
            ds = store.open_data(
                data_id="sentinel-2-l2a",
                bbox=bbox_utm,
                time_range=["2023-11-01", "2023-11-10"],
                query={"constellation": {"eq": "sentinel-3"}},
                spatial_res=10,
                crs="EPSG:32635",
                asset_names=["red", "green", "blue"],
                apply_scaling=True,
            )
        self.assertIsNone(ds)
        self.assertEqual(1, len(cm.output))
        msg = (
            "WARNING:xcube.stac:No items found in collection 'sentinel-2-l2a' "
            "for the parameters bbox (29.386939787289162, 53.1622888685164, "
            "29.389256239840616, 53.16368104066575), time_range ['2023-11-01', "
            "'2023-11-10'] and query {'constellation': {'eq': 'sentinel-3'}}."
        )
        self.assertEqual(msg, str(cm.output[-1]))

    @pytest.mark.vcr()
    @patch("rioxarray.open_rasterio")
    def test_open_data_cdse_sen2_ardc(self, mock_rioxarray_open):
        mock_rioxarray_open.return_value = sentinel_2_band_data_60m()
        store = new_data_store(
            DATA_STORE_ID_CDSE_ARDC,
            key=CDSE_CREDENTIALS["key"],
            secret=CDSE_CREDENTIALS["secret"],
        )

        # open Level-1C data in UTM crs
        bbox_wgs84 = [9.9, 53.1, 10.7, 53.5]
        crs_target = "EPSG:32632"
        bbox_utm = reproject_bbox(bbox_wgs84, "EPSG:4326", crs_target)
        ds = store.open_data(
            data_id="sentinel-2-l1c",
            bbox=bbox_utm,
            time_range=["2020-08-29", "2020-09-03"],
            spatial_res=60,
            crs=crs_target,
            asset_names=["B04"],
            apply_scaling=True,
            add_angles=True,
        )
        self.assertIsInstance(ds, xr.Dataset)

        self.assertCountEqual(
            ["B04", "solar_angle", "viewing_angle"],
            list(ds.data_vars),
        )
        self.assertEqual(
            [4, 759, 903, 11, 12, 2, 1],
            [
                ds.sizes["time"],
                ds.sizes["y"],
                ds.sizes["x"],
                ds.sizes["angle_y"],
                ds.sizes["angle_x"],
                ds.sizes["angle"],
                ds.sizes["band"],
            ],
        )
        self.assertEqual(
            [1, 759, 903, 11, 12, 2, 1],
            [
                ds.chunksizes["time"][0],
                ds.chunksizes["y"][0],
                ds.chunksizes["x"][0],
                ds.chunksizes["angle_y"][0],
                ds.chunksizes["angle_x"][0],
                ds.chunksizes["angle"][0],
                ds.chunksizes["band"][0],
            ],
        )

        # open data in UTM crs
        bbox_wgs84 = [9.9, 53.1, 10.7, 53.5]
        crs_target = "EPSG:32632"
        bbox_utm = reproject_bbox(bbox_wgs84, "EPSG:4326", crs_target)
        ds = store.open_data(
            data_id="sentinel-2-l2a",
            bbox=bbox_utm,
            time_range=["2020-08-29", "2020-09-03"],
            spatial_res=60,
            crs=crs_target,
            asset_names=["B04"],
            apply_scaling=True,
            add_angles=True,
        )
        self.assertIsInstance(ds, xr.Dataset)

        self.assertCountEqual(
            ["B04", "solar_angle", "viewing_angle"],
            list(ds.data_vars),
        )
        self.assertEqual(
            [4, 759, 903, 11, 12, 2, 1],
            [
                ds.sizes["time"],
                ds.sizes["y"],
                ds.sizes["x"],
                ds.sizes["angle_y"],
                ds.sizes["angle_x"],
                ds.sizes["angle"],
                ds.sizes["band"],
            ],
        )
        self.assertEqual(
            [1, 759, 903, 11, 12, 2, 1],
            [
                ds.chunksizes["time"][0],
                ds.chunksizes["y"][0],
                ds.chunksizes["x"][0],
                ds.chunksizes["angle_y"][0],
                ds.chunksizes["angle_x"][0],
                ds.chunksizes["angle"][0],
                ds.chunksizes["band"][0],
            ],
        )

        # open dataset in WGS84
        ds = store.open_data(
            data_id="sentinel-2-l2a",
            asset_names=["B04"],
            bbox=bbox_wgs84,
            time_range=["2020-07-26", "2020-08-01"],
            spatial_res=0.00054,
            crs="EPSG:4326",
            apply_scaling=True,
            add_angles=True,
        )
        self.assertIsInstance(ds, xr.Dataset)

        self.assertCountEqual(
            ["B04", "solar_angle", "viewing_angle"],
            list(ds.data_vars),
        )
        self.assertEqual(
            [4, 742, 1483, 10, 19, 2, 1],
            [
                ds.sizes["time"],
                ds.sizes["lat"],
                ds.sizes["lon"],
                ds.sizes["angle_lat"],
                ds.sizes["angle_lon"],
                ds.sizes["angle"],
                ds.sizes["band"],
            ],
        )
        self.assertEqual(
            [1, 742, 1483, 10, 19, 2, 1],
            [
                ds.chunksizes["time"][0],
                ds.chunksizes["lat"][0],
                ds.chunksizes["lon"][0],
                ds.chunksizes["angle_lat"][0],
                ds.chunksizes["angle_lon"][0],
                ds.chunksizes["angle"][0],
                ds.chunksizes["band"][0],
            ],
        )

        # catch NotImplementedError for multi-level dataset
        with self.assertRaises(DataStoreError) as cm:
            _ = store.open_data(
                data_id="sentinel-2-l2a",
                data_type="mldataset",
                asset_names=["B01", "B02", "B03"],
                bbox=bbox_wgs84,
                time_range=["2023-11-01", "2023-11-10"],
                spatial_res=0.00018,
                crs="EPSG:4326",
                apply_scaling=True,
                add_angles=True,
            )
        self.assertEqual(
            "Data type must be one of ('dataset',), but got 'mldataset'.",
            f"{cm.exception}",
        )

    # TODO add test for Sen3 ardc
    # @pytest.mark.vcr()
    # @patch("rioxarray.open_rasterio")
    # def test_open_data_cdse_sen3_ardc(self, mock_rioxarray_open):
    #     mock_rioxarray_open.side_effect = [
    #         sentinel_3_geolocation_data(),
    #         sentinel_3_data(),
    #         sentinel_3_geolocation_data(),
    #         sentinel_3_data(),
    #         sentinel_3_geolocation_data(),
    #         sentinel_3_data(),
    #         sentinel_3_geolocation_data(),
    #         sentinel_3_data(),
    #         sentinel_3_geolocation_data(),
    #         sentinel_3_data(),
    #     ]
    #     store = new_data_store(
    #         DATA_STORE_ID_CDSE_ARDC,
    #         key=CDSE_CREDENTIALS["key"],
    #         secret=CDSE_CREDENTIALS["secret"],
    #     )
    #
    #     ds = store.open_data(
    #         data_id="sentinel-3-syn-2-syn-ntc",
    #         bbox=[8, 52, 12, 55],
    #         time_range=["2020-07-31", "2020-08-01"],
    #         spatial_res=300 / 111320,  # meter in degree
    #         crs="EPSG:4326",
    #         asset_names=["syn_Oa01_reflectance"],
    #     )
    #     ds
    #     self.assertIsInstance(ds, xr.Dataset)
    #
    #     self.assertCountEqual(["syn_Oa01_reflectance"], list(ds.data_vars))
    #     self.assertEqual(
    #         [2, 1115, 1486],
    #         [ds.sizes["time"], ds.sizes["y"], ds.sizes["x"]],
    #     )
    #     self.assertEqual(
    #         [2, 1115, 1486],
    #         [ds.chunksizes["time"][0], ds.chunksizes["y"][0], ds.chunksizes["x"][0]],
    #     )

    @pytest.mark.vcr()
    def test_open_data_wrong_opener_id(self):
        self.maxDiff = None
        store = new_data_store(DATA_STORE_ID, url=self.url_nonsearchable)
        with self.assertRaises(DataStoreError) as cm:
            store.open_data(self.data_id_nonsearchable, opener_id="wrong_opener_id")
        self.assertEqual(
            "Data opener identifier must be one of ('dataset:netcdf:https', "
            "'dataset:zarr:https', 'dataset:geotiff:https', 'mldataset:geotiff:https', "
            "'dataset:levels:https', 'mldataset:levels:https', 'dataset:netcdf:s3', "
            "'dataset:zarr:s3', 'dataset:geotiff:s3', 'mldataset:geotiff:s3', "
            "'dataset:levels:s3', 'mldataset:levels:s3'), but got 'wrong_opener_id'.",
            f"{cm.exception}",
        )

    @pytest.mark.vcr()
    def test_search_data(self):
        store = new_data_store(DATA_STORE_ID, url=self.url_nonsearchable)
        descriptors = list(
            store.search_data(
                data_type="dataset",
                collections=["zanzibar-collection"],
                bbox=[39.28, -5.74, 39.31, -5.72],
                time_range=["2019-04-23", "2019-04-24"],
            )
        )

        expected_descriptor = dict(
            data_id="zanzibar/znz001.json",
            data_type="dataset",
            bbox=[
                39.28919876472999,
                -5.743028283012867,
                39.31302874892266,
                -5.722212794937691,
            ],
            time_range=("2019-04-23T00:00:00Z", None),
        )

        self.assertEqual(1, len(descriptors))
        self.assertIsInstance(descriptors[0], DatasetDescriptor)
        self.assertEqual(expected_descriptor, descriptors[0].to_dict())

    @pytest.mark.vcr()
    def test_search_data_searchable_catalog(self):
        store = new_data_store(DATA_STORE_ID, url=self.url_searchable)
        descriptors = list(
            store.search_data(
                data_type="dataset",
                collections=["sentinel-2-l2a"],
                bbox=[9, 47, 10, 48],
                time_range=["2020-03-01", "2020-03-05"],
            )
        )

        prefix = "collections/sentinel-2-l2a/items/"
        data_ids_expected = [
            "S2A_32TMT_20200305_1_L2A",
            "S2A_32TMT_20200305_0_L2A",
            "S2A_32TNT_20200305_1_L2A",
            "S2A_32TNT_20200305_0_L2A",
            "S2A_32UMU_20200305_1_L2A",
            "S2A_32UMU_20200305_0_L2A",
            "S2A_32UNU_20200305_1_L2A",
            "S2A_32UNU_20200305_0_L2A",
            "S2A_32TMT_20200302_1_L2A",
            "S2A_32TMT_20200302_0_L2A",
            "S2A_32TNT_20200302_1_L2A",
            "S2A_32TNT_20200302_0_L2A",
            "S2A_32UMU_20200302_1_L2A",
            "S2A_32UMU_20200302_0_L2A",
            "S2A_32UNU_20200302_1_L2A",
            "S2A_32UNU_20200302_0_L2A",
        ]
        data_ids_expected = [prefix + data_id for data_id in data_ids_expected]

        expected_descriptor = dict(
            data_id=data_ids_expected[0],
            data_type="dataset",
            bbox=[
                7.662878883910047,
                46.85818510451771,
                9.130456971519783,
                47.85361872923358,
            ],
            time_range=("2020-03-05T10:37:41.587000Z", None),
        )

        self.assertEqual(16, len(descriptors))
        for d in descriptors:
            self.assertIsInstance(d, DatasetDescriptor)
        self.assertCountEqual(data_ids_expected, [d.data_id for d in descriptors])
        self.assertEqual(expected_descriptor, descriptors[0].to_dict())

    @pytest.mark.vcr()
    def test_search_data_cdse_sentinel_2(self):
        store = new_data_store(
            DATA_STORE_ID_CDSE,
            key=CDSE_CREDENTIALS["key"],
            secret=CDSE_CREDENTIALS["secret"],
        )
        descriptors = list(
            store.search_data(
                collections=["sentinel-2-l2a"],
                bbox=[9.0, 47.0, 9.1, 47.1],
                time_range=["2020-07-01", "2020-07-05"],
            )
        )

        prefix = "collections/sentinel-2-l2a/items/"
        data_ids_expected = [
            "S2B_MSIL2A_20200705T101559_N0500_R065_T32TNT_20230530T175912",
            "S2B_MSIL2A_20200705T101559_N0500_R065_T32TMT_20230530T175912",
            "S2A_MSIL2A_20200703T103031_N0500_R108_T32TNT_20230613T212700",
            "S2A_MSIL2A_20200703T103031_N0500_R108_T32TMT_20230613T212700",
        ]

        data_ids_expected = [prefix + data_id for data_id in data_ids_expected]

        expected_descriptor = dict(
            data_id=data_ids_expected[0],
            data_type="dataset",
            bbox=[8.999733, 46.85664, 10.467277, 47.853702],
            time_range=("2020-07-05T10:15:59.024Z", "2020-07-05T10:15:59.024Z"),
        )

        for d in descriptors:
            self.assertIsInstance(d, DatasetDescriptor)
        self.assertCountEqual(data_ids_expected, [d.data_id for d in descriptors])
        selected_descriptor = [
            d for d in descriptors if d.data_id == data_ids_expected[0]
        ][0]
        self.assertEqual(expected_descriptor, selected_descriptor.to_dict())

    @pytest.mark.vcr()
    def test_search_data_cdse_ardc(self):
        store = new_data_store(
            DATA_STORE_ID_CDSE_ARDC,
            key=CDSE_CREDENTIALS["key"],
            secret=CDSE_CREDENTIALS["secret"],
        )
        descriptors = list(
            store.search_data(
                data_type="dataset",
                bbox=[9, 47, 10, 48],
                time_range=["2020-03-01", "2020-03-05"],
            )
        )

        expected_descriptor = dict(
            data_id="sentinel-1-global-mosaics",
            data_type="dataset",
            bbox=[-180, -90, 180, 90],
            time_range=("2020-01-01T00:00:00+00:00", None),
        )
        self.assertEqual(expected_descriptor, descriptors[0].to_dict())

    @pytest.mark.vcr()
    def test_describe_data(self):
        store = new_data_store(DATA_STORE_ID, url=self.url_netcdf)
        data_id = (
            "collections/D4H/items/S1A_IW_GRDH_1SDV_20231031T030757_"
            "20231031T030822_051003_062646_8C53?f=application%2Fgeo%2Bjson"
        )
        descriptor = store.describe_data(data_id, data_type="mldataset")
        expected_descriptor = dict(
            data_id=data_id,
            data_type="mldataset",
            num_levels=7,
            bbox=[
                36.211544470664904,
                -15.48028,
                37.86349678056911,
                -14.169830229695073,
            ],
            time_range=("2023-10-31T00:00:00.000+00:00", None),
        )
        self.assertIsInstance(descriptor, MultiLevelDatasetDescriptor)
        self.assertDictEqual(expected_descriptor, descriptor.to_dict())

    # run server demo in xcube/examples/serve/demo by running
    # "xcube serve --verbose -c examples/serve/demo/config.yml" in the terminal
    @unittest.skipUnless(XCUBE_SERVER_IS_RUNNING, SKIP_HELP)
    def test_describe_data_xcube_server(self):
        store = new_data_store(DATA_STORE_ID_XCUBE, url="http://127.0.0.1:8080/ogc")
        data_id = "collections/datacubes/items/local"
        descriptor = store.describe_data(data_id, data_type="mldataset")
        expected_descriptor = dict(
            data_id=data_id,
            data_type="mldataset",
            num_levels=3,
            bbox=[0, 50, 5, 52.5],
            time_range=("2017-01-16T10:09:21Z", "2017-01-30T10:46:33Z"),
        )
        self.assertIsInstance(descriptor, MultiLevelDatasetDescriptor)
        self.assertDictEqual(expected_descriptor, descriptor.to_dict())

    @pytest.mark.vcr()
    def test_get_search_params_schema(self):
        store = new_data_store(DATA_STORE_ID, url=self.url_nonsearchable)
        schema = store.get_search_params_schema()
        self.assertIsInstance(schema, JsonObjectSchema)
        self.assertIn("time_range", schema.properties)
        self.assertIn("bbox", schema.properties)
        self.assertIn("query", schema.properties)
        self.assertIn("collections", schema.properties)

    @pytest.mark.vcr()
    def test_get_search_params_schema_cdse_ardc(self):
        store = new_data_store(
            DATA_STORE_ID_CDSE_ARDC,
            key=CDSE_CREDENTIALS["key"],
            secret=CDSE_CREDENTIALS["secret"],
        )
        schema = store.get_search_params_schema()
        self.assertIsInstance(schema, JsonObjectSchema)
        self.assertIn("time_range", schema.properties)
        self.assertIn("bbox", schema.properties)
        self.assertNotIn("query", schema.properties)
        self.assertNotIn("collections", schema.properties)
