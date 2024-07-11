# The MIT License (MIT)
# Copyright (c) 2024 by the xcube development team and contributors
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

import datetime
import itertools
import unittest
import urllib.request

import pytest
import requests
import xarray as xr
from xcube.core.mldataset import MultiLevelDataset
from xcube.core.store import (
    DatasetDescriptor,
    DataStoreError,
    MultiLevelDatasetDescriptor,
    new_data_store,
)
from xcube.util.jsonschema import JsonObjectSchema

from xcube_stac.constants import DATA_STORE_ID
from xcube_stac.utils import _get_formats_from_item
from xcube_stac.accessor import HttpsDataAccessor, S3DataAccessor

SKIP_HELP = (
    "Skipped, because server is not running:"
    " $ xcube serve2 -vvv -c examples/serve/demo/config.yml"
)
SERVER_URL = "http://localhost:8080"
SERVER_ENDPOINT_URL = f"{SERVER_URL}/s3"


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
            "collections/sentinel-1-grd/items/"
            "S1A_EW_GRDM_1SDV_20240711T073732_20240711T073811_054710_06A929"
        )
        self.data_id_time_range = (
            "lcv_blue_landsat.glad.ard/lcv_blue_landsat.glad.ard_1999.12.02"
            "..2000.03.20/lcv_blue_landsat.glad.ard_1999.12.02..2000.03.20.json"
        )
        self.data_id_netcdf = (
            "collections/S5P_TROPOMI_L3_P1D_CF/items/"
            "S5P_DLR_NRTI_01_040201_L3_CF_20240619?f=application%2Fgeo%2Bjson"
        )

    @pytest.mark.vcr()
    def test_get_data_store_params_schema(self):
        store = new_data_store(DATA_STORE_ID, url=self.url_searchable)
        schema = store.get_data_store_params_schema()
        self.assertIsInstance(schema, JsonObjectSchema)
        self.assertIn("url", schema.properties)
        self.assertIn("storage_options_s3", schema.properties)
        self.assertIn("url", schema.required)

    @pytest.mark.vcr()
    def test_get_data_types(self):
        store = new_data_store(DATA_STORE_ID, url=self.url_searchable)
        self.assertEqual(("dataset", "mldataset"), store.get_data_types())

    @pytest.mark.vcr()
    def test_get_data_types_for_data(self):
        store = new_data_store(DATA_STORE_ID, url=self.url_nonsearchable)
        self.assertEqual(
            ("mldataset", "dataset"),
            store.get_data_types_for_data(self.data_id_nonsearchable),
        )
        store = new_data_store(DATA_STORE_ID, url=self.url_netcdf)
        self.assertEqual(
            ("dataset",),
            store.get_data_types_for_data(self.data_id_netcdf),
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
        item = store._access_item(data_ids[0])
        formats = _get_formats_from_item(item)
        self.assertEqual(["geotiff"], formats)

    @pytest.mark.vcr()
    def test_get_data_ids_include_attrs(self):
        store = new_data_store(DATA_STORE_ID, url=self.url_searchable)
        include_attrs = ["id", "bbox", "geometry", "properties", "links", "assets"]
        data_id, attrs = next(store.get_data_ids(include_attrs=include_attrs))
        self.assertEqual(self.data_id_searchable, data_id)
        self.assertCountEqual(include_attrs, list(attrs.keys()))

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
        self.assertEqual(opener_ids, store.get_data_opener_ids())
        opener_ids = (
            "mldataset:geotiff:https",
            "mldataset:levels:https",
            "mldataset:geotiff:s3",
            "mldataset:levels:s3",
        )
        self.assertEqual(opener_ids, store.get_data_opener_ids(data_type="mldataset"))

    @pytest.mark.vcr()
    def test_get_data_opener_ids_optional_args(self):
        store = new_data_store(DATA_STORE_ID, url=self.url_nonsearchable)
        with self.assertRaises(DataStoreError) as cm:
            store.get_data_opener_ids(data_id="wrong_data_id")
        self.assertEqual(
            "Data resource 'wrong_data_id' is not available.",
            f"{cm.exception}",
        )
        with self.assertRaises(DataStoreError) as cm:
            store.get_data_opener_ids(data_type=str)
        self.assertEqual(
            "Data type must be 'dataset' or 'mldataset', but got <class 'str'>.",
            f"{cm.exception}",
        )

    @pytest.mark.vcr()
    def test_get_open_data_params_schema(self):
        store = new_data_store(DATA_STORE_ID, url=self.url_nonsearchable)
        schema = store.get_open_data_params_schema()
        # no optional arguments
        self.assertIsInstance(schema, JsonObjectSchema)
        self.assertIn("asset_names", schema.properties)
        self.assertIn("open_params_dataset_netcdf", schema.properties)
        self.assertIn("open_params_dataset_zarr", schema.properties)
        self.assertIn("open_params_dataset_geotiff", schema.properties)
        self.assertIn("open_params_mldataset_geotiff", schema.properties)
        self.assertIn("open_params_dataset_levels", schema.properties)
        self.assertIn("open_params_mldataset_levels", schema.properties)

        # test opener_id argument
        schema = store.get_open_data_params_schema(opener_id="dataset:zarr:https")
        self.assertIsInstance(schema, JsonObjectSchema)
        self.assertIn("asset_names", schema.properties)
        self.assertIn("open_params_dataset_zarr", schema.properties)
        self.assertNotIn("open_params_dataset_netcdf", schema.properties)
        self.assertCountEqual(
            [
                "log_access",
                "cache_size",
                "group",
                "chunks",
                "mask_and_scale",
                "decode_cf",
                "decode_times",
                "decode_coords",
                "drop_variables",
                "consolidated",
            ],
            schema.properties["open_params_dataset_zarr"].properties.keys(),
        )

        # test data_id argument
        schema = store.get_open_data_params_schema(data_id=self.data_id_nonsearchable)
        self.assertIn("asset_names", schema.properties)
        self.assertIn("open_params_dataset_geotiff", schema.properties)
        self.assertIn("open_params_mldataset_geotiff", schema.properties)
        self.assertNotIn("open_params_dataset_zarr", schema.properties)

    @pytest.mark.vcr()
    def test_open_data_tiff(self):
        store = new_data_store(DATA_STORE_ID, url=self.url_time_range)

        # open data without open_params
        ds = store.open_data(self.data_id_time_range, opener_id="dataset:geotiff:https")
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

        # open data with open_params
        mlds = store.open_data(self.data_id_time_range, asset_names=["blue_p25"])
        self.assertIsInstance(mlds, MultiLevelDataset)
        ds = mlds.base_dataset
        self.assertCountEqual(["blue_p25"], list(ds.data_vars))
        self.assertCountEqual([151000, 188000], [ds.sizes["y"], ds.sizes["x"]])
        self.assertCountEqual(
            [512, 512], [ds.chunksizes["x"][0], ds.chunksizes["y"][0]]
        )

        # open data where multiple assets are stored in one mldataset
        mlds = store.open_data(
            self.data_id_time_range, asset_names=["blue_p25", "blue_p75"]
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
                "data_radiometric_cloud_fraction",
                "data_radiometric_cloud_fraction_precision",
                "data_number_of_observations",
                "data_quality_flag",
            ],
            list(ds.data_vars),
        )
        self.assertCountEqual([1800, 3600], [ds.sizes["lat"], ds.sizes["lon"]])

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
                "Only 's3' and 'https' protocols are supported, not 'abfs'. The asset "
                "'surface_air_pressure' has a href 'abfs://era5/ERA5/2020/12/"
                "surface_air_pressure.zarr'. The item's url is given by "
                "'https://planetarycomputer.microsoft.com/api/stac/v1/collections/"
                "era5-pds/items/era5-pds-2020-12-an'."
            ),
            f"{cm.exception}",
        )

    # run server demo in xcube/examples/serve/demo by running
    # "xcube serve --verbose -c examples/serve/demo/config.yml" in the terminal
    @unittest.skipUnless(XCUBE_SERVER_IS_RUNNING, SKIP_HELP)
    def test_open_data_xcube_server(self):
        store = new_data_store(DATA_STORE_ID, url="http://127.0.0.1:8080/ogc")

        # open data in zarr format
        ds = store.open_data("collections/datacubes/items/local_ts")
        self.assertIsInstance(ds, xr.Dataset)
        self.assertCountEqual(
            [
                "analytic_c2rcc_flags",
                "analytic_conc_chl",
                "analytic_conc_tsm",
                "analytic_kd489",
                "analytic_lat_bnds",
                "analytic_lon_bnds",
                "analytic_quality_flags",
                "analytic_time_bnds",
            ],
            list(ds.data_vars),
        )
        self.assertCountEqual(
            [1000, 2000, 5], [ds.sizes["lat"], ds.sizes["lon"], ds.sizes["time"]]
        )
        # open data in zarr format with open_params
        open_params_dataset_zarr = dict(chunks={"time": 5, "lat": 128, "lon": 128})
        ds = store.open_data(
            "collections/datacubes/items/local_ts",
            open_params_dataset_zarr=open_params_dataset_zarr,
        )
        self.assertIsInstance(ds, xr.Dataset)
        self.assertCountEqual(
            [
                "analytic_c2rcc_flags",
                "analytic_conc_chl",
                "analytic_conc_tsm",
                "analytic_kd489",
                "analytic_lat_bnds",
                "analytic_lon_bnds",
                "analytic_quality_flags",
                "analytic_time_bnds",
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

        # open data in levels format
        ds = store.open_data("collections/datacubes/items/local")
        self.assertIsInstance(ds, xr.Dataset)
        self.assertCountEqual(
            [
                "analytic_c2rcc_flags",
                "analytic_chl_category",
                "analytic_chl_tsm_sum",
                "analytic_conc_chl",
                "analytic_conc_tsm",
                "analytic_kd489",
                "analytic_lat_bnds",
                "analytic_lon_bnds",
                "analytic_quality_flags",
                "analytic_time_bnds",
            ],
            list(ds.data_vars),
        )
        self.assertCountEqual(
            [1000, 2000, 5], [ds.sizes["lat"], ds.sizes["lon"], ds.sizes["time"]]
        )
        # open data store in tif format
        mlds = store.open_data(
            "collections/datacubes/items/cog_local", data_type="mldataset"
        )
        ds = mlds.base_dataset
        self.assertIsInstance(mlds, MultiLevelDataset)
        self.assertEqual(3, mlds.num_levels)
        self.assertIsInstance(ds, xr.Dataset)
        self.assertCountEqual(
            [
                "analytic_multires_band_1",
                "analytic_multires_band_2",
                "analytic_multires_band_3",
                "analytic_multires_spatial_ref",
            ],
            list(ds.data_vars),
        )
        self.assertCountEqual([343, 343], [ds.sizes["y"], ds.sizes["x"]])

    @pytest.mark.vcr()
    def test_open_data_wrong_opener_id(self):
        store = new_data_store(DATA_STORE_ID, url=self.url_nonsearchable)
        with self.assertRaises(DataStoreError) as cm:
            store.open_data(self.data_id_nonsearchable, opener_id="wrong_opener_id")
        self.assertEqual(
            "Data opener identifier must be one of ('dataset:netcdf:https', "
            "'dataset:zarr:https', 'dataset:geotiff:https', 'mldataset:geotiff:https', "
            "'dataset:levels:https', 'mldataset:levels:https', "
            "'dataset:netcdf:s3', 'dataset:zarr:s3', 'dataset:geotiff:s3', "
            "'mldataset:geotiff:s3', 'dataset:levels:s3', 'mldataset:levels:s3'), "
            "but got 'wrong_opener_id'.",
            f"{cm.exception}",
        )

    @pytest.mark.vcr()
    def test_search_data(self):
        store = new_data_store(DATA_STORE_ID, url=self.url_nonsearchable)
        descriptors = list(
            store.search_data(
                collections="zanzibar-collection",
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
    def test_search_data_time_range(self):
        store = new_data_store(DATA_STORE_ID, url=self.url_time_range)
        descriptors = list(
            store.search_data(
                collections=["lcv_blue_landsat.glad.ard"],
                bbox=[-10, 40, 40, 70],
                time_range=["2000-01-01", "2000-04-01"],
            )
        )

        expected_descriptors = [
            dict(
                data_id=(
                    "lcv_blue_landsat.glad.ard/lcv_blue_landsat.glad.ard_1999"
                    ".12.02..2000.03.20/lcv_blue_landsat.glad.ard_1999.12.02"
                    "..2000.03.20.json"
                ),
                data_type="dataset",
                bbox=[
                    -23.550818268711048,
                    24.399543432891665,
                    63.352379098951936,
                    77.69295185585888,
                ],
                time_range=("1999-12-02", "2000-03-20"),
            ),
            dict(
                data_id=(
                    "lcv_blue_landsat.glad.ard/lcv_blue_landsat.glad.ard_2000"
                    ".03.21..2000.06.24/lcv_blue_landsat.glad.ard_2000.03.21"
                    "..2000.06.24.json"
                ),
                data_type="dataset",
                bbox=[
                    -23.550818268711048,
                    24.399543432891665,
                    63.352379098951936,
                    77.69295185585888,
                ],
                time_range=("2000-03-21", "2000-06-24"),
            ),
        ]

        self.assertEqual(2, len(descriptors))
        for d in descriptors:
            self.assertIsInstance(d, DatasetDescriptor)
        self.assertEqual(expected_descriptors, [d.to_dict() for d in descriptors])

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

    @pytest.mark.vcr()
    def test_describe_data_assert_log(self):
        store = new_data_store(DATA_STORE_ID, url=self.url_netcdf)
        with self.assertLogs("xcube.stac", level="INFO") as cm:
            descriptor = store.describe_data(self.data_id_netcdf, data_type="mldataset")
        self.assertIsInstance(descriptor, DatasetDescriptor)
        self.assertEqual(1, len(cm.output))
        msg = (
            f"INFO:xcube.stac:The data ID {self.data_id_netcdf!r} contains not only "
            f"assets in geotiff format. Therefore, data_type is set to 'dataset'"
        )
        self.assertEqual(msg, str(cm.output[-1]))
        expected_descriptor = dict(
            data_id=self.data_id_netcdf,
            data_type="dataset",
            bbox=[-180.0, -90.0, 180.0, 90.0],
            time_range=("2024-06-19T00:00:00Z", None),
        )
        self.assertDictEqual(expected_descriptor, descriptor.to_dict())

    @pytest.mark.vcr()
    def test_get_search_params_schema(self):
        store = new_data_store(DATA_STORE_ID, url=self.url_nonsearchable)
        schema = store.get_search_params_schema()
        self.assertIsInstance(schema, JsonObjectSchema)
        self.assertIn("time_range", schema.properties)
        self.assertIn("bbox", schema.properties)
        self.assertIn("collections", schema.properties)

    @pytest.mark.vcr()
    def test_access_item_failed(self):
        store = new_data_store(DATA_STORE_ID, url=self.url_nonsearchable)
        with self.assertRaises(requests.exceptions.HTTPError) as cm:
            store._access_item(self.data_id_nonsearchable.replace("z", "s"))
        self.assertIn("404 Client Error: Not Found for url", f"{cm.exception}")

    @pytest.mark.vcr()
    def test_get_s3_accessor(self):
        store = new_data_store(DATA_STORE_ID, url=self.url_searchable)

        opener = store._get_s3_accessor(root="datasets", storage_options={})
        self.assertIsInstance(opener, S3DataAccessor)
        self.assertEqual("datasets", opener.root)

        with self.assertLogs("xcube.stac", level="DEBUG") as cm:
            opener2 = store._get_s3_accessor(root="datasets2", storage_options={})
        self.assertIsInstance(opener2, S3DataAccessor)
        self.assertEqual("datasets2", opener2.root)
        self.assertEqual(1, len(cm.output))
        msg = (
            "DEBUG:xcube.stac:The bucket 'datasets' of the S3 object storage "
            "changed to 'datasets2'. A new s3 data opener will be initialized."
        )
        self.assertEqual(msg, str(cm.output[-1]))

    @pytest.mark.vcr()
    def test_get_https_accessor(self):
        store = new_data_store(DATA_STORE_ID, url=self.url_searchable)

        opener = store._get_https_accessor(root="earth-search.aws.element84.com")
        self.assertIsInstance(opener, HttpsDataAccessor)
        self.assertEqual("earth-search.aws.element84.com", opener.root)

        with self.assertLogs("xcube.stac", level="DEBUG") as cm:
            opener2 = store._get_https_accessor(root="planetarycomputer.microsoft.com")
        self.assertIsInstance(opener2, HttpsDataAccessor)
        self.assertEqual("planetarycomputer.microsoft.com", opener2.root)
        self.assertEqual(1, len(cm.output))
        msg = (
            "DEBUG:xcube.stac:The root 'earth-search.aws.element84.com' of the "
            "https data opener changed to 'planetarycomputer.microsoft.com'. "
            "A new https data opener will be initialized."
        )
        self.assertEqual(msg, str(cm.output[-1]))
