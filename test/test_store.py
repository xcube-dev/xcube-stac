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

import warnings
import unittest
import urllib.request

import pytest
import requests
import xarray as xr
from xcube.core.store import DatasetDescriptor, DataStoreError
from xcube.core.store.store import new_data_store
from xcube.util.jsonschema import JsonObjectSchema

from xcube_stac.constants import DATA_STORE_ID
from xcube_stac.opener import HttpsDataOpener, S3DataOpener

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
            "S1A_EW_GRDM_1SDH_20240625T111656_20240625T111758_054479_06A12E"
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
        store = new_data_store(DATA_STORE_ID, url=self.url_nonsearchable)
        self.assertTrue(store.has_data(self.data_id_nonsearchable))
        self.assertFalse(store.has_data(self.data_id_nonsearchable, data_type=str))

    @pytest.mark.vcr()
    def test_get_data_opener_ids(self):
        store = new_data_store(DATA_STORE_ID, url=self.url_nonsearchable)
        opener_ids = (
            "dataset:netcdf:https",
            "dataset:zarr:https",
            "dataset:geotiff:https",
            "mldataset:geotiff:https",
            "dataset:netcdf:s3",
            "dataset:zarr:s3",
            "dataset:geotiff:s3",
            "mldataset:geotiff:s3",
        )
        self.assertEqual(opener_ids, store.get_data_opener_ids())

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
        # no optional arguments
        store = new_data_store(DATA_STORE_ID, url=self.url_nonsearchable)
        schema = store.get_open_data_params_schema()
        self.assertIsInstance(schema, JsonObjectSchema)
        self.assertIn("asset_names", schema.properties)

        # optional arguments such that warning of
        # multiple data formats will be triggered
        store = new_data_store(DATA_STORE_ID, url=self.url_netcdf)
        with warnings.catch_warnings(record=True) as w:
            schema = store.get_open_data_params_schema(self.data_id_netcdf)
            self.assertIsInstance(schema, JsonObjectSchema)
            self.assertIn("asset_names", schema.properties)
            self.assertEqual(1, len(w))
            warn_msg = (
                f"The data ID '{self.data_id_netcdf}' contains the formats "
                "['geotiff' 'netcdf']. Please, do not specify 'opener_id' as "
                "multiple openers will be used."
            )
            self.assertEqual(warn_msg, str(w[-1].message))

        # optional arguments such that warning of
        # wrong opener_id will be returned
        store = new_data_store(DATA_STORE_ID, url=self.url_searchable)
        with warnings.catch_warnings(record=True) as w:
            schema = store.get_open_data_params_schema(
                self.data_id_searchable, opener_id="dataset:netcdf:https"
            )
            self.assertIsInstance(schema, JsonObjectSchema)
            self.assertIn("asset_names", schema.properties)
            self.assertEqual(1, len(w))
            warn_msg = (
                f"The data ID '{self.data_id_searchable}' contains the format "
                "'geotiff', but 'opener_id' is set to 'dataset:netcdf:https'. The "
                "'opener_id' will be changed in the open_data method."
            )
            self.assertEqual(warn_msg, str(w[-1].message))

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
        ds = store.open_data(self.data_id_time_range, asset_names=["blue_p25"])
        self.assertCountEqual(["blue_p25"], list(ds.data_vars))
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
            _ = store.open_data(data_id)
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

    # run server example in xcube-stac/examples/xcube_server by running
    # "xcube serve --verbose -c examples/xcube_server/config.yml" in the terminal
    @unittest.skipUnless(XCUBE_SERVER_IS_RUNNING, SKIP_HELP)
    def test_open_data_xcube_server_zarr(self):
        store = new_data_store(DATA_STORE_ID, url="http://127.0.0.1:8080/ogc")

        # open data store in zarr format
        ds = store.open_data("collections/datacubes/items/zarr_file")
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
        # open data store in levels format
        ds = store.open_data("collections/datacubes/items/levels_file")
        self.assertIsInstance(ds, xr.Dataset)
        self.assertCountEqual(
            [
                "analytic_c2rcc_flags",
                "analytic_conc_chl",
                "analytic_conc_tsm",
                "analytic_kd489",
                "analytic_quality_flags",
            ],
            list(ds.data_vars),
        )
        self.assertCountEqual(
            [1000, 2000, 5], [ds.sizes["lat"], ds.sizes["lon"], ds.sizes["time"]]
        )
        # open data store in tif format
        ds = store.open_data("collections/datacubes/items/geotiff_file")
        self.assertIsInstance(ds, xr.Dataset)
        self.assertCountEqual(
            [
                "analytic_band_1",
                "analytic_band_2",
                "analytic_band_3",
                "analytic_spatial_ref",
            ],
            list(ds.data_vars),
        )
        self.assertCountEqual([1387, 1491], [ds.sizes["y"], ds.sizes["x"]])
        # open data store in cloud-optimized tif format
        ds = store.open_data("collections/datacubes/items/cog_geotiff_file")
        self.assertIsInstance(ds, xr.Dataset)
        self.assertCountEqual(
            [
                "analytic_band_1",
                "analytic_band_2",
                "analytic_band_3",
                "analytic_spatial_ref",
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
            "'dataset:netcdf:s3', 'dataset:zarr:s3', 'dataset:geotiff:s3', "
            "'mldataset:geotiff:s3'), but got 'wrong_opener_id'.",
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
            time_range=["2019-04-23", None],
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
            "S2A_32TMT_20200305_0_L2A",
            "S2A_32TNT_20200305_0_L2A",
            "S2A_32UMU_20200305_0_L2A",
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
            time_range=["2020-03-05", None],
        )

        self.assertEqual(12, len(descriptors))
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
                time_range=["1999-12-02", "2000-03-20"],
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
                time_range=["2000-03-21", "2000-06-24"],
            ),
        ]

        self.assertEqual(2, len(descriptors))
        for d in descriptors:
            self.assertIsInstance(d, DatasetDescriptor)
        self.assertEqual(expected_descriptors, [d.to_dict() for d in descriptors])

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
    def test_get_s3_opener(self):
        store = new_data_store(DATA_STORE_ID, url=self.url_searchable)

        opener = store._get_s3_opener(
            root="datasets", opener_id="dataset:netcdf:s3", storage_options={}
        )
        self.assertIsInstance(opener, S3DataOpener)
        self.assertEqual("datasets", opener.root)

        with warnings.catch_warnings(record=True) as w:
            opener2 = store._get_s3_opener(
                root="datasets2", opener_id="dataset:netcdf:s3", storage_options={}
            )
            self.assertIsInstance(opener2, S3DataOpener)
            self.assertEqual("datasets2", opener2.root)
            self.assertEqual(1, len(w))
            warn_msg = (
                "The bucket 'datasets' of the S3 object storage changed to "
                "'datasets2'. A new s3 data opener will be initialized."
            )
            self.assertEqual(warn_msg, str(w[-1].message))

    @pytest.mark.vcr()
    def test_get_https_opener(self):
        store = new_data_store(DATA_STORE_ID, url=self.url_searchable)

        opener = store._get_https_opener(
            root="earth-search.aws.element84.com", opener_id="dataset:netcdf:https"
        )
        self.assertIsInstance(opener, HttpsDataOpener)
        self.assertEqual("earth-search.aws.element84.com", opener.root)

        with warnings.catch_warnings(record=True) as w:
            opener2 = store._get_https_opener(
                root="planetarycomputer.microsoft.com", opener_id="dataset:netcdf:s3"
            )
            self.assertIsInstance(opener2, HttpsDataOpener)
            self.assertEqual("planetarycomputer.microsoft.com", opener2.root)
            self.assertEqual(1, len(w))
            warn_msg = (
                "The root 'earth-search.aws.element84.com' of the https data opener "
                "changed to 'planetarycomputer.microsoft.com'. "
                "A new https data opener will be initialized."
            )
            self.assertEqual(warn_msg, str(w[-1].message))
