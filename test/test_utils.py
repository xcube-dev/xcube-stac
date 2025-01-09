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
import unittest

import dask.array as da
import pystac
import pyproj
import numpy as np
import xarray as xr
from xcube.core.store import DataStoreError

from xcube_stac._utils import (
    get_format_id,
    get_format_from_path,
    reproject_bbox,
    get_spatial_dims,
    merge_datasets,
    normalize_crs,
    convert_datetime2str,
    convert_str2datetime,
    do_bboxes_intersect,
    is_collection_in_time_range,
    is_item_in_time_range,
    update_dict,
)


class UtilsTest(unittest.TestCase):

    def test_get_format_id(self):
        asset = pystac.Asset(
            href="https://example.com/data/test.tif",
            media_type="image/tiff",
            roles=["data"],
            extra_fields=dict(id="test"),
        )
        self.assertEqual("geotiff", get_format_id(asset))
        asset = pystac.Asset(
            href="https://example.com/data/test.tif",
            roles=["data"],
            extra_fields=dict(id="test"),
        )
        self.assertEqual("geotiff", get_format_id(asset))
        asset = pystac.Asset(
            href="https://example.com/data/test.xml",
            title="Meta data",
            roles=["meta"],
            extra_fields=dict(id="test"),
        )
        with self.assertLogs("xcube.stac", level="DEBUG") as cm:
            format_id = get_format_id(asset)
        self.assertIsNone(format_id)
        self.assertEqual(1, len(cm.output))
        msg = "DEBUG:xcube.stac:No format_id found for asset 'Meta data'"
        self.assertEqual(msg, str(cm.output[-1]))

    def test_convert_datetime2str(self):
        dt = datetime.datetime(2024, 1, 1, 12, 00, 00)
        self.assertEqual("2024-01-01T12:00:00", convert_datetime2str(dt))
        dt = datetime.date(2024, 1, 1)
        self.assertEqual("2024-01-01", convert_datetime2str(dt))

    def test_convert_str2datetime(self):
        dt = datetime.datetime(2024, 1, 1, 12, 00, 00, tzinfo=datetime.timezone.utc)
        self.assertEqual(dt, convert_str2datetime("2024-01-01T12:00:00.000000Z"))
        self.assertEqual(dt, convert_str2datetime("2024-01-01T12:00:00"))

    def test_is_item_in_time_range(self):
        item1 = pystac.Item(
            "test_item",
            geometry=None,
            bbox=[0, 0, 1, 1],
            datetime=datetime.datetime(2024, 5, 1, 9, 19, 38),
            properties=dict(datetime="2024-05-02T09:19:38.000000Z"),
        )
        item2 = pystac.Item(
            "test_item",
            geometry=None,
            bbox=[0, 0, 1, 1],
            datetime=None,
            properties=dict(
                datetime="null",
                start_datetime="2023-12-02T09:19:38.543000Z",
                end_datetime="2024-05-02T09:19:38.543000Z",
            ),
        )
        item3 = pystac.Item(
            "test_item",
            geometry=None,
            bbox=[0, 0, 1, 1],
            datetime=datetime.datetime(2024, 5, 1, 9, 19, 38),
            properties=dict(),
        )

        item1_test_paramss = [
            ("2024-04-30", "2024-05-03", self.assertTrue),
            ("2024-04-26", "2024-05-02", self.assertFalse),
            ("2024-04-26", "2024-05-01", self.assertFalse),
        ]

        item2_test_paramss = [
            ("2024-05-05", "2024-05-08", self.assertFalse),
            ("2024-04-30", "2024-05-03", self.assertTrue),
            ("2024-04-26", "2024-04-29", self.assertTrue),
            ("2023-11-26", "2023-12-31", self.assertTrue),
            ("2023-11-26", "2023-11-30", self.assertFalse),
            ("2023-11-26", "2024-05-08", self.assertTrue),
        ]

        for time_start, time_end, fun in item1_test_paramss:
            fun(is_item_in_time_range(item1, time_range=[time_start, time_end]))

        for time_start, time_end, fun in item2_test_paramss:
            fun(is_item_in_time_range(item2, time_range=[time_start, time_end]))

        with self.assertRaises(DataStoreError) as cm:
            is_item_in_time_range(
                item3, time_range=[item1_test_paramss[0][0], item1_test_paramss[0][1]]
            )
        self.assertEqual(
            "The item`s property needs to contain either 'start_datetime' and "
            "'end_datetime' or 'datetime'.",
            f"{cm.exception}",
        )

    def test_is_collection_in_time_range(self):
        collection1 = pystac.Collection(
            "test_collection",
            description="Test description",
            extent=pystac.Extent(
                pystac.SpatialExtent(bboxes=[[-180, -90, 180, 90]]),
                pystac.TemporalExtent(
                    intervals=[
                        [
                            datetime.datetime(
                                2020, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc
                            ),
                            datetime.datetime(
                                2020, 2, 1, 12, 0, 0, tzinfo=datetime.timezone.utc
                            ),
                        ]
                    ]
                ),
            ),
        )
        collection2 = pystac.Collection(
            "test_collection",
            description="Test description",
            extent=pystac.Extent(
                pystac.SpatialExtent(bboxes=[[-180, -90, 180, 90]]),
                pystac.TemporalExtent(
                    intervals=[
                        [
                            datetime.datetime(
                                2020, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc
                            ),
                            None,
                        ]
                    ]
                ),
            ),
        )
        collection3 = pystac.Collection(
            "test_collection",
            description="Test description",
            extent=pystac.Extent(
                pystac.SpatialExtent(bboxes=[[-180, -90, 180, 90]]),
                pystac.TemporalExtent(
                    intervals=[
                        [
                            None,
                            datetime.datetime(
                                2020, 2, 1, 12, 0, 0, tzinfo=datetime.timezone.utc
                            ),
                        ]
                    ]
                ),
            ),
        )

        collection1_test_paramss = [
            ("2019-12-15", "2019-12-20", self.assertFalse),
            ("2019-12-25", "2020-02-15", self.assertTrue),
            ("2019-12-25", "2020-01-15", self.assertTrue),
            ("2020-01-12", "2020-01-15", self.assertTrue),
            ("2020-01-25", "2020-02-15", self.assertTrue),
            ("2020-02-25", "2020-03-27", self.assertFalse),
        ]

        collection2_test_paramss = [
            ("2019-12-15", "2019-12-20", self.assertFalse),
            ("2019-12-25", "2020-02-15", self.assertTrue),
            ("2019-12-25", "2020-01-15", self.assertTrue),
            ("2020-01-12", "2020-01-15", self.assertTrue),
            ("2020-01-25", "2020-02-15", self.assertTrue),
            ("2020-02-25", "2020-03-27", self.assertTrue),
        ]

        collection3_test_paramss = [
            ("2019-12-15", "2019-12-20", self.assertTrue),
            ("2019-12-25", "2020-02-15", self.assertTrue),
            ("2019-12-25", "2020-01-15", self.assertTrue),
            ("2020-01-12", "2020-01-15", self.assertTrue),
            ("2020-01-25", "2020-02-15", self.assertTrue),
            ("2020-02-25", "2020-03-27", self.assertFalse),
        ]

        for time_start, time_end, fun in collection1_test_paramss:
            fun(
                is_collection_in_time_range(
                    collection1, time_range=[time_start, time_end]
                )
            )

        for time_start, time_end, fun in collection2_test_paramss:
            fun(
                is_collection_in_time_range(
                    collection2, time_range=[time_start, time_end]
                )
            )

        for time_start, time_end, fun in collection3_test_paramss:
            fun(
                is_collection_in_time_range(
                    collection3, time_range=[time_start, time_end]
                )
            )

    def test_do_bboxes_intersect(self):
        item = pystac.Item(
            "test_item",
            geometry=None,
            bbox=[0, 0, 1, 1],
            datetime=datetime.datetime(2024, 1, 1, 12, 00, 00),
            properties={},
        )

        item_test_paramss = [
            (0, 0, 1, 1, self.assertTrue),
            (0.5, 0.5, 1.5, 1.5, self.assertTrue),
            (-0.5, -0.5, 0.5, 0.5, self.assertTrue),
            (1, 1, 2, 2, self.assertTrue),
            (2, 2, 3, 3, self.assertFalse),
        ]

        for west, south, east, north, fun in item_test_paramss:
            fun(do_bboxes_intersect(item.bbox, bbox=[west, south, east, north]))

    def test_get_format_from_path(self):
        path = "https://example/data/file.tif"
        self.assertEqual("geotiff", get_format_from_path(path))
        path = "https://example/data/file.zarr"
        self.assertEqual("zarr", get_format_from_path(path))

    def test_update_nested_dict(self):
        dic = dict(a=1, b=dict(c=3))
        dic_update = dict(d=1, b=dict(c=5, e=8))
        dic_expected = dict(a=1, d=1, b=dict(c=5, e=8))
        self.assertDictEqual(dic_expected, update_dict(dic, dic_update))

    def test_reproject_bbox(self):
        bbox_wgs84 = [2, 50, 3, 51]
        crs_wgs84 = "EPSG:4326"
        crs_3035 = "EPSG:3035"
        bbox_3035 = [
            3748675.952977144,
            3018751.225593612,
            3830472.135997862,
            3122243.2680214494,
        ]
        self.assertEqual(bbox_wgs84, reproject_bbox(bbox_wgs84, crs_wgs84, crs_wgs84))
        self.assertEqual(bbox_3035, reproject_bbox(bbox_3035, crs_3035, crs_3035))
        np.testing.assert_almost_equal(
            bbox_3035, reproject_bbox(bbox_wgs84, crs_wgs84, crs_3035)
        )
        np.testing.assert_almost_equal(
            bbox_wgs84, reproject_bbox(bbox_3035, crs_3035, crs_wgs84)
        )

    def test_normalize_crs(self):
        crs_str = "EPSG:4326"
        crs_pyproj = pyproj.CRS.from_string(crs_str)
        self.assertEqual(crs_pyproj, normalize_crs(crs_str))
        self.assertEqual(crs_pyproj, normalize_crs(crs_pyproj))

    def test_merge_datasets(self):
        ds1 = xr.Dataset()
        ds1["B01"] = xr.DataArray(
            data=da.ones((3, 3)),
            dims=("y", "x"),
            coords=dict(x=[1000, 1020, 1040], y=[1000, 1020, 1040]),
        )
        ds2 = xr.Dataset()
        ds2["B02"] = xr.DataArray(
            data=da.ones((5, 5)),
            dims=("y", "x"),
            coords=dict(
                x=[995, 1005, 1015, 1025, 1035],
                y=[995, 1005, 1015, 1025, 1035],
            ),
        )
        ds3 = xr.Dataset()
        ds3["B03"] = xr.DataArray(
            data=da.ones((5, 5)),
            dims=("y", "x"),
            coords=dict(
                x=[995, 1005, 1015, 1025, 1035],
                y=[995, 1005, 1015, 1025, 1035],
            ),
        )
        ds_list = [ds1, ds2, ds3]
        wkt = (
            'PROJCRS["ETRS89 / LAEA Europe",'
            'BASEGEOGCRS["ETRS89",'
            'DATUM["European Terrestrial Reference System 1989",'
            'ELLIPSOID["GRS 1980",6378137,298.257222101,LENGTHUNIT["metre",1]]]],'
            'CONVERSION["Europe Equal Area",'
            'METHOD["Lambert Azimuthal Equal Area"],'
            'PARAMETER["Latitude of natural origin",52,'
            'ANGLEUNIT["degree",0.0174532925199433]],'
            'PARAMETER["Longitude of natural origin",10,'
            'ANGLEUNIT["degree",0.0174532925199433]],'
            'PARAMETER["False easting",4321000,LENGTHUNIT["metre",1]],'
            'PARAMETER["False northing",3210000,LENGTHUNIT["metre",1]]],'
            "CS[Cartesian,2],"
            'AXIS["easting (X)",east,ORDER[1]],'
            'AXIS["northing (Y)",north,ORDER[2]],'
            'LENGTHUNIT["metre",1]]'
        )
        for ds in ds_list:
            ds["crs"] = xr.DataArray(
                data=0,
                attrs={
                    "long_name": "Coordinate Reference System",
                    "description": "WKT representation of EPSG:3035",
                    "grid_mapping_name": "lambert_azimuthal_equal_area",
                    "crs_wkt": wkt,
                },
            )
        ds_merged = merge_datasets(ds_list)
        ds_merged = ds_merged.drop_vars("crs")
        ds_merged_expected = xr.Dataset()
        ds_merged_expected["B01"] = ds3["B03"]
        ds_merged_expected["B02"] = ds3["B03"]
        ds_merged_expected["B03"] = ds3["B03"]
        xr.testing.assert_allclose(ds_merged_expected.B01, ds_merged.B01)

    def test_get_spatial_dims(self):
        ds = xr.Dataset()
        ds["var"] = xr.DataArray(
            data=np.ones((2, 2)), dims=("y", "x"), coords=dict(y=[0, 10], x=[0, 10])
        )
        self.assertEqual(("y", "x"), get_spatial_dims(ds))
        ds = xr.Dataset()
        ds["var"] = xr.DataArray(
            data=np.ones((2, 2)),
            dims=("lat", "lon"),
            coords=dict(lat=[0, 10], lon=[0, 10]),
        )
        self.assertEqual(("lat", "lon"), get_spatial_dims(ds))
        ds = xr.Dataset()
        ds["var"] = xr.DataArray(
            data=np.ones((2, 2)),
            dims=("dim_false0", "dim_false1"),
            coords=dict(dim_false0=[0, 10], dim_false1=[0, 10]),
        )
        with self.assertRaises(DataStoreError) as cm:
            get_spatial_dims(ds)
        self.assertEqual(
            "No spatial dimensions found in dataset.",
            f"{cm.exception}",
        )
