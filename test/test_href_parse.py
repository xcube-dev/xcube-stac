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

import unittest

from xcube.core.store import DataStoreError

from xcube_stac._href_parse import assert_aws_s3_bucket
from xcube_stac._href_parse import assert_aws_s3_region_name
from xcube_stac._href_parse import decode_href


class HrefParseTest(unittest.TestCase):

    def test_decode_href(self):
        hrefs = [
            "https://s3.amazonaws.com/bucket-name/filename",
            "s3://bucket-name/filename",
            "https://bucket-name.s3.amazonaws.com/filename",
            "https://s3-us-east-1.amazonaws.com/bucket-name/filename",
            "https://bucket-name.s3-us-east-1.amazonaws.com/filename",
            "https://bucket-name.s3.us-east-1.amazonaws.com/filename",
            (
                "https://s3.eu-central-1.wasabisys.com/eumap/lcv/"
                "lcv_blue_landsat.glad.ard_p50_30m_0..0cm_1999.12."
                "02..2000.03.20_eumap_epsg3035_v1.1.tif"
            ),
            (
                "https://download.geoservice.dlr.de/ENMAP/files/L0/2024/05/21/"
                "ENMAP01-_____L0-DT0000074151_20240521T055623Z_002_V010402_"
                "20240521T143405Z-QL_VNIR_COG.TIF"
            ),
            (
                "https://sentinel2l2a01.blob.core.windows.net/sentinel2-l2/"
                "55/X/EJ/2024/05/22/S2B_MSIL2A_20240522T032519_N0510_R018_"
                "T55XEJ_20240522T060936.SAFE/GRANULE/L2A_T55XEJ_A037653_"
                "20240522T032513/IMG_DATA/R60m/T55XEJ_20240522T032519_B01_60m.tif"
            ),
        ]

        expected_fs_paths = [
            (
                "eumap/lcv/lcv_blue_landsat.glad.ard_p50_30m_0..0cm_1999"
                ".12.02..2000.03.20_eumap_epsg3035_v1.1.tif"
            ),
            (
                "ENMAP/files/L0/2024/05/21/ENMAP01-_____L0-DT0000074151_"
                "20240521T055623Z_002_V010402_20240521T143405Z-QL_"
                "VNIR_COG.TIF"
            ),
            (
                "sentinel2-l2/55/X/EJ/2024/05/22/S2B_MSIL2A_20240522T032519"
                "_N0510_R018_T55XEJ_20240522T060936.SAFE/"
                "GRANULE/L2A_T55XEJ_A037653_20240522T032513/IMG_DATA/R60m/"
                "T55XEJ_20240522T032519_B01_60m.tif"
            ),
        ]
        expected_roots = [
            "s3.eu-central-1.wasabisys.com",
            "download.geoservice.dlr.de",
            "sentinel2l2a01.blob.core.windows.net",
        ]
        storage_options_region = dict(client_kwargs=dict(region_name="us-east-1"))
        expected_returns = [
            ("s3", "bucket-name", "filename", {}),
            ("s3", "bucket-name", "filename", {}),
            ("s3", "bucket-name", "filename", {}),
            ("s3", "bucket-name", "filename", storage_options_region),
            ("s3", "bucket-name", "filename", storage_options_region),
            ("s3", "bucket-name", "filename", storage_options_region),
            ("https", expected_roots[0], expected_fs_paths[0], {}),
            ("https", expected_roots[1], expected_fs_paths[1], {}),
            ("https", expected_roots[2], expected_fs_paths[2], {}),
        ]

        for expected, href in zip(expected_returns, hrefs):
            self.assertEqual(expected, decode_href(href), msg=href)

    def test_assert_aws_s3_bucket(self):
        with self.assertRaises(DataStoreError) as cm:
            bucket = "test_123-s3alias"
            href = "https://s3-us-east-1.amazonaws.com/bucket-name/filename"
            assert_aws_s3_bucket(bucket, href)
        self.assertEqual(
            (
                f"Bucket name {bucket!r} extracted from the href {href!r} "
                f"does not follow the AWS S3 bucket naming rules."
            ),
            f"{cm.exception}",
        )
        with self.assertRaises(DataStoreError) as cm:
            bucket = "m" * 64
            href = "https://s3-us-east-1.amazonaws.com/bucket-name/filename"
            assert_aws_s3_bucket(bucket, href)
        self.assertEqual(
            (
                f"Bucket name {bucket!r} extracted from the href {href!r} "
                f"does not follow the AWS S3 bucket naming rules."
            ),
            f"{cm.exception}",
        )

    def test_assert_aws_s3_region_name(self):
        with self.assertRaises(DataStoreError) as cm:
            region_name = "us-east-5"
            href = "https://s3-us-east-5.amazonaws.com/bucket-name/filename"
            assert_aws_s3_region_name(region_name, href)
        self.assertEqual(
            (
                f"Region name {region_name!r} extracted from the href {href!r} "
                "is not supported by AWS S3"
            ),
            f"{cm.exception}",
        )
