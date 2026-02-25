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

import re

from xcube.core.store import DataStoreError

# Bucket naming rules:
# https://docs.aws.amazon.com/AmazonS3/latest/userguide/bucketnamingrules.html
AWS_REGEX_BUCKET_NAME = (
    r"(?!^([0-9]{1,3}\.){3}[0-9]{1,3}$)"
    r"(?!(^xn--|^sthree-|^sthree-configurator|"
    r".+--ol-s3$|.+-s3alias$))"
    r"^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$"
)


def decode_href(href: str) -> tuple[str, str, str]:
    """Decodes a href into protocol, root, and remaining file path

    Note:
        The bucket naming rules are given by
        https://docs.aws.amazon.com/AmazonS3/latest/userguide/bucketnamingrules.html.

    Args:
        href: href string of data resource

    Returns:
        protocol: protocol name
        root: root of https or S3 bucket name
        fs_path: remaining file path, and storage options.

    Raises:
        DataStoreError: Error, AWS S3 root cannot be decoded since
            it does not follow the uri pattern mentioned in Note.
    """
    protocol, remain = href.split("://")
    root = remain.split("/")[0]
    fs_path = remain.replace(f"{root}/", "")
    if protocol == "s3":
        assert_aws_s3_bucket(root, href)

    return protocol, root, fs_path


def assert_aws_s3_bucket(bucket: str, href: str):
    """Test if bucket name follows the prescribed AWS S3 naming rules.

    Note:
        The bucket naming rules are given by
        https://docs.aws.amazon.com/AmazonS3/latest/userguide/bucketnamingrules.html.


    Args:
        bucket: bucket name
        href: href string of data resource

    Raises:
        DataStoreError: Error, if the bucket name does not follow the naming rules
    """
    if re.search(AWS_REGEX_BUCKET_NAME, bucket) is None:
        raise DataStoreError(
            f"Bucket name {bucket!r} extracted from the href {href!r} "
            "does not follow the AWS S3 bucket naming rules."
        )
