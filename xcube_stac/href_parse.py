# The MIT License (MIT)
# Copyright (c) 2024 by the xcube development team and contributors
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
from typing import Tuple

from xcube.core.store import DataStoreError


# Bucket naming rules:
# https://docs.aws.amazon.com/AmazonS3/latest/userguide/bucketnamingrules.html
AWS_REGEX_BUCKET_NAME = (
    r"(?!^([0-9]{1,3}\.){3}[0-9]{1,3}$)"
    r"(?!(^xn--|^sthree-|^sthree-configurator|"
    r".+--ol-s3$|.+-s3alias$))"
    r"^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$"
)
# Region names: https://docs.aws.amazon.com/general/latest/gr/s3.html
AWS_REGION_NAMES = [
    "us-east-1",
    "us-east-2",
    "us-west-1",
    "us-west-2",
    "af-south-1",
    "ap-east-1",
    "ap-south-1",
    "ap-south-2",
    "ap-southeast-1",
    "ap-southeast-2",
    "ap-southeast-3",
    "ap-southeast-4",
    "ap-northeast-1",
    "ap-northeast-2",
    "ap-northeast-3",
    "ca-central-1",
    "ca-west-1",
    "eu-central-1",
    "eu-central-2",
    "eu-west-1",
    "eu-west-2",
    "eu-west-3",
    "eu-south-1",
    "eu-south-2",
    "eu-north-1",
    "il-central-1",
    "me-south-1",
    "me-central-1",
    "sa-east-1",
    "us-gov-east-1",
    "us-gov-west-1",
]


def _decode_href(href: str) -> Tuple[str, str, str, dict]:
    """Decodes a href into protocol, root, remaining file path,
    and region name if given.

    Note:
        The aws s3 URI formats are given by
        https://docs.aws.amazon.com/quicksight/latest/user/supported-manifest-file-format.html.
        Furthermore, the format
        'https://<bucket-name>.s3.<region-name>.amazonaws.com/<filename>'
        is encountered and will be supported as well.
        The bucket naming rules are given by
        https://docs.aws.amazon.com/AmazonS3/latest/userguide/bucketnamingrules.html.
        The region names are given by
        https://docs.aws.amazon.com/general/latest/gr/s3.html.

    Args:
        href: href string of data resource

    Returns: protocol, root, remaining file path, and storage options.

    Raises:
        DataStoreError: Error, AWS S3 root cannot be decoded since
            it does not follow the uri pattern mentioned in Note.
    """
    protocol, root, fs_path, storage_options = _decode_aws_s3_href(href)
    if root is None:
        protocol, remain = href.split("://")
        root = remain.split("/")[0]
        fs_path = remain.replace(f"{root}/", "")
        storage_options = {}

    return protocol, root, fs_path, storage_options


def _decode_aws_s3_href(href: str):
    """Decodes an AWS S3 href into protocol, root, remaining file path,
    and storage options needed for the S3 data store. If href does not fit to
    the AWS S3 pattern, root, fs_path and region_name will be None.

    Note:
        The aws s3 URI formats are given by
        https://docs.aws.amazon.com/quicksight/latest/user/supported-manifest-file-format.html.
        Furthermore, the format
        'https://<bucket-name>.s3.<region-name>.amazonaws.com/<filename>'
        is encountered and will be supported as well.
        The bucket naming rules are given by
        https://docs.aws.amazon.com/AmazonS3/latest/userguide/bucketnamingrules.html.
        The region names are given by
        https://docs.aws.amazon.com/general/latest/gr/s3.html.

    Args:
        href: href string of data resource

    Returns: protocol, root, remaining file path, and storage options

    Raises:
        DataStoreError: Error if AWS S3 bucket name or region name does not follow
            the prescribed naming rules given in the notes above.
    """
    # check for aws s3; see notes for different uri formats
    protocol = "s3"
    fs_path = None
    region_name = None
    root = None
    if re.search(r"^https://s3\.amazonaws\.com/.{3,63}/", href) is not None:
        tmp = href[8:].split("/")
        root = tmp[1]
        fs_path = "/".join(tmp[2:])
    elif re.search(r"^s3://.{3,63}/", href) is not None:
        tmp = href[5:].split("/")
        root = tmp[0]
        fs_path = "/".join(tmp[1:])
    elif re.search(r"^https://.{3,63}\.s3\.amazonaws\.com/", href) is not None:
        tmp = href[8:].split("/")
        root = tmp[0][:-17]
        fs_path = "/".join(tmp[1:])
    elif re.search(r"^https://s3-.{9,14}\.amazonaws\.com/.{3,63}/", href) is not None:
        tmp = href[8:].split("/")
        region_name = tmp[0][3:-14]
        root = tmp[1]
        fs_path = "/".join(tmp[2:])
    elif re.search(r"^https://.{3,63}\.s3-.{9,14}\.amazonaws\.com/", href) is not None:
        tmp = href[8:].split("/")
        region_name = tmp[0].split(".s3-")[-1][:-14]
        root = tmp[0].replace(f".s3-{region_name}.amazonaws.com", "")
        fs_path = "/".join(tmp[1:])
    elif re.search(r"^https://.{3,63}\.s3\..{9,14}\.amazonaws\.com/", href) is not None:
        tmp = href[8:].split("/")
        region_name = tmp[0].split(".s3.")[-1][:-14]
        root = tmp[0].replace(f".s3.{region_name}.amazonaws.com", "")
        fs_path = "/".join(tmp[1:])

    if root is not None:
        _assert_aws_s3_bucket(root, href)
    if region_name is not None:
        _assert_aws_s3_region_name(region_name, href)

    if region_name is None:
        storage_options = {}
    else:
        storage_options = {"client_kwargs": {"region_name": region_name}}

    return protocol, root, fs_path, storage_options


def _assert_aws_s3_bucket(bucket: str, href: str):
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


def _assert_aws_s3_region_name(region_name: str, href: str):
    """Test if region name is a valid AWS S3 region name.

    Note:
        The region names are given by
        https://docs.aws.amazon.com/general/latest/gr/s3.html.


    Args:
        region_name: region name
        href: href string of data resource

    Raises:
        DataStoreError: Error, if the bucket name does not follow the naming rules
    """
    if region_name not in AWS_REGION_NAMES:
        raise DataStoreError(
            f"Region name {region_name!r} extracted from the "
            f"href {href!r} is not supported by AWS S3"
        )
