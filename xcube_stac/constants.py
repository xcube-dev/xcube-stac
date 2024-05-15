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


from xcube.util.jsonschema import (
    JsonArraySchema,
    JsonDateSchema,
    JsonNumberSchema,
    JsonStringSchema
)


DATA_STORE_ID = "stac"
DATASET_OPENER_ID = f"dataset:zarr:{DATA_STORE_ID}"

MIME_TYPES = [
    "application/zarr",
    "image/tiff"
]

STAC_SEARCH_ITEM_PARAMETERS = dict(
    variable_names=JsonArraySchema(
        items=(JsonStringSchema(min_length=0)),
        unique_items=True,
        title="Variable names in xarray.Dataset",
        description=(
            "Variable names are defined by assets"
        )
    ),
    time_range=JsonArraySchema(
        items=[
            JsonDateSchema(nullable=True),
            JsonDateSchema(nullable=True),
        ],
        title="Time Range",
        description=(
            "Time range given as pair of start and stop dates.  "
            "Dates must be given using format 'YYYY-MM-DD'. "
            "Start and stop are inclusive."
        )
    ),
    bbox=JsonArraySchema(
        items=(
            JsonNumberSchema(),
            JsonNumberSchema(),
            JsonNumberSchema(),
            JsonNumberSchema(),
        ),
        title="Bounding box [x1,y1, x2,y2] in geographical coordinates",
    ),
    collections=JsonArraySchema(
        items=(JsonStringSchema(min_length=0)),
        unique_items=True,
        title="Collection IDs",
        description=(
            "Collection IDs to be included in the search request."
        )
    )
)
