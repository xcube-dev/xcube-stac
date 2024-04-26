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

import xarray as xr


class Stac:
    """ Common operations on STAC catalogs.

    Attributes:
        url: URL to STAC catalog
        collection_prefix: Path of collection used as
            entry point. Defaults to None.
        data_id_delimiter: Delimiter used to separate
            collections, items and assets from each other.
            Defaults to "/".
    """

    def __init__(
        self,
        url: str,
        collection_prefix: str = None,
        data_id_delimiter: str = "/"
    ):
        self._url = url
        self._collection_prefix = collection_prefix
        self._data_id_delimiter = data_id_delimiter
        # ToDo: open Catalog and direct to entry point defined by *collection_prefix*
        # ToDo: Add a data store "file", which will be used to open the hrefs

    def open_data(self, data_id: str, **open_params) -> xr.Dataset:
        """ Open the data given by the data resource identifier *data_id*
        using the supplied *open_params*.

        Args:
            data_id: An identifier of data that is provided by this
                store.

        Raises:
            NotImplementedError: Not implemented yet.

        Returns:
            An in-memory representation of the data resources
            identified by *data_id* and *open_params*.
        """
        # ToDo: implement this method using data store "file", see __init__()
        raise NotImplementedError("open_data() operation is not supported yet")
