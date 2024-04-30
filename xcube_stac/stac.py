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

from typing import Any, Tuple, Iterator, Dict, Container, Union

import xarray as xr
from itertools import chain

import pystac
import pystac_client
from pystac.catalog import Catalog
from pystac.collection import Collection

from xcube_stac.constants import MIME_TYPES


class Stac:
    """ Class containing methods handling STAC catalogs
    """

    def __init__(
        self, url: str,
        collection_prefix: str = None,
        data_id_delimiter: str = "/"
    ):
        """
        Args:
            url (str): URL to STAC catalog
            collection_prefix (str, optional): Path of collection used as
                entry point. Defaults to None.
            data_id_delimiter (str, optional): Delimiter used to separate
                collections, items and assets from each other.
                Defaults to "/".
        """
        self._url = url
        self._collection_prefix = collection_prefix
        self._data_id_delimiter = data_id_delimiter

        # if STAC catalog is not searchable, pystac_client
        # falls back to pystac; to prevent warnings from pystac_client
        # use catalog from pystac instead. For more discussion refer to
        # https://github.com/xcube-dev/xcube-stac/issues/5
        catalog = pystac_client.Client.open(url)
        if not catalog.conforms_to("ITEM_SEARCH"):
            catalog = pystac.Catalog.from_file(url)

        # navigate to entry point of the data store
        if collection_prefix:
            collection_ids = collection_prefix.split(data_id_delimiter)
            for collection_id in collection_ids:
                catalog = catalog.get_child(collection_id)
        self.catalog = catalog

        # TODO: Add a data store "file" here when implementing
        # open_data(), which will be used to open the hrefs

    def open_data(self, data_id: str, **open_params) -> xr.Dataset:
        """ Open the data given by the data resource identifier *data_id*
        using the supplied *open_params*.

        Args:
            data_id (str): An identifier of data that is provided by this
                store.

        Raises:
            NotImplementedError: Not implemented yet.

        Returns:
            xr.Dataset: An in-memory representation of the data resources
                identified by *data_id* and *open_params*.
        """
        # TODO: implement this method using data store "file", see __init__()
        raise NotImplementedError("open_data() operation is not supported yet")

    def build_data_ids(
        self,
        pystac_object: Union[Catalog, Collection],
        path: str = "",
        recursive: bool = True,
        include_attrs: Container[str] = None
    ) -> Union[Iterator[str], Iterator[Tuple[str, Dict[str, Any]]]]:
        """ Build the data IDs from the structure of the catalog.
        The data resource identifiers follow the following structure:

            `collection_id_0/../collection_id_n/item_id/asset`

        Args:
            pystac_object (Union[Catalog, Collection]): either a
                `pystac.catalog:Catalog` or a `pystac.collection:Collection` object
            path (str, optional): collection path referring to
                `collection_id_0/../collection_id_n`. Defaults to "".
            recursive (bool, optional): If True, data IDs of a multiple collection
                and/or nested collection STAC catalog can be build. If False,
                flat STAC catalog hierarchy is assumed consisting only of items.
                Defaults to True.
            include_attrs (Container[str], optional): A sequence of names
                of attributes to be returned for each dataset identifier.
                If given, the store will attempt to provide the set of
                requested dataset attributes in addition to the data ids.
                If no attributes are found, empty dictionary is retured.
                So far only the attribute 'title' is supported.
                Defaults to None.

        Yields:
            Iterator[str]: An iterator over the identifiers (and additional
                attributes defined by *include_attrs* of data resources provided
                by this data store.
        """
        if recursive:
            if pystac_object == self.catalog:
                pass
            else:
                path += pystac_object.id
                path += self._data_id_delimiter
            if any(True for _ in pystac_object.get_children()):
                iterators = (self.build_data_ids(
                    child,
                    path=path,
                    recursive=True,
                    include_attrs=include_attrs
                ) for child in pystac_object.get_children())
                yield from chain(*iterators)
            else:
                iterator = self.build_data_ids(
                    pystac_object,
                    path=path,
                    recursive=False,
                    include_attrs=include_attrs
                )
                yield from iterator
        else:
            return_tuples = include_attrs is not None
            # TODO: support other attributes in include_attrs
            include_titles = return_tuples and "title" in include_attrs
            for item in pystac_object.get_items():
                for k, v in item.assets.items():
                    if any(x in MIME_TYPES for x in v.media_type.split("; ")):
                        data_id = path + item.id + self._data_id_delimiter + k
                        if include_titles:
                            if hasattr(v, "title"):
                                attrs = {"title": v.title}
                            else:
                                attrs = {}
                            yield (data_id, attrs)
                        else:
                            yield data_id
