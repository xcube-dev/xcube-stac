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

import logging
import xarray as xr

from xcube.util.jsonschema import JsonObjectSchema
from xcube.core.store import (
    DataOpener,
    DatasetDescriptor
)
from .stac import Stac

_LOG = logging.getLogger("xcube")


class StacDataOpener(DataOpener):
    """ STAC implementation of the data opener.

    Attributes:
        stac: Common operations on STAC catalogs
    """

    def __init__(self, stac: Stac):
        self.stac = stac

    def get_open_data_params_schema(self, data_id: str = None) -> JsonObjectSchema:
        """ Get the schema for the parameters passed as *open_params* to
        :meth:`open_data`.

        Args:
            data_id: An identifier of data that is provided by this
                store. Defaults to None.

        Returns:
            Schema containing the parameters used by the data opener
            to open data.
        """
        # ToDo: to be adjusted
        open_parms = {}
        stac_schema = JsonObjectSchema(
            properties=dict(**open_parms),
            required=[],
            additional_properties=False
        )
        return stac_schema

    def open_data(self, data_id: str, **open_params) -> xr.Dataset:
        """ Open the data given by the data resource identifier *data_id*
        using the supplied *open_params*.

        Args:
            data_id: An identifier of data that is provided by this
                store.

        Returns:
            An in-memory representation of the data resources
            identified by *data_id* and *open_params*.
        """
        stac_schema = self.get_open_data_params_schema()
        stac_schema.validate_instance(open_params)
        return self.stac.open_data(data_id, **open_params)

    def describe_data(
        self, data_id: str, **open_params
    ) -> DatasetDescriptor:
        """ Get the descriptor for the data resource given by *data_id*.

        Args:
            data_id: An identifier of data that is provided by this
                store.

        Raises:
            NotImplementedError: Not implemented yet.

        Returns:
            Data descriptor containing meta data of
            the data resources identified by *data_id*
        """
        # ToDo: implement describe_data method.
        raise NotImplementedError("describe_data() operation is not supported yet")