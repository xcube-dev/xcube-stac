# The MIT License (MIT)
# Copyright (c) 2023 by the xcube development team and contributors
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
import stackstac
import pystac_client
import dask.diagnostics


class Stac:

    def __init__(self, url: str):
        self._url = url
        self.client = pystac_client.Client.open(url)

    def open_dataset(
        self, client_params: dict, stackstac_params: dict
    ) -> xr.Dataset:
        try:
            items = self.client.search(**client_params).item_collection()
            return stackstac.stack(items, **stackstac_params)
        # ToDo: better error message, what is the error?
        except Exception as e:
            print(e)
            return None

    def retrieve_dataset(self, stack: xr.Dataset) -> xr.Dataset:
        try:
            with dask.diagnostics.ProgressBar():
                data = stack.compute()
            return data
        # ToDo: better error message, what is the error?
        except Exception as e:
            print(e)
            return None
