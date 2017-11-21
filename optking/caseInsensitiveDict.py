# independent, miscellaneous functions

import collections
import copy

class CaseInsensitiveDict(collections.Mapping):
    def __init__(self, d):
        self._d = d
        self._s = dict((k.lower(), k) for k in d)
    def __contains__(self, k):
        return k.lower() in self._s
    def __len__(self):
        return len(self._s)
    def __iter__(self):
        return iter(self._s)
    def __getitem__(self, k):
        return self._d[self._s[k.lower()]]

    def __setitem__(self, k, val):
        self._s[ k.lower() ] = k
        self._d[ k ] = val
        return

    def actual_key_case(self, k):
        return self._s.get(k.lower())
    def copy(self):
        return copy.copy(self)

