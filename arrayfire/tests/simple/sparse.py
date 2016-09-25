#!/usr/bin/python
#######################################################
# Copyright (c) 2015, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause
########################################################

import arrayfire as af
from . import _util

def simple_sparse(verbose=False):
    display_func = _util.display_func(verbose)
    print_func   = _util.print_func(verbose)

    dd = af.randu(5, 5)
    ds = dd * (dd > 0.5)
    sp = af.create_sparse_from_dense(ds)
    display_func(af.sparse_get_info(sp))
    display_func(af.sparse_get_values(sp))
    display_func(af.sparse_get_row_idx(sp))
    display_func(af.sparse_get_col_idx(sp))
    print_func(af.sparse_get_nnz(sp))
    print_func(af.sparse_get_storage(sp))

_util.tests['sparse'] = simple_sparse
