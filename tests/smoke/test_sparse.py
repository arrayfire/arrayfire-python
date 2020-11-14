#!/usr/bin/env python

#######################################################
# Copyright (c) 2020, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause
########################################################

import arrayfire as af


def test_simple_sparse() -> None:
    dd = af.randu(5, 5)
    ds = dd * (dd > 0.5)
    sp = af.create_sparse_from_dense(ds)
    assert af.sparse_get_info(sp)
    assert af.sparse_get_values(sp)
    assert af.sparse_get_row_idx(sp)
    assert af.sparse_get_col_idx(sp)
    assert af.sparse_get_nnz(sp)
    assert af.sparse_get_storage(sp)
