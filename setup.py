#!/usr/bin/env python

#######################################################
# Copyright (c) 2015, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause
########################################################

import os

# package can be distributed with arrayfire binaries or
# just with python wrapper files, the AF_BUILD_LOCAL
# environment var determines whether to build the arrayfire
# binaries locally rather than searching in a system install

AF_BUILD_LOCAL_LIBS = os.environ.get('AF_BUILD_LOCAL_LIBS')
print(f'AF_BUILD_LOCAL_LIBS={AF_BUILD_LOCAL_LIBS}')
if AF_BUILD_LOCAL_LIBS:
    print('Proceeding to build ArrayFire libraries')
else:
    print('Skipping binaries installation, only python files will be installed')

AF_BUILD_CPU = os.environ.get('AF_BUILD_CPU')
AF_BUILD_CPU = 1 if AF_BUILD_CPU is None else int(AF_BUILD_CPU)
AF_BUILD_CPU_CMAKE_STR = '-DAF_BUILD_CPU:BOOL=ON' if (AF_BUILD_CPU == 1) else '-DAF_BUILD_CPU:BOOL=OFF'

AF_BUILD_CUDA = os.environ.get('AF_BUILD_CUDA')
AF_BUILD_CUDA = 1 if AF_BUILD_CUDA is None else int(AF_BUILD_CUDA)
AF_BUILD_CUDA_CMAKE_STR = '-DAF_BUILD_CUDA:BOOL=ON' if (AF_BUILD_CUDA == 1) else '-DAF_BUILD_CUDA:BOOL=OFF'

AF_BUILD_OPENCL = os.environ.get('AF_BUILD_OPENCL')
AF_BUILD_OPENCL = 1 if AF_BUILD_OPENCL is None else int(AF_BUILD_OPENCL)
AF_BUILD_OPENCL_CMAKE_STR = '-DAF_BUILD_OPENCL:BOOL=ON' if (AF_BUILD_OPENCL == 1) else '-DAF_BUILD_OPENCL:BOOL=OFF'

AF_BUILD_UNIFIED = os.environ.get('AF_BUILD_UNIFIED')
AF_BUILD_UNIFIED = 1 if AF_BUILD_UNIFIED is None else int(AF_BUILD_UNIFIED)
AF_BUILD_UNIFIED_CMAKE_STR = '-DAF_BUILD_UNIFIED:BOOL=ON' if (AF_BUILD_UNIFIED == 1) else '-DAF_BUILD_UNIFIED:BOOL=OFF'

if AF_BUILD_LOCAL_LIBS:
    # invoke cmake and build arrayfire libraries to install locally in package
    from skbuild import setup

    def filter_af_files(cmake_manifest):
        cmake_manifest = list(filter(lambda name: not (name.endswith('.h') 
            or name.endswith('.cpp')
            or name.endswith('.hpp')
            or name.endswith('.cmake')
            or name.endswith('jpg')
            or name.endswith('png')
            or 'examples' in name), cmake_manifest))
        return cmake_manifest

    print('Building CMAKE with following configurable variables: ')
    print(AF_BUILD_CPU_CMAKE_STR)
    print(AF_BUILD_CUDA_CMAKE_STR)
    print(AF_BUILD_OPENCL_CMAKE_STR)
    print(AF_BUILD_UNIFIED_CMAKE_STR)


    setup(
        cmake_install_dir='arrayfire',
        cmake_process_manifest_hook=filter_af_files,
        cmake_args=[AF_BUILD_CPU_CMAKE_STR,
                    AF_BUILD_CUDA_CMAKE_STR,
                    AF_BUILD_OPENCL_CMAKE_STR,
                    AF_BUILD_UNIFIED_CMAKE_STR,
                    '-DCUDA_architecture_build_targets:STRING=All' # todo: pass from environ
                    '-DAF_BUILD_DOCS:BOOL=OFF',
                    '-DAF_BUILD_EXAMPLES:BOOL=OFF',
                    '-DAF_INSTALL_STANDALONE:BOOL=ON',
                    '-DBUILD_TESTING:BOOL=OFF',
                    ]
    )

else:
    # ignores local arrayfire libraries, will search system instead
    from setuptools import setup
    setup()

