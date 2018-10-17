#!/usr/bin/python

#######################################################
# Copyright (c) 2015, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause
########################################################

from setuptools import setup, find_packages
#from __af_version__ import full_version

#TODO:
#1) Look for af libraries during setup

setup(
    author="Pavan Yalamanchili",
    author_email="contact@pavanky.com",
    name="arrayfire",
    version="3.6.20181017",
    description="Python bindings for ArrayFire",
    license="BSD",
    url="http://arrayfire.com",
    packages=find_packages()
)
