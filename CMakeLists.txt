cmake_minimum_required(VERSION 3.11.0)
project(arrayfire-python)
find_package(PythonExtensions REQUIRED)
include(FetchContent)

set(CMAKE_MODULE_PATH_OLD ${CMAKE_MODULE_PATH})
set(CMAKE_MODULE_PATH "")
set(NO_SONAME)

FetchContent_Declare(
    arrayfire
    GIT_REPOSITORY https://github.com/arrayfire/arrayfire.git
    GIT_TAG v3.8
)
FetchContent_MakeAvailable(arrayfire)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH_OLD})

set(ignoreWarning "${SKBUILD}")
