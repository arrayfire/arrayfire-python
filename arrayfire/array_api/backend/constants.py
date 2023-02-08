import enum


class ErrorCodes(enum.Enum):
    none = 0

    # 100-199 Errors in environment
    no_mem = 101
    driver = 102
    runtime = 103

    # 200-299 Errors in input parameters
    invalid_array = 201
    arg = 202
    size = 203
    type = 204
    diff_type = 205
    batch = 207
    device = 208

    # 300-399 Errors for missing software features
    not_supported = 301
    not_configured = 302
    nonfree = 303

    # 400-499 Errors for missing hardware features
    no_dbl = 401
    no_gfx = 402
    no_half = 403

    # 500-599 Errors specific to the heterogeneous API
    load_lib = 501
    load_sym = 502
    arr_bknd_mismatch = 503

    # 900-999 Errors from upstream libraries and runtimes
    internal = 998
    unknown = 999
