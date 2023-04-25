import platform


def is_arch_x86() -> bool:
    machine = platform.machine()
    return platform.architecture()[0][0:2] == "32" and (machine[-2:] == "86" or machine[0:3] == "arm")
