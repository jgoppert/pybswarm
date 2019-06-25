import pkg_resources
import sys


if sys.version_info[0] != 3:
    raise ValueError("bswarm requires Python 3")

def main():
    """Entry point for the application script"""
    print("Call your main application code here")


def load_file():
    data_file = pkg_resources.resource_filename('bswarm', 'data/package_data.dat')
    with open(data_file, 'r') as f:
        print(f.read())
