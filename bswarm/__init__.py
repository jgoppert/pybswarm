import pkg_resources


def main():
    """Entry point for the application script"""
    print("Call your main application code here")


def load_file():
    data_file = pkg_resources.resource_filename('bswarm', 'data/package_data.dat')
    with open(data_file, 'r') as f:
        print(f.read())
