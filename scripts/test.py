import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('mode')
    parser.add_argument('--json')
    args = parser.parse_args()

    print(args.json)
