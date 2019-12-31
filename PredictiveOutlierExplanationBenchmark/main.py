import argparse
from benchmark import run_benchmark


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""Perform the benchmark with the given config.""")
    parser.add_argument('-c', '--config', help='Configuration file name.', required=True)
    args = parser.parse_args()
    run_benchmark(args.config)
