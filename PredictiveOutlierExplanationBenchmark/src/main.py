import argparse
from PredictiveOutlierExplanationBenchmark.src.pipeline.Pipeline import Pipeline


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='Configuration file path.', required=True)
    parser.add_argument('-sd', '--save_dir', help='The directory to save to old_results', default='old_results')
    args = parser.parse_args()

    Pipeline.run(args.config, args.save_dir)
