import argparse
import matplotlib.pyplot as plt
import json
import numpy as np
import os


def draw_barplot(args):
    with open(args.results) as json_file:
        results_json = json.load(json_file)
        metrics_avg = calculate_metrics_avg(results_json)
    bar_width = 0.25
    for metric_id, conf in metrics_avg.items():
        counter = 1
        pos_ticks = []
        for conf_id, mval in conf.items():
            pos = bar_width * counter
            pos_ticks.append(pos)
            plt.bar([pos], [mval], width=bar_width, edgecolor='black')
            counter += 1
        plt.title(args.dataset)
        plt.ylabel(metric_id)
        plt.ylim((0,1))
        plt.xticks(ticks=np.array(pos_ticks), labels=list(conf.keys()), rotation=40, ha='right')
        output = os.path.join(args.savedir, metric_id + '.png')
        if not os.path.exists(args.savedir):
            os.makedirs(args.savedir)
        plt.savefig(output, dpi=300, bbox_inches='tight', pad_inches=0.3)
        plt.clf()


def calculate_metrics_avg(results_json):
    metrics_avg = {}
    reps = len(results_json.keys())
    for rep_id, conf in results_json.items():
        for conf_id, val in conf.items():
            for metric_key in val['metrics']:
                if metric_key not in metrics_avg:
                    metrics_avg[metric_key] = {}
                if conf_id not in metrics_avg[metric_key]:
                    metrics_avg[metric_key][conf_id] = 0
                metrics_avg[metric_key][conf_id] += val['metrics'][metric_key]
    # Take the average
    for metric_id, conf in metrics_avg.items():
        for conf_id, mval in conf.items():
            metrics_avg[metric_id][conf_id] = mval / reps
    return metrics_avg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""Plot benchmark results.""")
    parser.add_argument('-r', '--results', help='The results file (json file).', required=True)
    parser.add_argument('-sdir', '--savedir', help='The directory of the created plots.', required=True)
    parser.add_argument('-dataset', help='The dataset of the results.', required=True)
    args = parser.parse_args()
    draw_barplot(args)
