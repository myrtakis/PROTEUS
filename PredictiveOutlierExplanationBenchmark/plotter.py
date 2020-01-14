import argparse
import matplotlib.pyplot as plt
import json
import numpy as np
import os
import collections


markers = ['s', 'o', 'd', 'v', '^', '<', '>', 'x', '+']


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


def plot_dim_experiment(args):
    metrics_dim_results_dict = get_results_as_dict(get_results_files(args.dimexp))
    for metric, algos_dict in metrics_dim_results_dict.items():
        beautiful_x_ticks = None
        x = []
        counter = 0
        for alg, dim_perfomances in algos_dict.items():
            x = np.arange(0, len(dim_perfomances.keys()), 1)
            y = list(dim_perfomances.values())
            if beautiful_x_ticks is None:
                beautiful_x_ticks = [str(x) + '-d' for x in dim_perfomances.keys()]
            plt.plot(x, y, label=alg, linestyle='solid', marker=markers[counter])
            counter += 1
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.125), ncol=3, frameon=False)
        plt.ylim([0, 1.05])
        plt.xticks(x, beautiful_x_ticks)
        plt.yticks(np.arange(.0, 1.1, .1))
        plt.xlabel('Dataset Dimensionality')
        plt.ylabel(metric)
        plt.grid(lw=0.3, zorder=0, linestyle='dotted')
        outputfile = os.path.join(args.savedir, 'hics_dimexp_' + metric + '.png')
        if not os.path.exists(args.savedir):
            os.makedirs(args.savedir)
        plt.savefig(outputfile, dpi=300, bbox_inches='tight', pad_inches=0.3)
        plt.clf()


def calculate_metrics_avg(results_json):
    metrics_avg = {}
    reps = len(results_json.keys())
    for rep_id, res in results_json.items():
        for metric_key, val in res.items():
            for conf_id in val:
                if metric_key not in metrics_avg:
                    metrics_avg[metric_key] = {}
                if conf_id not in metrics_avg[metric_key]:
                    metrics_avg[metric_key][conf_id] = 0
                metrics_avg[metric_key][conf_id] += val[conf_id]['performance']
    # Take the average
    for metric_id, conf in metrics_avg.items():
        for conf_id, mval in conf.items():
            metrics_avg[metric_id][conf_id] = mval / reps
    return metrics_avg


def get_log_files(dir_path):
    return None


def get_results_files(dir_path):
    fileslist = []
    assert os.path.isdir(dir_path)
    allfiles = os.listdir(dir_path)
    for f in allfiles:
        if f.endswith('.json') and 'log' not in f:
            fileslist.append(os.path.join(dir_path, f))
    return fileslist


def get_results_as_dict(results_files):
    metric_dim_results_dict = {}
    for f in results_files:
        with open(f) as json_file:
            results = json.load(json_file)
            datasetname = (os.path.splitext(os.path.basename(f))[0]).split('_')[0]
            dim = int(datasetname.replace("hics", "").strip())
            metrics_results_avg = calculate_metrics_avg(results)
            for metric_key, results in metrics_results_avg.items():
                if metric_key not in metric_dim_results_dict:
                    metric_dim_results_dict[metric_key] = {}
                for alg, perfomance in results.items():
                    if alg not in metric_dim_results_dict[metric_key]:
                        metric_dim_results_dict[metric_key][alg] = {}
                    if dim not in metric_dim_results_dict[metric_key][alg]:
                        metric_dim_results_dict[metric_key][alg][dim] = perfomance
                metric_dim_results_dict[metric_key] = \
                    dict(collections.OrderedDict(sorted(metric_dim_results_dict[metric_key].items())))
    for metric_key, algs_dict in metric_dim_results_dict.items():
        for alg, performances in algs_dict.items():
            metric_dim_results_dict[metric_key][alg] = dict(collections.OrderedDict(sorted(performances.items())))
    return metric_dim_results_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""Plot benchmark results.""")
    parser.add_argument('-r', '--results', help='The results file (json file).', default=None)
    parser.add_argument('-dimexp', help='-dimexp dir')
    parser.add_argument('-sdir', '--savedir', help='The directory of the created plots.', required=True)
    parser.add_argument('-dataset', help='The dataset of the results.')
    args = parser.parse_args()
    if args.results is not None:
        draw_barplot(args)
    if args.dimexp is not None:
        plot_dim_experiment(args)
