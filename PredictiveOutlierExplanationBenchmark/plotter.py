import argparse
import matplotlib.pyplot as plt
import json
import numpy as np
import os
import seaborn as sns
from utils import *


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
    metrics_dim_results_dict = get_results_as_dict(get_results_files(args.dimexp, ending='json', exclude='log'))
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


def plot_features(args):
    alg_dim_fcount_df = feature_count_df(args.plot_features)
    for alg, fcount_df in alg_dim_fcount_df.items():
        sns.heatmap(fcount_df, annot=True, fmt='d', cbar_kws={'label': 'Frequency'})
        plt.xlabel('Dataset Dimensionality')
        plt.ylabel('Relevant Features')
        plt.title(alg)
        plt.tight_layout()
        title = alg + '_' + ('' if args.title is None else args.title)
        plt.title(title)
        if not os.path.exists(args.savedir):
            os.makedirs(args.savedir)
        outputdir = os.path.join(args.savedir, title + '.png')
        plt.savefig(outputdir, dpi=300)
        plt.clf()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""Plot benchmark results.""")
    parser.add_argument('-r', '--results', help='The results file (json file).', default=None)
    parser.add_argument('-dimexp', help='-dimexp dir')
    parser.add_argument('-sdir', '--savedir', help='The directory of the created plots.', required=True)
    parser.add_argument('-dataset', help='The dataset of the results.')
    parser.add_argument('-f', '--plot_features', default=None)
    parser.add_argument('-t', '--title', default=None)
    args = parser.parse_args()
    if args.results is not None:
        draw_barplot(args)
    if args.dimexp is not None:
        plot_dim_experiment(args)
    if args.plot_features is not None:
        plot_features(args)
