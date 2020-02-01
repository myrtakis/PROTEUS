import argparse
import matplotlib.pyplot as plt
import json
import numpy as np
import os
import seaborn as sns
from utils import *
from matplotlib.font_manager import FontProperties


markers = ['s', 'o', 'd', 'v', '^', '<', '>', 'x', '+']


def draw_barplot(args):
    res_files = []
    if os.path.isdir(args.results):
        res_files = get_results_files(args.results, ending='json', exclude='log')
    else:
        res_files.append(args.results)
    for f in res_files:
        with open(f) as json_file:
            results_json = json.load(json_file)
            metrics_avg = calculate_metrics_avg(results_json)
        bar_width = 0.25
        for metric_id, conf in metrics_avg.items():
            counter = 1
            pos_ticks = []
            xticks_arr = []
            for conf_id, mval in conf.items():
                pos = bar_width * counter
                pos_ticks.append(pos)
                plt.bar([pos], [mval], width=bar_width, edgecolor='black', zorder=4)
                if 'none_' in conf_id:
                    conf_id = conf_id.replace('none_', '')
                conf_id = conf_id.upper()
                xticks_arr.append(conf_id)
                counter += 1
            plt.ylabel('mean_' + metric_id, fontsize=14)
            plt.ylim((0,1))
            plt.xticks(fontsize=14, ticks=np.array(pos_ticks), labels=xticks_arr, rotation=40, ha='right')
            plt.yticks(np.arange(0, 1.1, 0.1), fontsize=14)
            plt.grid()
            res_name = os.path.splitext(os.path.basename(f))[0]
            title = res_name
            if '_results' in title:
                title = title.replace('_results', '')
            if 'knn' in title:
                title = title.replace('knn', 'db')
            plt.title(title, fontsize=14)
            output = os.path.join(args.savedir, res_name + "_" + metric_id + '.png')
            if not os.path.exists(args.savedir):
                os.makedirs(args.savedir)
            plt.savefig(output, dpi=300, bbox_inches='tight', pad_inches=0.2)
            print('Figure saved in', output)
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
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.125), ncol=3, frameon=False, fontsize=14)
        plt.ylim([0, 1.05])
        plt.xticks(x, beautiful_x_ticks)
        plt.yticks(np.arange(.0, 1.1, .1))
        plt.xlabel('Dataset Dimensionality', fontsize=14)
        plt.ylabel(metric, fontsize=14)
        plt.grid(lw=0.3, zorder=0, linestyle='dotted')
        outputfile = os.path.join(args.savedir, 'hics_dimexp_' + metric + '.png')
        if not os.path.exists(args.savedir):
            os.makedirs(args.savedir)
        plt.savefig(outputfile, dpi=300, bbox_inches='tight', pad_inches=0.3)
        plt.clf()


def plot_features(args):
    alg_dim_fcount_df, alg_feature_mean_prec_dfs, colorbar_range = feature_count_df(args.plot_features)
    for alg, fcount_df in alg_dim_fcount_df.items():
        sns.heatmap(fcount_df, annot=True, cbar_kws={'label': 'Frequency', 'ticks': colorbar_range},
                    annot_kws={'size':10}, xticklabels=False, vmin=min(colorbar_range), vmax=max(colorbar_range))
        sns.set({'axes.labelsize': 11})
        #plt.xlabel('Dataset Dimensionality')
        plt.ylabel('Relevant Features')
        plt.title(alg)
        title = alg + '_' + ('' if args.title is None else args.title)
        plt.title(title)
        plt.tick_params(axis='x', labelsize=11)
        plt.tick_params(axis='y', labelsize=11)
        plt.yticks(rotation=45)
        if not os.path.exists(args.savedir):
            os.makedirs(args.savedir)
        outputdir = os.path.join(args.savedir, title + '.png')
        alg_features_mean_prec_df = alg_feature_mean_prec_dfs[alg]
        # To add row name just add to the plt.table function rowLabels=['Mean Precision']
        table = plt.table(cellText=alg_features_mean_prec_df.values, colLabels=alg_features_mean_prec_df.columns,
                          loc='bottom', cellLoc='center')
        table.scale(1, 2)
        table.set_fontsize(12)
        for (row, col), cell in table.get_celld().items():
            if (row == 0) or (col == -1):
                cell.set_text_props(fontproperties=FontProperties(weight='bold'))
                cell.set_linewidth(0)
        plt.subplots_adjust(left=0.2, bottom=0.3, hspace=1.0)
        print('Figure saved in ', outputdir)
        plt.savefig(outputdir, dpi=300, bbox_inches='tight', pad_inches=0.3)
        plt.clf()


def plot_runtime_cluster_size(args):
    markers = ['s', '^']
    fsel_dim_runtime_dict = get_dim_runtime_dict(args.run_time)
    beautiful_x_ticks = None
    counter = 0
    for alg, dim_perfomances in fsel_dim_runtime_dict.items():
        x = np.arange(0, len(dim_perfomances.keys()), 1)
        y = list(dim_perfomances.values())
        if beautiful_x_ticks is None:
            beautiful_x_ticks = [str(x) + '-d' for x in dim_perfomances.keys()]
        plt.plot(x, y, label=alg, linestyle='solid', marker=markers[counter])
        counter += 1
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.125), ncol=3, frameon=False, fontsize=14)
    plt.ylim([0, 500])
    print(beautiful_x_ticks)
    plt.xticks(x, beautiful_x_ticks, fontsize=14)
    plt.xlabel('Dataset Dimensionality', fontsize=14)
    plt.ylabel('Runtime (sec)', fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(lw=0.3, zorder=0, linestyle='dotted')
    outputfile = os.path.join(args.savedir, 'hics_runtime.png')
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)
    plt.savefig(outputfile, dpi=300, bbox_inches='tight', pad_inches=0.3)
    plt.clf()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""Plot benchmark results.""")
    parser.add_argument('-r', '--results', help='The results file (json file).', default=None)
    parser.add_argument('-dimexp', help='-dimexp dir', default=None)
    parser.add_argument('-sdir', '--savedir', help='The directory of the created plots.', required=True)
    parser.add_argument('-f', '--plot_features', default=None)
    parser.add_argument('-t', '--title', default=None)
    parser.add_argument('-rt', '--run_time', default=None, help='-rt <logs dir>')
    args = parser.parse_args()
    if args.results is not None:
        draw_barplot(args)
    if args.dimexp is not None:
        plot_dim_experiment(args)
    if args.plot_features is not None:
        plot_features(args)
    if args.run_time is not None:
        plot_runtime_cluster_size(args)
