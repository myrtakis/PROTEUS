import argparse
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import os


def tsne_visualizer(dataset_path, savedir):
    df = pd.read_csv(dataset_path)
    final_df = df.iloc[:, 0:df.shape[1]-2]
    feat_cols = list(range(final_df.shape[1]))
    final_df['y'] = df.iloc[:, df.shape[1]-1]
    final_df['label'] = final_df['y'].apply(lambda i: str(i))
    tsne_results = TSNE(n_components=2).fit_transform(df.iloc[:, feat_cols])

    final_df['tsne-2d-one'] = tsne_results[:, 0]
    final_df['tsne-2d-two'] = tsne_results[:, 1]

    colors = ["#383838", "#FF0B04"]

    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="y",
        palette=sns.color_palette(colors),
        data=final_df,
        legend="full",
    )

    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    output = os.path.join(args.savedir, 'tsne_' + dataset_name + '.png')
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    plt.title(dataset_name)
    plt.savefig(output, dpi=300)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', '--dataset', help='Dataset to visualize.')
    parser.add_argument('-sd', '--savedir', default='figures', help='Directory to save the image.')
    args = parser.parse_args()
    if args.dataset is not None:
        tsne_visualizer(args.dataset, args.savedir)
