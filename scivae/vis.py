###############################################################################
#                                                                             #
#    This program is free software: you can redistribute it and/or modify     #
#    it under the terms of the GNU General Public License as published by     #
#    the Free Software Foundation, either version 3 of the License, or        #
#    (at your option) any later version.                                      #
#                                                                             #
#    This program is distributed in the hope that it will be useful,          #
#    but WITHOUT ANY WARRANTY; without even the implied warranty of           #
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the            #
#    GNU General Public License for more details.                             #
#                                                                             #
#    You should have received a copy of the GNU General Public License        #
#    along with this program. If not, see <http://www.gnu.org/licenses/>.     #
#                                                                             #
###############################################################################

from scivae import VAE
from sciviso import Heatmap, Histogram, Scatterplot
import seaborn as sns
from scipy import stats
from sciutil import SciUtil
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from statsmodels.stats.multitest import multipletests
import matplotlib
from matplotlib import rcParams
import pandas as pd
import os

"""
This class is build when VAE is created and allows for the plots to be generated.
"""


class Vis:

    def __init__(self, vae: VAE, sciutil: SciUtil, colours: list):
        self.vae = vae
        self.u = sciutil
        self.colours = colours if colours else ['#483873', '#1BD8A6', '#B117B7', '#AAC7E2', '#FFC107', '#016957', '#9785C0',
             '#D09139', '#338A03', '#FF69A1', '#5930B1', '#FFE884', '#35B567', '#1E88E5',
             '#ACAD60', '#A2FFB4', '#B618F5', '#854A9C']
        self.df = None
        self.label_font_size = 8
        self.title_font_size = 10
        self.title_font_weight = 700
        self.fig_size = (2, 2)
        sns.set_style('ticks')
        self.alpha = 0.6
        rcParams['figure.figsize'] = self.fig_size
        sns.set_context("paper", rc={"font.size": self.label_font_size, "axes.titlesize": self.title_font_size,
                                     "axes.labelsize": self.label_font_size,
                                     'figure.figsize': self.fig_size})
        self.style = {}

    def plot_input_distribution(self, df: pd.DataFrame, row_id="", columns=None, nbins=20, ymax=None, output_dir="",
                                c="grey", fig_type='svg', title="", show_plt=False, save_fig=True):
        columns = columns if columns is not None else [c for c in df.columns if c != row_id]

        for column in columns:
            set_style(self.fig_size, self.label_font_size)
            try:
                hist = Histogram(df, column, title=column.replace('_', ' '), bins=nbins, colour=c, max_y=ymax)
                hist.load_style(self.style)
                hist.plot()
                if save_fig:
                    plt.savefig(os.path.join(output_dir, f'{title}_Hist_{column}.{fig_type}'))
                if show_plt:
                    plt.show()
            except:
                self.u.warn_p(["plot_input_distribution: Unable to plot histogram for column:", column])

    def plot_feature_correlation(self, df: pd.DataFrame, row_id="", columns=None, output_dir="", vmin=-1, vmax=1, fig_type='svg',
                                cluster_cols=True, cluster_rows=True, print_vals=True, cmap='RdBu_r',
                                 title="RCM heatmap of feature correlation", show_plt=False, save_fig=True):
        heatmap_df = pd.DataFrame()
        lbls = []
        padj_cols = []
        columns = columns if columns is not None else [c for c in df.columns if c != row_id]
        for feature_1 in columns:
            vals = []
            pvals = []
            for feature_2 in columns:
                rho, p = stats.spearmanr(df[feature_1].values, df[feature_2].values)
                if print_vals:
                    print(feature_1, feature_2, rho, p)
                vals.append(rho)
                pvals.append(p)
            lbls.append(f'{feature_1}')
            heatmap_df[f'{feature_1}'] = vals
            reg, padj, a, b = multipletests(pvals, alpha=0.1, method='fdr_bh', returnsorted=False)
            heatmap_df[f'{feature_1} padj'] = [float(f'{p:.3f}') for p in padj] #heatmap_df[f'Node {i + 1} padj'] = [float(f'rho:{vals[ip]:.3f} p:{p:.3f}') for ip, p in enumerate(padj)]
            padj_cols.append(f'{feature_1} padj')
        heatmap_df['labels'] = lbls
        annot_values = heatmap_df[padj_cols].values
        heatmap = Heatmap(heatmap_df, lbls, 'labels',
                          title=f'{title}',
                          cluster_cols=cluster_cols, cluster_rows=cluster_rows,
                          vmin=vmin, vmax=vmax, cmap=cmap, figsize=(4, 4)
                          )
        g = heatmap.plot()
        if save_fig:
            plt.savefig(os.path.join(output_dir, f'Heatmap_{title.replace(" ", "")}.{fig_type}'))
        if show_plt:
            plt.show()
        return heatmap_df

    def plot_top_values_by_rank(self, df: pd.DataFrame, rank_colums: list, vis_columns: list, id_col: str,
                                num_values=10, cmap='RdBu_r',
                                fig_type='svg', cluster_cols=True, cluster_rows=True, output_dir="",
                                title="Ranked top values", show_plt=False, save_fig=True):
        """
        Basically, for each of the rank columns we want to plot the values (from value filter) and the
        top and bottom of the ranks. We plot the vis columns for each of these. Note the vis columns
        should all have the same range.
        """
        rank_colours = []
        order_colours = []
        value_cols = []
        heatmap_df = pd.DataFrame()
        for v_c in vis_columns:
            values = []
            gs = []

            for r_i, r_c in enumerate(rank_colums):
                # First add the bottom ones
                rank_values = df[r_c].values
                desc_sorted = (-1 * rank_values).argsort()  # Sort the values by descending order
                ids_sorted = df[id_col].values[desc_sorted]
                data_sorted = df[v_c].values[desc_sorted]
                i = 0
                for g in ids_sorted:
                    gs.append(g)
                    rank_colours.append(self.colours[r_i])
                    order_colours.append("red")
                    i += 1
                    if i >= num_values:
                        break
                values += list(data_sorted[:num_values])
                # Do the same with the top values
                rank_values = df[r_c].values
                desc_sorted = (rank_values).argsort()  # Sort the values by descending order
                ids_sorted = df[id_col].values[desc_sorted]
                data_sorted = df[v_c].values[desc_sorted]
                i = 0
                for g in ids_sorted:
                    gs.append(g)
                    rank_colours.append(self.colours[r_i])
                    order_colours.append("blue")
                    i += 1
                    if i >= num_values:
                        break
                values += list(data_sorted[:num_values])

            heatmap_df[v_c] = values
            value_cols.append(v_c)

        heatmap_df[id_col] = gs  # This should be the same irrespective since the

        heatmap = Heatmap(heatmap_df, value_cols, id_col, row_colours=[rank_colours, order_colours], title=title,
                          cluster_cols=cluster_cols, cluster_rows=cluster_rows, cmap=cmap)
        set_style(self.fig_size, self.label_font_size)
        heatmap.plot()
        if save_fig:
            plt.savefig(os.path.join(output_dir, f'Heatmap_{title.replace(" ", "_")}.{fig_type}'))
        if show_plt:
            plt.show()
        plt.show()

    def plot_node_correlation(self, output_dir="", vmin=-1, vmax=1, fig_type='svg',
                                cluster_cols=True, cluster_rows=True, print_vals=True, cmap='RdBu_r',
                                 title="RCM heatmap of node correlation", show_plt=False, save_fig=True):
        heatmap_df = pd.DataFrame()
        lbls = []
        if not self.vae:
            self.u.err_p(["plot_node_correlation: You haven't run the VAE yet! "
                          "Please run compute_vae before attempting to plot."])
            return
        data = self.vae.get_encoded_data()

        padj_cols = []
        num_nodes = len(data[0])
        for i in range(0, num_nodes):
            vals = []
            pvals = []
            for j in range(0, num_nodes):
                rho, p = stats.spearmanr(data[:, i], data[:, j])
                if print_vals:
                    print(i + 1, j + 1, rho, p)
                vals.append(rho)
                pvals.append(p)
            lbls.append(f'Node {i + 1}')
            heatmap_df[f'Node {i + 1}'] = vals
            reg, padj, a, b = multipletests(pvals, alpha=0.1, method='fdr_bh', returnsorted=False)
            heatmap_df[f'Node {i + 1} padj'] = [float(f'{p:.3f}') for p in padj]
            padj_cols.append(f'Node {i + 1} padj')
        heatmap_df['labels'] = lbls

        annot_values = heatmap_df[padj_cols].values
        heatmap = Heatmap(heatmap_df, lbls, 'labels',
                          title=f'{title}',
                          cluster_cols=cluster_cols, cluster_rows=cluster_rows,
                          vmin=vmin, vmax=vmax, cmap=cmap, figsize=self.fig_size
                          )
        set_style(self.fig_size, self.label_font_size)
        heatmap.plot()
        if save_fig:
            plt.savefig(os.path.join(output_dir, f'Heatmap_{title.replace(" ", "_")}.{fig_type}'))
        if show_plt:
            plt.show()
        return heatmap_df

    def plot_node_feature_correlation(self, df: pd.DataFrame, row_id: str, vae_data=None, columns=None, output_dir="", vmin=-1, vmax=1, fig_type='svg',
                                cluster_cols=True, cluster_rows=True, print_vals=True, encoding_type="z", cmap='RdBu_r',
                                 title="RCM heatmap of node feature correlation", show_plt=False, save_fig=True):
        set_style(self.fig_size, self.label_font_size)
        heatmap_df = pd.DataFrame()
        lbls, cols = [], []
        columns = columns if columns is not None else [c for c in df.columns if c != row_id]
        if not self.vae:
            self.u.err_p(["plot_scatters: You haven't run the VAE yet! "
                          "Please run compute_vae before attempting to plot."])
        if vae_data is None:
            scaler = MinMaxScaler(copy=True)
            data_cols = [c for c in df.columns if c != row_id]
            dataset = df[data_cols].values
            scaled_vals = scaler.fit_transform(dataset)
            data = self.vae.encode_new_data(scaled_vals, encoding_type=encoding_type)
        else:
            data = vae_data
        num_nodes = len(data[0])
        first = True
        padj_cols = []
        for i in range(0, num_nodes):
            vals = []
            pvals = []
            for column in columns:
                if first:
                    lbls.append(f'{column}')
                rho, p = stats.spearmanr(data[:, i], df[column].values)
                if print_vals:
                    print(i + 1, column, rho, p)
                vals.append(rho)
                pvals.append(p)
            cols.append(f'Node {i + 1}')
            heatmap_df[f'Node {i + 1}'] = vals
            reg, padj, a, b = multipletests(pvals, alpha=0.1, method='fdr_bh', returnsorted=False)
            heatmap_df[f'Node {i + 1} padj'] =  [float(f'{p:.3f}') for p in padj]
            padj_cols.append(f'Node {i + 1} padj')
            first = False
        heatmap_df['labels'] = lbls
        annot_values = heatmap_df[padj_cols].values
        heatmap = Heatmap(heatmap_df, cols, 'labels',
                          title=f'{title}',
                          cluster_cols=cluster_cols, cluster_rows=cluster_rows,
                          vmin=vmin, vmax=vmax, figsize=(3, 2))
        heatmap.plot()
        if save_fig:
            plt.savefig(os.path.join(output_dir, f'Heatmap_{title.replace(" ", "_")}.{fig_type}'))
        if show_plt:
            plt.show()
        set_style(self.fig_size, self.label_font_size)
        return heatmap_df

    def plot_node_hists(self, nbins=20, ymax=None, output_dir="", c="grey", fig_type='svg', method="z",
                        show_plt=False, save_fig=True, title=""):
        if not self.vae:
            self.u.err_p(["plot_node_correlation: You haven't run the VAE yet! "
                          "Please run compute_vae before attempting to plot."])
            return
        plt.clf()
        data = self.vae.get_encoded_data(method)
        num_nodes = len(data[0])
        for i in range(0, num_nodes):
            set_style(self.fig_size, self.label_font_size)
            tmp_df = pd.DataFrame()
            tmp_df[f'Node {i + 1}'] = data[:, i]
            hist = Histogram(tmp_df, f'Node {i + 1}', title=f'{title} Node {i + 1}', bins=nbins, max_y=ymax, colour=c)
            hist.load_style(self.style)
            hist.plot()

            if save_fig:
                plt.savefig(os.path.join(output_dir, f'{title}_Hist_Node-{i + 1}.{fig_type}'))
            if show_plt:
                plt.show()
            plt.clf()

    @staticmethod
    def get_row_idxs(df, row_id, row_labels, row_values):

        row_idxs = {}
        for m in row_labels:
            row_idxs[m] = []

        g_i = 0
        for g in df[row_id].values:
            i = 0
            for m in row_labels:
                if g in row_values[i]:
                    row_idxs[m].append(g_i)

                i += 1
            g_i += 1
        row_idxs_lst = []
        for g in row_labels:
            row_idxs_lst.append(row_idxs.get(g))
        return row_idxs_lst

    def plot_feature_scatters(self, df: pd.DataFrame, row_id: str, vae_data=None, columns=None, output_dir="", fig_type='svg',
                              vmin=None, vmax=None,
                              title="latent space", show_plt=False, save_fig=True, encoding_type="z", angle_plot=0, cmap='RdBu_r'):
        set_style(self.fig_size, self.label_font_size)
        columns = columns if columns is not None else [c for c in df.columns if c != row_id]
        if not self.vae:
            self.u.err_p(["plot_scatters: You haven't run the VAE yet! "
                          "Please run compute_vae before attempting to plot."])
        if vae_data is None:
            scaler = MinMaxScaler(copy=True)
            data_cols = [c for c in df.columns if c != row_id]
            dataset = df[data_cols].values
            scaled_vals = scaler.fit_transform(dataset)
            data = self.vae.encode_new_data(scaled_vals, encoding_type=encoding_type)
        else:
            data = vae_data
        num_nodes = len(data[0])
        if num_nodes > 3:
            self.u.warn_p(["plot_scatters", "You have more than three latent nodes: ", num_nodes,
                           "\n We will only be plotting the first three"])
        if num_nodes < 2:
            self.u.warn_p(["plot_scatters", "You have less than two nodes: ", num_nodes,
                           "\nWe won't plot this ... returning"])
            return
        for column in columns:
            vis_df = pd.DataFrame()
            vis_df[row_id] = df[row_id].values
            vis_df[f'Dim. 1'] = data[:, 0]
            vis_df[f'Dim. 2'] = data[:, 1]
            if num_nodes == 2:
                scatter = Scatterplot(vis_df, f'Dim. 1', f'Dim. 2', f'{column} {title}', f'Dim. 1', f'Dim. 2',
                                      colour=pd.to_numeric(df[column].values), )
            else:
                vis_df[f'Dim. 3'] = data[:, 2]
                scatter = Scatterplot(vis_df, f'Dim. 1', f'Dim. 2', f'{column} {title}', f'Dim. 1', f'Dim. 2', z=f'Dim. 3',
                                      colour=pd.to_numeric(df[column].values))
            ax = scatter.plot()
            if save_fig:
                plt.savefig(os.path.join(output_dir, f'Scatter_{column}_{title.replace(" ", "_")}.{fig_type}'))
            if show_plt:
                plt.show()
            if angle_plot:
                # Plot different angles
                for ii in range(0, 360, angle_plot):
                    ax.view_init(elev=10., azim=ii)
                    plt.savefig(os.path.join(output_dir, f'Scatter_{column}_{title.replace(" ", "_")}_{ii}.{fig_type}'))

    def plot_values_on_scatters(self, df: pd.DataFrame, row_id: str, row_labels: list, row_values: list, vae_data=None,
                                output_dir="", fig_type='svg', title="RCM genes on latent space", plt_bg=False,
                                show_plt=False,
                                save_fig=True, color_map=None, angle_plot=None, encoding_type="z"):
        set_style(self.fig_size, self.label_font_size)
        marker_idxs = self.get_row_idxs(df, row_id, row_labels, row_values)

        if vae_data is None:
            if not self.vae:
                self.u.err_p(["plot_scatters: You haven't run the VAE yet! "
                              "Please run compute_vae before attempting to plot."])
            scaler = MinMaxScaler(copy=True)
            data_cols = [c for c in df.columns if c != row_id]
            dataset = df[data_cols].values
            scaled_vals = scaler.fit_transform(dataset)
            data = self.vae.encode_new_data(scaled_vals, encoding_type=encoding_type)
        else:
            data = vae_data
        num_nodes = len(data[0])
        if num_nodes > 3:
            self.u.warn_p(["plot_scatters", "You have more than three latent nodes: ", num_nodes,
                           "\n We will only be plotting the first three"])
        if num_nodes < 2:
            self.u.warn_p(["plot_scatters", "You have less than two nodes: ", num_nodes,
                           "\nWe won't plot this ... returning"])
            return
        i = 0
        if color_map is None:
            color_map = []
            for c in row_labels:
                color_map.append(self.colours[i])
                i += 1
                if i > len(self.colours):
                    i = 0

        vis_df = pd.DataFrame()
        vis_df[row_id] = df[row_id].values
        vis_df[f'Dim. 1'] = data[:, 0]
        vis_df[f'Dim. 2'] = data[:, 1]
        if num_nodes == 2:
            scatter = Scatterplot(vis_df, f'Dim. 1', f'Dim. 2', f'{title}', f'Dim. 1', f'Dim. 2')
            ax = scatter.plot_groups_2D(row_labels, marker_idxs, color_map, alpha_bg=0.1)
            plt.savefig(os.path.join(output_dir, f'Scatter-genes_{title.replace(" ", "_")}.{fig_type}'))
        else:
            vis_df[f'Dim. 3'] = data[:, 2]
            scatter = Scatterplot(vis_df, f'Dim. 1', f'Dim. 2', f'{title}', f'Dim. 1', f'Dim. 2', z=f'Dim. 3')
            ax = scatter.plot_groups_3D(row_labels, marker_idxs, color_map, alpha_bg=0.1)
            plt.savefig(os.path.join(output_dir, f'Scatter-genes_{title.replace(" ", "_")}.{fig_type}'))
        if show_plt:
            plt.show()
        if angle_plot:
            # Plot different angles
            for ii in range(0, 360, angle_plot):
                ax.view_init(elev=10., azim=ii)
                plt.savefig(os.path.join(output_dir, f'Scatter-genes_{title.replace(" ", "_")}_{ii}.{fig_type}'))
        set_style(self.fig_size, self.label_font_size)
        return ax


def set_style(figsize, label_font_size):
    # plot
    plt.clf()
    font = {'family': 'normal', 'size': 6}
    matplotlib.rc('font', **font)
    sns.set(rc={'figure.figsize': figsize, 'font.family': 'sans-serif',
                'font.sans-serif': 'Arial', 'font.size': label_font_size}, style="ticks")
    rcParams['figure.figsize'] = (2, 2)