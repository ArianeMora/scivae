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
from scipy import stats
from sciutil import SciUtil
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from statsmodels.stats.multitest import multipletests
import pandas as pd
import os

"""
This class is build when VAE is created and allows for the plots to be generated.
"""


class Vis:

    def __init__(self, vae: VAE, sciutil: SciUtil, colours: list, config=None):
        self.vae = vae
        self.u = sciutil
        self.colours = colours if colours else ['#483873', '#1BD8A6', '#B117B7', '#AAC7E2', '#FFC107', '#016957',
                                                '#9785C0', '#D09139', '#338A03', '#FF69A1', '#5930B1', '#FFE884',
                                                '#35B567', '#1E88E5', '#ACAD60', '#A2FFB4', '#B618F5', '#854A9C']
        self.df = None
        self.base_config = {
            'font': {'family': 'normal', 'size': 6},
            'label_font_size': 7,
            'title_font_size': 10,
            'style': "ticks",
            'figsize': (3, 3),
            'alpha': 0.6
        }
        self.base_config = self.__merge_config(self.base_config, config)

    def __merge_config(self, default_config, user_config):
        # Add in default params from our base
        for c in self.base_config:
            if not default_config.get(c):
                default_config[c] = self.base_config[c]
        if user_config:
            for c in default_config:
                # Update any of the config if the user has supplied it
                user_config[c] = user_config.get(c) if user_config.get(c) is not None else default_config.get(c)
            return user_config
        return default_config

    def plot_input_distribution(self, df: pd.DataFrame, row_id="", columns=None, output_dir="",
                                fig_type='svg', title="Hist", show_plt=False, save_fig=True, user_config=None):
        columns = columns if columns is not None else [c for c in df.columns if c != row_id]
        config = {
            'figsize': (4, 4),
            'nbins': 20,
            'c': "grey",
            'max_y': None,
            'title': title
        }
        config = self.__merge_config(config, user_config)
        for column in columns:
            set_style()
            try:
                hist = Histogram(df, column, title=column.replace('_', ' '), config=config)
                hist.plot()
                if save_fig:
                    plt.savefig(os.path.join(output_dir, f'{title}_{column}.{fig_type}'))
                if show_plt:
                    plt.show()
            except:
                self.u.warn_p(["plot_input_distribution: Unable to plot histogram for column:", column])

    def plot_feature_correlation(self, df: pd.DataFrame, row_id="", columns=None, output_dir="",
                                 fig_type='svg', print_vals=True, title="RCM heatmap of feature correlation",
                                 show_plt=False, save_fig=True, user_config=None):
        heatmap_df = pd.DataFrame()
        config = {
            'figsize': (4, 4),
            'cluster_cols': True,
            'cluster_rows': True,
            'vmin': -1,
            'vmax': 1,
            'cmap': 'RdBu_r',
            'title': title
        }
        config = self.__merge_config(config, user_config)

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
            heatmap_df[f'{feature_1} padj'] = [float(f'{p:.3f}') for p in padj]
            # heatmap_df[f'Node {i + 1} padj'] = [float(f'rho:{vals[ip]:.3f} p:{p:.3f}') for ip, p in enumerate(padj)]
            padj_cols.append(f'{feature_1} padj')
        heatmap_df['labels'] = lbls
        # annot_values = heatmap_df[padj_cols].values
        heatmap = Heatmap(heatmap_df, lbls, 'labels', config=config)
        g = heatmap.plot()
        if save_fig:
            plt.savefig(os.path.join(output_dir, f'Heatmap_{title.replace(" ", "")}.{fig_type}'))
        if show_plt:
            plt.show()
        return heatmap_df

    def plot_top_values_by_rank(self, df: pd.DataFrame, rank_colums: list, vis_columns: list, id_col: str,
                                num_values=10, fig_type='svg', output_dir="", title="Ranked top values",
                                show_plt=False, save_fig=True, user_config=None):
        """
        Basically, for each of the rank columns we want to plot the values (from value filter) and the
        top and bottom of the ranks. We plot the vis columns for each of these. Note the vis columns
        should all have the same range.
        """
        config = {
            'figsize': (4, 4),
            'cluster_cols': True,
            'cluster_rows': True,
            'cmap': 'RdBu_r',
            'title': title,
        }
        config = self.__merge_config(config, user_config)
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
                          config=config)
        heatmap.plot()
        if save_fig:
            plt.savefig(os.path.join(output_dir, f'Heatmap_{title.replace(" ", "_")}.{fig_type}'))
        if show_plt:
            plt.show()
        plt.show()

    def plot_node_correlation(self, output_dir="", fig_type='svg', print_vals=True, title="RCM heatmap of node cor.",
                              show_plt=False, save_fig=True, user_config=None):
        heatmap_df = pd.DataFrame()
        config = {
            'figsize': (4, 4),
            'cluster_cols': True,
            'cluster_rows': True,
            'vmin': -1,
            'vmax': 1,
            'cmap': 'RdBu_r',
            'title': title,
        }
        config = self.__merge_config(config, user_config)
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

        # annot_values = heatmap_df[padj_cols].values
        set_style()
        heatmap = Heatmap(heatmap_df, lbls, 'labels', config=config)
        heatmap.plot()
        if save_fig:
            plt.savefig(os.path.join(output_dir, f'Heatmap_{title.replace(" ", "_")}.{fig_type}'))
        if show_plt:
            plt.show()
        return heatmap_df

    def plot_node_feature_correlation(self, df: pd.DataFrame, row_id: str, vae_data=None, columns=None, output_dir="",
                                      fig_type='svg', print_vals=True, encoding_type="z",
                                      title="RCM heatmap of node feature corr", show_plt=False, save_fig=True,
                                      user_config=None):
        set_style()
        config = {
            'figsize': (4, 4),
            'cluster_cols': True,
            'cluster_rows': True,
            'vmin': -1,
            'vmax': 1,
            'cmap': 'RdBu_r',
            'title': title,
        }
        config = self.__merge_config(config, user_config)
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
        # annot_values = heatmap_df[padj_cols].values
        set_style()
        heatmap = Heatmap(heatmap_df, cols, 'labels', config=config)
        heatmap.plot()
        if save_fig:
            plt.savefig(os.path.join(output_dir, f'Heatmap_{title.replace(" ", "_")}.{fig_type}'))
        if show_plt:
            plt.show()
        return heatmap_df

    def plot_node_hists(self, title="Hist", output_dir="", fig_type='svg', method="z", show_plt=False, save_fig=True,
                        user_config=None):
        if not self.vae:
            self.u.err_p(["plot_node_correlation: You haven't run the VAE yet! "
                          "Please run compute_vae before attempting to plot."])
            return
        config = {
            'figsize': (4, 4),
            'nbins': 20,
            'c': "grey",
            'max_y': None,
            'title': title
        }
        config = self.__merge_config(config, user_config)
        set_style()
        data = self.vae.get_encoded_data(method)
        num_nodes = len(data[0])
        for i in range(0, num_nodes):
            set_style()
            tmp_df = pd.DataFrame()
            tmp_df[f'Node {i + 1}'] = data[:, i]
            config['title'] = f'{title} Node {i + 1}'
            hist = Histogram(tmp_df, f'Node {i + 1}', config=config)
            hist.plot()

            if save_fig:
                plt.savefig(os.path.join(output_dir, f'{title}_Node-{i + 1}.{fig_type}'))
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

    def plot_feature_scatters(self, df: pd.DataFrame, row_id: str, vae_data=None, columns=None, output_dir="",
                              fig_type='svg', title="latent space", show_plt=False, save_fig=True, encoding_type="z",
                              angle_plot=90, user_config=None):
        set_style()
        config = {
            'figsize': (4, 4),
            'angle_plot': angle_plot,
            'c': "grey",
            'max_y': None,
            'title': title
        }
        config = self.__merge_config(config, user_config)
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
                                      colour=pd.to_numeric(df[column].values), config=config)
            else:
                vis_df[f'Dim. 3'] = data[:, 2]
                scatter = Scatterplot(vis_df, f'Dim. 1', f'Dim. 2', f'{column} {title}', f'Dim. 1', f'Dim. 2',
                                      z=f'Dim. 3', colour=pd.to_numeric(df[column].values), config=config)
            ax = scatter.plot()
            if save_fig:
                plt.savefig(os.path.join(output_dir, f'Scatter_{column}_{title.replace(" ", "_")}.{fig_type}'))
            if angle_plot:
                # Plot different angles
                for ii in range(0, 360, angle_plot):
                    ax.view_init(elev=10., azim=ii)
                    plt.savefig(os.path.join(output_dir, f'Scatter_{column}_{title.replace(" ", "_")}_{ii}.{fig_type}'))
            if show_plt:
                plt.show()

    def plot_values_on_scatters(self, df: pd.DataFrame, row_id: str, row_labels: list, row_values: list, vae_data=None,
                                output_dir="", fig_type='svg', title="RCM genes on latent space", show_plt=False,
                                color_map=None, save_fig=True, angle_plot=None, encoding_type="z", user_config=None):
        set_style()
        config = {
            'figsize': (4, 4),
            'angle_plot': angle_plot,
            'c': "grey",
            'max_y': None,
            'title': title,
            'vmin': None,
            'vmax': None,
            'plt_bg': False
        }
        config = self.__merge_config(config, user_config)
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
            scatter = Scatterplot(vis_df, f'Dim. 1', f'Dim. 2', f'{title}', f'Dim. 1', f'Dim. 2', config=config)
            ax = scatter.plot_groups_2D(row_labels, marker_idxs, color_map, alpha_bg=0.1)
        else:
            vis_df[f'Dim. 3'] = data[:, 2]
            scatter = Scatterplot(vis_df, f'Dim. 1', f'Dim. 2', f'{title}', f'Dim. 1', f'Dim. 2', z=f'Dim. 3',
                                  config=config)
            ax = scatter.plot_groups_3D(row_labels, marker_idxs, color_map, alpha_bg=0.1)
        if save_fig:
            plt.savefig(os.path.join(output_dir, f'Scatter-genes_{title.replace(" ", "_")}.{fig_type}'))
        if angle_plot:
            # Plot different angles
            for ii in range(0, 360, angle_plot):
                ax.view_init(elev=10., azim=ii)
                plt.savefig(os.path.join(output_dir, f'Scatter-genes_{title.replace(" ", "_")}_{ii}.{fig_type}'))
        if show_plt:
            plt.show()
        set_style()
        return ax


def set_style():
    # plot
    plt.clf()
