'''

Main interactive model syllable statistics results application functionality.
This module facilitates the interactive functionality for the statistics plotting, and
 transition graph features.

'''

import os
import warnings
import numpy as np
import pandas as pd
from collections import defaultdict
from IPython.display import clear_output
from moseq2_viz.util import get_sorted_index, read_yaml
from moseq2_viz.info.util import transition_entropy
from moseq2_app.util import merge_labels_with_scalars
from scipy.cluster.hierarchy import linkage, dendrogram
from moseq2_viz.model.dist import get_behavioral_distance
from moseq2_viz.model.util import (parse_model_results, relabel_by_usage, normalize_usages,
                                   sort_syllables_by_stat, sort_syllables_by_stat_difference)
from moseq2_app.stat.widgets import SyllableStatWidgets, TransitionGraphWidgets
from moseq2_app.stat.view import graph_dendrogram, bokeh_plotting, plot_interactive_transition_graph
from moseq2_viz.model.trans_graph import (get_trans_graph_groups, get_group_trans_mats,
                                         convert_transition_matrix_to_ebunch,
                                         convert_ebunch_to_graph, make_transition_graphs, get_pos)

class InteractiveSyllableStats(SyllableStatWidgets):
    '''

    Interactive Syllable Statistics grapher class that holds the context for the current
     inputted session.

    '''

    def __init__(self, index_path, model_path, df_path, info_path, max_sylls, load_parquet):
        '''
        Initialize the main data inputted into the current context

        Parameters
        ----------
        index_path (str): Path to index file.
        model_path (str): Path to trained model file.
        info_path (str): Path to syllable information file.
        max_sylls (int): Maximum number of syllables to plot.
        load_parquet (bool): Indicates to load previously loaded data
        '''

        super().__init__()

        self.model_path = model_path
        self.info_path = info_path
        self.max_sylls = max_sylls
        self.index_path = index_path
        self.df_path = df_path

        if load_parquet:
            if df_path is not None:
                if not os.path.exists(df_path):
                    self.df_path = None
        else:
            self.df_path = None

        self.df = None

        # Load Syllable Info
        self.syll_info = read_yaml(self.info_path)

        self.results = None
        self.icoord, self.dcoord = None, None
        self.cladogram = None

        # Load all the data
        self.interactive_stat_helper()
        self.df = self.df[self.df['syllable'] < self.max_sylls]
        self.session_names = sorted(list(self.df.SessionName.unique()))
        self.subject_names = sorted(list(self.df.SubjectName.unique()))

        # Update the widget values
        self.session_sel.options = self.session_names
        self.session_sel.value = [self.session_sel.options[0]]

        self.ctrl_dropdown.options = list(self.df.group.unique())
        self.exp_dropdown.options = list(self.df.group.unique())
        self.exp_dropdown.value = self.ctrl_dropdown.options[-1]

        self.dropdown_mapping = {
            'usage': 'usage',
            'distance to center': 'dist_to_center_px',
            '2d velocity': 'velocity_2d_mm',
            '3d velocity': 'velocity_3d_mm',
            'height': 'height_ave_mm',
            'similarity': 'similarity',
            'difference': 'difference',
        }

        self.clear_button.on_click(self.clear_on_click)
        self.grouping_dropdown.observe(self.on_grouping_update, names='value')

    def clear_on_click(self, b):
        '''
        Clears the cell output

        Parameters
        ----------
        b (button click)

        Returns
        -------
        '''

        clear_output()

    def on_grouping_update(self, event):
        '''
        Updates the MultipleSelect widget upon selecting groupby == SubjectName or SessionName.
        Hides it if groupby == group.

        Parameters
        ----------
        event (user clicks new grouping)

        Returns
        -------
        '''

        if event.new == 'SessionName':
            self.session_sel.layout.display = "flex"
            self.session_sel.layout.align_items = 'stretch'
            self.session_sel.options = self.session_names
        elif event.new == 'SubjectName':
            self.session_sel.layout.display = "flex"
            self.session_sel.layout.align_items = 'stretch'
            self.session_sel.options = self.subject_names
        else:
            self.session_sel.layout.display = "none"

        self.session_sel.value = [self.session_sel.options[0]]

    def compute_dendrogram(self):
        '''
        Computes the pairwise distances between the included model AR-states, and
        generates the graph information to be plotted after the stats.

        Returns
        -------
        '''
        # Get Pairwise distances
        X = get_behavioral_distance(self.sorted_index,
                                    self.model_path,
                                    max_syllable=self.max_sylls,
                                    distances='ar[init]')['ar[init]']
        Z = linkage(X, 'complete')

        # Get Dendrogram Metadata
        self.results = dendrogram(Z, distance_sort=False, no_plot=True, get_leaves=True)

        # Get Graph layout info
        icoord, dcoord = self.results['icoord'], self.results['dcoord']

        icoord = pd.DataFrame(icoord) - 5
        icoord = icoord * (self.df['syllable'].max() / icoord.max().max())
        self.icoord = icoord.values

        dcoord = pd.DataFrame(dcoord)
        dcoord = dcoord * (self.df['usage'].max() / dcoord.max().max())
        self.dcoord = dcoord.values

    def interactive_stat_helper(self):
        '''
        Computes and saves the all the relevant syllable information to be displayed.
         Loads the syllable information dict and merges it with the syllable statistics DataFrame.

        Returns
        -------
        '''
        # Read syllable information dict
        syll_info = read_yaml(self.info_path)

        # Getting number of syllables included in the info dict
        max_sylls = len(self.syll_info)
        for k in range(max_sylls):
            # remove group_info
            syll_info[k].pop('group_info', None)

        info_df = pd.DataFrame(syll_info).T.sort_index()
        info_df['syllable'] = info_df.index

        # Load the model and sort labels - also remaps the ar matrices
        model_data = parse_model_results(self.model_path, sort_labels_by_usage=True, count='usage')

        # Read index file
        self.sorted_index = get_sorted_index(self.index_path)

        if set(self.sorted_index['files']) != set(model_data['metadata']['uuids']):
            print('Error: Index file UUIDs do not match model UUIDs.')

        # Get max syllables if None is given
        if self.max_sylls is None:
            self.max_sylls = max_sylls

        if self.df_path is not None:
            print('Loading parquet files')
            df = pd.read_parquet(self.df_path, engine='fastparquet')
        else:
            print('Syllable DataFrame not found. Computing syllable statistics...')
            df, _ = merge_labels_with_scalars(self.sorted_index, self.model_path)

        self.df = df.merge(info_df, on='syllable')
        self.df['SubjectName'] = self.df['SubjectName'].astype(str)
        self.df['SessionName'] = self.df['SessionName'].astype(str)

    def interactive_syll_stats_grapher(self, stat, sort, groupby, errorbar, sessions, ctrl_group, exp_group):
        '''
        Helper function that is responsible for handling ipywidgets interactions and updating the currently
         displayed Bokeh plot.

        Parameters
        ----------
        stat (str or ipywidgets.DropDown): Statistic to plot: ['usage', 'distance to center']
        sort (str or ipywidgets.DropDown): Statistic to sort syllables by (in descending order).
            ['usage', 'distance to center', 'similarity', 'difference'].
        groupby (str or ipywidgets.DropDown): Data to plot; either group averages, or individual session data.
        errorbar (str or ipywidgets.DropDown): Error bar to display. ['CI 95%' ,'SEM', 'STD']
        sessions (list or ipywidgets.MultiSelect): List of selected sessions to display data from.
        ctrl_group (str or ipywidgets.DropDown): Name of control group to compute group difference sorting with.
        exp_group (str or ipywidgets.DropDown): Name of comparative group to compute group difference sorting with.

        Returns
        -------
        '''

        # Get current dataFrame to plot
        df = self.df

        # Handle names to query DataFrame with
        stat = self.dropdown_mapping[stat.lower()]
        sortby = self.dropdown_mapping[sort.lower()]

        # Get selected syllable sorting
        if sort.lower() == 'difference':
            # display Text for groups to input experimental groups
            ordering = sort_syllables_by_stat_difference(df, ctrl_group, exp_group, stat=stat)
        elif sort.lower() == 'similarity':
            ordering = self.results['leaves']
        elif sort.lower() != 'usage':
            ordering, _ = sort_syllables_by_stat(df, stat=sortby)
        else:
            ordering = range(len(df.syllable.unique()))

        # Handle selective display for whether mutation sort is selected
        if sort.lower() == 'difference':
            self.mutation_box.layout.display = "block"
            sort = f'Difference: {self.exp_dropdown.value} - {self.ctrl_dropdown.value}'
        else:
            self.mutation_box.layout.display = "none"

        # Handle selective display to select included sessions to graph
        if groupby == 'SessionName' or groupby == 'SubjectName':
            mean_df = df.copy()
            df = df[df[groupby].isin(self.session_sel.value)]
        else:
            mean_df = None

        # Compute cladogram if it does not already exist
        if self.cladogram is None:
            self.cladogram = graph_dendrogram(self, self.syll_info)
            self.results['cladogram'] = self.cladogram

        self.stat_fig = bokeh_plotting(df, stat, ordering, mean_df=mean_df, groupby=groupby,
                                       errorbar=errorbar, syllable_families=self.results, sort_name=sort)


class InteractiveTransitionGraph(TransitionGraphWidgets):
    '''

    Interactive transition graph class used to facilitate interactive graph generation
    and thresholding functionality.

    '''

    def __init__(self, model_path, index_path, info_path, df_path, max_sylls, plot_vertically, load_parquet):
        '''
        Initializes context variables

        Parameters
        ----------
        model_path (str): Path to trained model file
        index_path (str): Path to index file containing trained session metadata.
        info_path (str): Path to labeled syllable info file
        max_sylls (int): Maximum number of syllables to plot.
        '''

        super().__init__()

        self.model_path = model_path
        self.index_path = index_path
        self.info_path = info_path
        self.df_path = df_path
        self.max_sylls = max_sylls
        self.plot_vertically = plot_vertically

        if load_parquet:
            if df_path is not None and not os.path.exists(df_path):
                self.df_path = None
        else:
            self.df_path = None

        # Load Model
        self.model_fit = parse_model_results(model_path)

        # Load Index File
        self.sorted_index = get_sorted_index(index_path)

        if set(self.sorted_index['files']) != set(self.model_fit['metadata']['uuids']):
            print('Warning: Index file UUIDs do not match model UUIDs.')

        # Load and store transition graph data
        self.initialize_transition_data()

        self.set_range_widget_values()

        # Set color dropdown callback
        self.color_nodes_dropdown.observe(self.on_set_scalar, names='value')

        self.clear_button.on_click(self.clear_on_click)

        # Manage dropdown menu values
        self.scalar_dict = {
            'Default': 'speeds_2d',
            '2D velocity': 'speeds_2d',
            '3D velocity': 'speeds_3d',
            'Height': 'heights',
            'Distance to Center': 'dists'
        }

    def clear_on_click(self, b):
        '''
        Clears the cell output

        Parameters
        ----------
        b (button click)

        Returns
        -------
        '''

        clear_output()

    def set_range_widget_values(self):
        '''
        After the dataset is initialized, the threshold range sliders' values will be set
         according to the standard deviations of the dataset.

        Returns
        -------
        '''

        # Update threshold range values
        edge_threshold_stds = int(np.max(self.trans_mats) / np.std(self.trans_mats))
        usage_threshold_stds = int(self.df['usage'].max() / self.df['usage'].std()) + 2
        speed_threshold_stds = int(self.df['velocity_2d_mm'].max() / self.df['velocity_2d_mm'].std()) + 2

        self.edge_thresholder.options = [float('%.3f' % (np.std(self.trans_mats) * i)) for i in
                                         range(edge_threshold_stds)]
        self.edge_thresholder.index = (1, edge_threshold_stds - 1)

        self.usage_thresholder.options = [float('%.3f' % (self.df['usage'].std() * i)) for i in
                                          range(usage_threshold_stds)]
        self.usage_thresholder.index = (0, usage_threshold_stds - 1)

        self.speed_thresholder.options = [float('%.3f' % (self.df['velocity_2d_mm'].std() * i)) for i in
                                          range(speed_threshold_stds)]
        self.speed_thresholder.index = (0, speed_threshold_stds - 1)

    def on_set_scalar(self, event):
        '''
        Updates the scalar threshold slider filter criteria according to the current node coloring.
        Changes the name of the slider as well.

        Parameters
        ----------
        event (dropdown event): User changes selected dropdown value

        Returns
        -------
        '''

        if event.new == 'Default' or event.new == '2D velocity':
            key = 'velocity_2d_mm'
            self.speed_thresholder.description = 'Threshold Nodes by 2D Velocity'
        elif event.new == '2D velocity':
            key = 'velocity_2d_mm'
            self.speed_thresholder.description = 'Threshold Nodes by 2D Velocity'
        elif event.new == '3D velocity':
            key = 'velocity_3d_mm'
            self.speed_thresholder.description = 'Threshold Nodes by 3D Velocity'
        elif event.new == 'Height':
            key = 'height_ave_mm'
            self.speed_thresholder.description = 'Threshold Nodes by Height'
        elif event.new == 'Distance to Center':
            key = 'dist_to_center_px'
            self.speed_thresholder.description = 'Threshold Nodes by Distance to Center'
        else:
            key = 'velocity_2d_mm'
            self.speed_thresholder.description = 'Threshold Nodes by 2D Velocity'

        scalar_threshold_stds = int(self.df[key].max() / self.df[key].std()) + 2
        self.speed_thresholder.options = [float('%.3f' % (self.df[key].std() * i)) for i in
                                          range(scalar_threshold_stds)]
        self.speed_thresholder.index = (0, scalar_threshold_stds - 1)

    def compute_entropies(self, labels, label_group):
        '''
        Compute individual syllable entropy and transition entropy rates for all sessions with in a label_group.

        Parameters
        ----------
        labels (2d list): list of session syllable labels over time.
        label_group (list): list of groups computing entropies for.

        Returns
        -------
        '''

        self.incoming_transition_entropy, self.outgoing_transition_entropy = [], []

        for g in self.group:
            use_labels = [lbl for lbl, grp in zip(labels, label_group) if grp == g]

            self.incoming_transition_entropy.append(np.mean(transition_entropy(use_labels,
                                                    tm_smoothing=0,
                                                    truncate_syllable=self.max_sylls,
                                                    transition_type='incoming',
                                                    relabel_by='usage'), axis=0))

            self.outgoing_transition_entropy.append(np.mean(transition_entropy(use_labels,
                                                    tm_smoothing=0,
                                                    truncate_syllable=self.max_sylls,
                                                    transition_type='outgoing',
                                                    relabel_by='usage'), axis=0))

    def compute_entropy_differences(self):
        '''
        Computes cross group entropy/entropy-rate differences
         and casts them to OrderedDict objects

        Returns
        -------
        '''

        # Compute entropy + entropy rate differences
        for i in range(len(self.group)):
            for j in range(i + 1, len(self.group)):
                self.incoming_transition_entropy.append(self.incoming_transition_entropy[j] - self.incoming_transition_entropy[i])
                self.outgoing_transition_entropy.append(self.outgoing_transition_entropy[j] - self.outgoing_transition_entropy[i])

    def initialize_transition_data(self):
        '''
        Performs all necessary pre-processing to compute the transition graph data and
         syllable metadata to display via HoverTool.
        Stores the syll_info dict, groups to explore, maximum number of syllables, and
         the respective transition graphs and syllable scalars associated.

        Returns
        -------
        '''
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            # Load Syllable Info
            self.syll_info = read_yaml(self.info_path)

            # Get labels and optionally relabel them by usage sorting
            labels = self.model_fit['labels']

            # get max_sylls
            if self.max_sylls is None:
                self.max_sylls = len(self.syll_info)

            if self.df_path is not None:
                print('Loading parquet files')
                df = pd.read_parquet(self.df_path, engine='fastparquet')
            else:
                print('Syllable DataFrame not found. Creating new dataframe and computing syllable statistics...')
                df, _ = merge_labels_with_scalars(self.sorted_index, self.model_path)
            self.df = df

            # Get groups and matching session uuids
            label_group, _ = get_trans_graph_groups(self.model_fit)
            self.group = list(set(label_group))

            labels = relabel_by_usage(labels, count='usage')[0]

            self.compute_entropies(labels, label_group)

            # Compute usages and transition matrices
            self.trans_mats, self.usages = get_group_trans_mats(labels, label_group, self.group, self.max_sylls)
            self.df = self.df[self.df['syllable'] < self.max_sylls]
            self.df = self.df.groupby(['group', 'syllable'], as_index=False).mean()

            self.compute_entropy_differences()

    def interactive_transition_graph_helper(self, layout, scalar_color, edge_threshold, usage_threshold, speed_threshold):
        '''

        Helper function that generates all the transition graphs given the currently selected
        thresholding values, then displays them in a Jupyter notebook or web page.

        Parameters
        ----------
        edge_threshold (tuple or ipywidgets.FloatRangeSlider): Transition probability range to include in graphs.
        usage_threshold (tuple or ipywidgets.FloatRangeSlider): Syllable usage range to include in graphs.
        speed_threshold (tuple or ipywidgets.FloatRangeSlider): Syllable speed range to include in graphs.

        Returns
        -------
        '''
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            # Get graph node anchors
            anchor = 0
            # make a list of normalized usages
            usages = [normalize_usages(u) for u in self.usages]
            usages_anchor = usages[anchor]

            # Get anchored group scalars
            scalars = defaultdict(list)
            _scalar_map = {
                'speeds_2d': 'velocity_2d_mm',
                'speeds_3d': 'velocity_3d_mm',
                'heights': 'height_ave_mm',
                'dists': 'dist_to_center_px'
            }
            # loop thru each group and append a syllable -> scalar value mapping to collection above
            for g in self.group:
                group_df = self.df.query('group == @g').set_index('syllable')
                for new_scalar, old_scalar in _scalar_map.items():
                    scalars[new_scalar].append(dict(group_df[old_scalar]))

            key = self.scalar_dict.get(scalar_color, 'speeds_2d')
            scalar_anchor = scalars[key][anchor]

            usage_kwargs = {
                'usages': usages_anchor,
                'usage_threshold': usage_threshold
            }
            speed_kwargs = {
                'speeds': scalar_anchor,
                'speed_threshold': speed_threshold
            }

            # Create graph with nodes and edges
            ebunch_anchor, orphans = convert_transition_matrix_to_ebunch(
                self.trans_mats[anchor], self.trans_mats[anchor], edge_threshold=edge_threshold,
                keep_orphans=True, max_syllable=self.max_sylls, **usage_kwargs, **speed_kwargs)
            indices = [e[:-1] for e in ebunch_anchor]

            # Get graph anchor
            graph_anchor = convert_ebunch_to_graph(ebunch_anchor)

            pos = get_pos(graph_anchor, layout=layout, nnodes=self.max_sylls)

            # make transition graphs
            group_names = self.group.copy()

            # prepare transition graphs
            usages, group_names, _, _, _, graphs, scalars = make_transition_graphs(self.trans_mats,
                                                                                usages[:len(self.group)],
                                                                                self.group,
                                                                                group_names,
                                                                                pos=pos,
                                                                                indices=indices,
                                                                                orphans=orphans,
                                                                                edge_threshold=edge_threshold,
                                                                                arrows=True,
                                                                                scalars=scalars,
                                                                                usage_kwargs=usage_kwargs,
                                                                                speed_kwargs=speed_kwargs)

            # interactive plot transition graphs
            plot_interactive_transition_graph(graphs, pos, self.group,
                                            group_names, usages, self.syll_info,
                                            self.incoming_transition_entropy, self.outgoing_transition_entropy,
                                            scalars=scalars, scalar_color=scalar_color, plot_vertically=self.plot_vertically)