__author__ = 'Miroslav Kuchta <mirok@math.uio.no>'
__date__ = '2014-04-23'
__copyright__ = 'Copyright (C) 2013 ' + __author__
__license__ = 'GNU Lesser GPL version 3 or any later version'

from dofmaphandler import DofMapHandler
from common import x_to_str, subspace_index
from dolfin import Cell
import time


class DofHandler(DofMapHandler):

    'Handle events related to degrees of freedom.'

    def __init__(self, fig, dmp, options):
        # Connect figure to self
        DofMapHandler.__init__(self, dmp, options)
        fig.canvas.mpl_connect('key_press_event', self)

        # Get axes based on 2d or 3d
        gdim = self.mesh.geometry().dim()
        ax = fig.gca(projection='3d') if gdim > 2 else fig.gca()
        self.axes = ax
        self.fig = fig

        # Get the markersize
        self.markersize = options['markersize'] if not options['xkcd'] else 0

        # See which dofmaps will be plotted and which subspaces are used
        self.component = options['component']
        self.dofmaps = dmp.dofmaps
        self.bounds = dmp.bounds
        # Get ownership range. To be used with global ordering scheme
        self.first_dof = self.dofmaps[0].ownership_range()[0]
        # Get indices of subspaces to which component belongs
        self.subspace_index = subspace_index(self.component, self.bounds)
        # Unique subspaces
        self.subspaces = set(self.subspace_index)

        # Plot of dofs is a scatter for node/dof position
        # and text for label.
        self.showing_all_dofs = False
        self.scatter_objects = {}
        self.text_objects = {}

        # Scatter objects require position
        self.positions = {}

        # Taxt objects require numbering of dofs and position
        self.labels = {}

        # Notify in terminal and set the window title
        self.printer('Plotting dofmaps ' + str(self.component), 'blue')

        title = 'Figure %d : Dofs of %s' % (self.fig_num, str(self.component))
        if self.mpi_size > 1:
            title = ' '.join([title, 'on process %d' % self.mpi_rank])
        fig.canvas.set_window_title(title)

    def __call__(self, event):
        'Make actions based on event key.'
        # Plot dof locations and labels
        if event.key in ['d', 'D', 'ctrl+d']:
            self._dof_plot(event)
        else:
            # Ignore all other keys presses,
            # Keys h, i are handled by MeshEntityHandler
            pass

    def _dof_plot(self, event):
        'Plot degrees of freedom.'
        # Plot dofs in cell with event
        if (event.key == 'd') and not self.showing_all_dofs:
            cell_indices = self._locate_event(event)
            for cell_index in cell_indices:
                self._cell_dof_plot(cell_index)
            self.fig.canvas.draw()

        # Plot dofs in all cells of mesh
        elif (event.key == 'D') and not self.showing_all_dofs:
            start = time.time()
            self.printer('Processing ...', 'blue')
            cell_indices = xrange(self.mesh.num_cells())
            for cell_index in cell_indices:
                self._cell_dof_plot(cell_index)
            self.fig.canvas.draw()
            self.showing_all_dofs = True
            stop = time.time()
            self.printer('\t\tdone in %.2f seconds.' % (stop - start), 'blue')

        # Remove any existing dof labels/positions
        elif not((event.key == 'd') or (event.key == 'D')):
            [so.remove() for so in self.scatter_objects.itervalues()]

            [to.set_visible(False) for to in self.text_objects.itervalues()]

            self.showing_all_dofs = False
            self.scatter_objects = {}
            self.text_objects = {}
            self.positions = {}
            self.labels = {}
            self.fig.canvas.draw()

    def _cell_dof_plot(self, cell_index):
        'Plot degrees of freedom in the cell.'
        # Get the positions and labels that were changed in the cell
        changed_positions, changed_labels = self._get_changes(cell_index)

        # Update changed positions and labels
        # Hide old scatter, delete and create new scatter object
        for cp in changed_positions:
            if cp in self.scatter_objects:
                self.scatter_objects[cp].remove()
                del self.scatter_objects[cp]
            else:
                # Get the positions from dictionary
                dof_x = self.positions[cp]
                self.scatter_objects[cp] = self.axes.scatter(*dof_x, c='k',
                                                             s=self.markersize)

        # Hide old text object, create the new label and use it to make new
        # text object
        for cl in changed_labels:
            if cl in self.text_objects:
                self.text_objects[cl].set_visible(False)
                del self.text_objects[cl]

            text = self._make_text(self.labels[cl])
            # Combine text with position to make the label
            x_text = sum([self.positions[cl], [text]], [])
            self.text_objects[cl] = self.axes.text(*x_text)

    def _get_changes(self, cell_index):
        'Get identifiers for positions and labels that were changed in cell.'
        changed_positions = []
        changed_labels = []
        order = self.order
        first_dof = self.first_dof

        for i, j in enumerate(self.component):
            cell = Cell(self.mesh, cell_index)
            dofs = self.dofmaps[j].cell_dofs(cell_index)
            dofs_x = self.dofmaps[j].tabulate_coordinates(cell)
            for dof, dof_x in zip(dofs, dofs_x):
                dof_x_str = x_to_str(dof_x)
                # Append to change if dof position was not plotted yet
                if not (dof_x_str in self.positions):
                    self.positions[dof_x_str] = dof_x.tolist()
                    changed_positions.append(dof_x_str)

                dof = dof if order == 'global' else dof - first_dof
                # If dof label was no created yet create structure for label
                if not(dof_x_str in self.labels):
                    self.labels[dof_x_str] =\
                        [[] for k in range(len(self.dofmaps))]
                    self.labels[dof_x_str][i].append(dof)
                    changed_labels.append(dof_x_str)
                else:
                    # We have dofmaps that have dofs in shared position
                    # Structure created so just append
                    if not(dof in self.labels[dof_x_str][i]):
                        self.labels[dof_x_str][i].append(dof)
                        changed_labels.append(dof_x_str)

        # Make unique
        changed_labels = set(changed_labels)
        return changed_positions, changed_labels

    def _make_text(self, labels):
        'Turn the label structure to label text.'
        # Group together components by subspaces
        text = {sub: [] for sub in self.subspaces}
        for i, comp in enumerate(self.component):
            label = labels[i]
            if len(label):
                sub = self.subspace_index[i]
                text[sub].append(str(label[0])
                                 if len(label) == 1 else str(label))

        # Finalize the label
        label_text = '$'
        for sub in text:
            if len(text[sub]):
                dofs = ','.join(text[sub])
                label_text += '(' + dofs + ')_{%d}' % sub
        label_text += '$'

        return label_text
