__author__ = 'Miroslav Kuchta <mirok@math.uio.no>'
__date__ = '2014-04-23'
__copyright__ = 'Copyright (C) 2013 ' + __author__
__license__ = 'GNU Lesser GPL version 3 or any later version'

from common import dmt_number_entities
from dofmaphandler import DofMapHandler
from matplotlib.pyplot import get_cmap
from dolfin import Cell, MeshEntity
import time


class MeshEntityHandler(DofMapHandler):

    'Handle events related to mesh entities.'

    def __init__(self, fig, dmp, options):
        # Connect handler to self
        DofMapHandler.__init__(self, dmp, options)
        fig.canvas.mpl_connect('key_press_event', self)

        mesh = self.mesh
        self.gdim = mesh.geometry().dim()
        self.tdim = mesh.topology().dim()

        # Make sure we have global indicies for all mesh entities
        for tdim in range(self.tdim+1):
            dmt_number_entities(mesh, tdim)

        # Get axes based on 2d or 3d
        self.fig = fig
        self.axes = fig.gca(projection='3d') if self.gdim > 2 else fig.gca()

        # Labels and plotting flags for mesh entities of topological
        # dimension 0, 1, 2, 3
        self.mesh_entity_labels = {i: {} for i in range(4)}
        self.showing_all_mesh_entities = {i: False for i in range(4)}

        # Color for plotting enties
        cmap = get_cmap(options['colors']['mesh_entities'])
        N = cmap.N
        self.mesh_entity_colors = {i: cmap(i**2*N/10) for i in range(4)}

    def __call__(self, event):
        'Make actions based on event key.'
        # Handle help and info only here not in DofHandler
        if event.key in ['h']:
            self._print_help()
        elif event.key in ['i']:
            self._print_info()
        else:
            tdim = self.tdim
            # Plot vertex labels, mesh entity of tdim=0
            if event.key in ['v', 'V', 'ctrl+v']:
                self._mesh_entity_plot(event, 'v', 'V', 0)
            # Plot cell labels, mesh entity of tdim=tdim
            elif event.key in ['c', 'C', 'ctrl+c']:
                self._mesh_entity_plot(event, 'c', 'C', tdim)
            # Plot edge labels, mesh entity of tdim=1
            elif event.key in ['e', 'E', 'ctrl+e']:
                self._mesh_entity_plot(event, 'e', 'E', 1)
            # Plot facet labels, mesh entity of tdim=tdim-1
            elif event.key in ['t', 'T', 'ctrl+t']:
                self._mesh_entity_plot(event, 't', 'T', tdim - 1)
            else:
                # Ignore all other keys presses
                pass

    def _mesh_entity_plot(self, event, local_key, global_key, tdim):
        'Plot labels of mesh entities of cell with event or all cells.'
        pressed_lc = event.key == local_key
        pressed_gc = event.key == global_key

        # Plot labels in cell
        if pressed_lc and not self.showing_all_mesh_entities[tdim]:
            cell_indices = self._locate_event(event)
            for cell_index in cell_indices:
                self._single_mesh_entity_plot(cell_index, tdim)

            # Update canvas
            self.fig.canvas.draw()

        # Plot labels for all cells in mesh
        elif pressed_gc and not self.showing_all_mesh_entities[tdim]:
            start = time.time()
            self.printer('Processing ...', 'blue')
            cell_indices = xrange(self.mesh.num_cells())

            for cell_index in cell_indices:
                self._single_mesh_entity_plot(cell_index, tdim)

            # Update canvas
            self.fig.canvas.draw()

            self.showing_all_mesh_entities[tdim] = True
            stop = time.time()
            self.printer('\t\tdone in %.2f seconds.' % (stop - start), 'blue')

        # Remove all the existing labels
        elif not (pressed_lc or pressed_gc):
            for label in self.mesh_entity_labels[tdim].itervalues():
                label.set_visible(False)
            self.mesh_entity_labels[tdim].clear()

            # Flag that work will have to be done again
            self.showing_all_mesh_entities[tdim] = False

            # Update canvas
            self.fig.canvas.draw()

    def _single_mesh_entity_plot(self, cell_index, tdim):
        'Plot labels of mesh entities of topological dim. that are in cell.'
        # Compute cell->entity connectivity unless cell-cell. Don't need patch
        if self.tdim == tdim:
            entity_indices = [cell_index]
        else:
            entity_indices = Cell(self.mesh, cell_index).entities(tdim)

        color = self.mesh_entity_colors[tdim]
        labels = self.mesh_entity_labels[tdim]
        # Loop through entities of the cell
        for entity_index in entity_indices:
            entity = MeshEntity(self.mesh, tdim, entity_index)
            # Midpoint is label location
            x = entity.midpoint()
            x = [x[i] for i in range(self.gdim)]
            if (self.order == 'global') and self.mpi_size > 1:
                args = x + [str(entity.global_index())]
            else:
                args = x + [str(entity_index)]
            # Create new label if entitiy not labeled already
            if not(entity_index in labels):
                labels[entity_index] = self.axes.text(*args, color=color)
