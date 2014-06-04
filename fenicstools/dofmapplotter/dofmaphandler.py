__author__ = 'Miroslav Kuchta <mirok@math.uio.no>'
__date__ = '2014-04-23'
__copyright__ = 'Copyright (C) 2013 ' + __author__
__license__ = 'GNU Lesser GPL version 3 or any later version'

from common import ParallelColorPrinter
from dolfin import Point


class DofMapHandler(object):
    'Parent class for DofHadler and MeshEntityHandler.'
    def __init__(self, dmp, options):
        # Customize the printer
        mpi_rank = dmp.mpi_rank
        mpi_size = dmp.mpi_size
        # Matplotlib counts the figures from 1
        fig_num = len(dmp.plots) + 1
        self.printer = ParallelColorPrinter(mpi_rank, mpi_size, fig_num)

        # Store attributes for children
        self.order = options['order']
        self.mesh = dmp.V.mesh()
        self.mpi_size = mpi_size
        self.mpi_rank = mpi_rank
        self.fig_num = fig_num

    def _locate_event(self, event):
        'Find cells(indices) where the key press was initiated.'
        # Locate all the cells that intersect event point. If locating is done
        # by compute_first_collision the cells are difficult to mark.
        bb_tree = self.mesh.bounding_box_tree()
        try:
            # Get coordinates of the event point
            x_string = self.axes.format_coord(event.xdata, event.ydata)
            split_on = ',' if x_string.find(',') > -1 else ' '
            try:
                x = map(float, [w.split('=')[1]
                                for w in x_string.split(split_on) if w])
                cell_indices = bb_tree.compute_entity_collisions(Point(*x))
            # Click outside figure?
            except ValueError:
                message = 'Are you clicking outside the active figure?'
                self.printer(message, 'pink')
                cell_indices = []
        # Click outside figure?
        except TypeError:
            message = 'Are you clicking outside the active figure?'
            self.printer(message, 'pink')
            cell_indices = []

        return cell_indices

    def _print_help(self):
        'Print the help message.'
        message =\
            '''
                              KEY FUNCTIONALITY
-------------------------------------------------------------------------------
v : vertex indices of vertices in the cell under cursor*
V : vertex indices of all the vertices of the mesh
ctr+v : clear all showing vertex indices

c : cell index for the cell under cursor*
C : cell indices of all the cells of the mesh
ctrl+c : clear all showing cell indices

e : edge indices of edges in the cell under cursor*
E : all edge indices
ctrl+e : clear all showing edge indices

t : facet indices of facets in the cell under cursor*
T : all facet indices
ctrl+t : clear all showing facet indices

d : degrees of freedom of cell under cursor*
D : degrees of freedom of all cells in the mesh
ctrl+D : clear all showing degrees of freedom

h : help menu
i : information about ordering scheme employed for figure

f : full screen (matplotlib functionality)
s : save the figure (matplotib functionality)
-------------------------------------------------------------------------------
* especially in 3D cursor position might not coincide with any cell. This
triggers warning. You might try to zoom in to help you hit something.
    '''
        self.printer(message, 'green')

    def _print_info(self):
        'Print ordering scheme used by the figure.'
        message = 'Using %s ordering scheme' % self.order
        self.printer(message, 'green')
