__author__ = 'Miroslav Kuchta <mirok@math.uio.no>'
__date__ = '2014-04-23'
__copyright__ = 'Copyright (C) 2013 ' + __author__
__license__ = 'GNU Lesser GPL version 3 or any later version'

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from meshentityhandler import MeshEntityHandler
from dofhandler import DofHandler
from dolfin import facets, Edge


class DofMapPlot(object):

    '''DofMapPlot is a figure with mesh and handlers for plotting dofs and
    mesh entities.'''

    def __init__(self, dmp, options):
        'Create mesh figure and attach the handlers.'
        self.mpi_size = dmp.mpi_size
        # Plot the mesh
        mesh = dmp.V.mesh()
        figure = self._plot_mesh(mesh, options)

        # Attach handler of mesh entity events to figure
        self.mesh_entity_handler = MeshEntityHandler(figure, dmp, options)

        # Attach handler of dof events to figure
        self.dof_handler = DofHandler(figure, dmp, options)

    def _plot_mesh(self, mesh, options):
        'Plot the mesh.'
        gdim = mesh.geometry().dim()
        if gdim < 2:
            raise ValueError('Invalid geometrical dimension. Must be > 1.')

        # Optionally turn on xkcd
        xkcd = options['xkcd']
        if xkcd:
            plt.xkcd()
        fig = plt.figure()

        # Get axes based on 2d or 3d
        ax = fig.gca(projection='3d') if gdim > 2 else fig.gca()

        # Prepare axis labels
        for i in range(gdim):
            x = chr(ord('x') + i)
            label = x if xkcd else ('$%s$' % x)
            eval("ax.set_%slabel('%s')" % (x, label))

        # Get colors for plotting edges
        cmap = plt.get_cmap(options['colors']['mesh'])
        edge_color = cmap(cmap.N/2)
        bdr_edge_color = cmap(9*cmap.N/10)

        # Plot the edges and get min/max of coordinate axis
        x_min_max = self._plot_edges(ax, mesh, edge_color, bdr_edge_color)

        # Fix the limits for figure
        for i in range(gdim):
            xi_min, xi_max = x_min_max[i]
            eval('ax.set_%slim(%g, %g)' % (chr(ord('x') + i), xi_min, xi_max))

        return fig

    def _plot_edges(self, ax, mesh, edge_color, bdr_edge_color):
        'Compute inner and boundary edges of the mesh and plot them.'
        # Create edges
        tdim = mesh.topology().dim()
        mesh.init(1)
        bdr_edges = set([])

        # Compute boundary edges for inter-process boundaries
        if self.mpi_size > 1:
            # Facet cell connectivity, 2d = edge->cell, 3d = facet->cell
            mesh.init(tdim-1, tdim)
            # In 2d bdr edge has different number of local and global cells
            if tdim == 2:
                bdr_edges =\
                    set([f.index() for f in facets(mesh)
                         if
                         f.num_entities(tdim) != f.num_global_entities(tdim)])
            # In 3d bdr edge belongs to facet who has different number of local
            # and global cells
            else:
                # Face -> edge connectivity
                mesh.init(tdim-1, 1)
                bdr_edges =\
                    set(map(int, sum([f.entities(1).tolist()
                                      for f in facets(mesh)
                                      if
                                      f.num_entities(tdim) !=
                                      f.num_global_entities(tdim)],
                                     [])))

        # Plot inner_edges
        inner_edges = set(range(mesh.size(1))) - bdr_edges
        x_min_max =\
            self._plot_edges_from_list(ax, mesh, inner_edges, edge_color)

        # Plot boundary edges
        if bdr_edges:
            self._plot_edges_from_list(ax, mesh, bdr_edges, bdr_edge_color)

        return x_min_max

    def _plot_edges_from_list(self, ax, mesh, edges, color):
        'Create and plot edges of mesh from list of local indices.'
        gdim = mesh.geometry().dim()
        n_vertices = mesh.num_vertices()
        coordinates = mesh.coordinates().reshape((n_vertices, gdim))

        # Compute min/max of axis
        x_min_max = []
        for i in range(gdim):
            x_min_max.append([coordinates[:, i].min(),
                              coordinates[:, i].max()])

        for edge in edges:
            vertex_indices = Edge(mesh, edge).entities(0)
            edge_coordinates = [[] for i in range(gdim)]
            for vertex_index in vertex_indices:
                for i in range(gdim):
                    edge_coordinates[i].append(coordinates[vertex_index, i])
            ax.plot(*edge_coordinates, color=color)

        return x_min_max
