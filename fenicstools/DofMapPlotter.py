__author__ = 'Miroslav Kuchta <mirok@math.uio.no>'
__date__ = '2014-04-23'
__copyright__ = 'Copyright (C) 2013 ' + __author__
__license__ = 'GNU Lesser GPL version 3 or any later version'

from dofmapplotter import *
from dolfin import MPI
from matplotlib.pyplot import show as plt_show


class DofMapPlotter(object):

    'Plot map of degrees of freedom of finite element function space.'

    def __init__(self, V, options=None):
        # See if we have dofmap
        if not is_function_space(V):
            raise ValueError('V is not a function space.')

        # Only allow 2d and 3d meshes
        if V.mesh().geometry().dim() == 1:
            raise ValueError('Only 2d and 3d meshes are supported.')

        # Get MPI info
        try:
            from dolfin import mpi_comm_world
            self.mpi_size = MPI.size(mpi_comm_world())
            self.mpi_rank = MPI.rank(mpi_comm_world())
        except ImportError:
            self.mpi_size = MPI.num_processes()
            self.mpi_rank = MPI.process_number()

        # Analyze the space V
        self.V = V
        self.dofmaps = extract_dofmaps(self.V)
        self.bounds = bounds(self.V)

        # Rewrite default plotting options if they are provided by user
        self.options = {'colors': {'mesh_entities': 'hsv',
                                   'mesh': 'Blues'},
                        'xkcd': False,
                        'markersize': 40}
        if options is not None:
            self.options.update(options)

        # Keep track of the plots
        self.plots = []

    def _arg_check(self, component):
        '''Check if component is valid argument, i.e. subset of
        [0, self.num_dofmaps()).'''
        if type(component) is list:
            # Must be subset of [0, n_dofmaps). It will be reordered to
            # to ascending order
            for c in component:
                if not((type(c) is int) and (0 <= c < self.num_dofmaps())):
                    return False
            # Reorder
            component.sort()
            return True
        else:
            return False

    def __str__(self):
        'String representation same as FunctionSpace.print_dofmap()'
        return '\n'.join(
            ['%d: ' % i + ' '.join(map(str,
                                        sum([d.cell_dofs(i).tolist()
                                            for d in self.dofmaps], [])))
             for i in range(self.V.mesh().num_cells())])

    def plot(self, component=[], sub=None, order='global'):
        '''Plot dofmap of component(s) or subspace of V. By default dofmap of
        entire V is used. Use global or local ordering scheme for mesh entities
        and degrees of freedom.'''
        # Subspace is not specified
        if sub is None:
            # Add the options and create new plot
            plot_options = self.options.copy()
            plot_options['order'] = order

            # Convert component to list if necessary and perform validity check
            component = component if type(component) is list else [component]
            # If component is empty plot all dofmaps
            component = component if component else range(self.num_dofmaps())
            if not self._arg_check(component):
                raise ValueError('Component is not list or in [0, %d)' %
                                 self.num_dofmaps())

            plot_options['component'] = component
            self.plots.append(DofMapPlot(self, plot_options))
        # Subspace is specified
        else:
            assert not component
            assert (0 <= sub < self.num_subspaces()) and (type(sub) is int)

            # Compute component of subspace
            sub_component = range(self.bounds[sub], self.bounds[sub+1])

            # Plot
            self.plot(component=sub_component, sub=None, order=order)

    def show(self):
        'Show all the plots. Should be called only once per lifecycle.'
        plt_show()

    def num_dofmaps(self):
        'Return number of dofmap extracted from the function space.'
        return len(self.dofmaps)

    def num_subspaces(self):
        'Return number of subspaces in the function space.'
        return len(self.bounds)-1
