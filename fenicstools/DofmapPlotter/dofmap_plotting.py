from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import time
from dolfin import Cell, Point, MeshEntity, edges, MPI, mpi_comm_world


class ParallelColorPrinter(object):

    'Print color messages proceeded by the info about process number.'

    def __init__(self, mpi_rank, mpi_size):
        self.mpi_rank = mpi_rank
        self.mpi_size = mpi_size

    def __call__(self, string, color=''):
        'Print string in given color.'
        if self.mpi_size > 1:
            mpi_string = 'Process number %d : ' % self.mpi_rank
        else:
            mpi_string = ''

        if color == 'blue':
            print '\033[1;37;34m%s%s\033[0m' % (mpi_string, string)
        elif color == 'green':
            print '\033[1;37;32m%s%s\033[0m' % (mpi_string, string)
        elif color == 'red':
            print '\033[1;37;31m%s%s\033[0m' % (mpi_string, string)
        elif color == 'yellow':
            print '\033[1;33;33m%s%s\033[0m' % (mpi_string, string)
        elif color == 'cyan':
            print '\033[0;36;36m%s%s\033[0m' % (mpi_string, string)
        elif color == 'pink':
            print '\033[1;31;31m%s%s\033[0m' % (mpi_string, string)
        else:
            print mpi_string + string

# -----------------------------------------------------------------------------


class DofmapPlotter(object):

    '''
    Plot map of degrees of freedom of finite element function space
    defined over mesh. To avoid cluttering of the screen the default plot
    shows only the mesh. Pressing the keys while mouse is in the figure window
    yields actions, see _help.
    '''

    def __init__(self, options=None):
        # Set dof plotting parameters
        self.dofs = {}
        self.dof_labels = {}
        self.scatter_objects = {}
        self.text_objects = {}
        self.showing_all_dofs = False

        # Set entity plotting parameters
        # Label for entities of topl. dimension 0, 1, 2, 3, and their flags
        self.entity_labels = {i: {} for i in range(4)}
        self.showing_all_entities = {i: False for i in range(4)}

        # Default options for colors of labels lof dofs and mesh entities with
        # tdim. Xkcd makes the plot look like xkcd comics. Requires
        # 'Humor Sans', 'Comic Sans MS' adn 'StayPuft' fonts to be installed on
        # the system. If the fonts are newly installed but matplotlib still
        # complains, try removing path/.cache/matplotlib/fontList.cache so that
        # it has to regenerate the list and pick up new fonts.
        # Ordering describes whether labels will show local or global ordering
        self.options = {'colors': {-1: 'b', 0: 'k', 1: 'r', 2: 'g', 3: 'c'},
                        'xkcd': True,
                        'ordering': 'local'}

        # Rewrite plotting options if they are provided by user
        if options is not None:
            self.options.update(options)

        # Get MPI info and customize the printer
        self.mpi_rank = MPI.rank(mpi_comm_world())
        self.mpi_size = MPI.size(mpi_comm_world())
        self.printer = ParallelColorPrinter(self.mpi_rank, self.mpi_size)

    def __call__(self, V):
        'Plot degrees of freedom of function space over mesh.'
        try:
            # Get info about the function space
            self.V = V
            self.mesh = V.mesh()
            self.dofmap = V.dofmap()
            self.num_cells = self.mesh.num_cells()
            self.gdim = self.mesh.geometry().dim()
            self.tdim = self.mesh.topology().dim()

            # Get first dof that belogs to the process

            self.first_dof = self.dofmap.ownership_range()[0]

        except AttributeError:
            self.printer('V is not a function space!', 'red')
            exit()

        # Do the plotting.
        self._do_plot()

    def _help(self):
        'Print help about key functionality.'
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
ctrl+t : clear facet indices

d : degrees of freedom of cell under cursor*
D : degrees of freedom of all cells in the mesh
ctrl+D : clear all showing degrees of freedom

h : displays the help menu

f : full screen (matplotlib functionality)
s : save the figure (matplotib functionality)
-------------------------------------------------------------------------------
* especially in 3D cursor position might not coincide with any cell. This
triggers warning. You might try to zoom in to help you hit something.
    '''
        self.printer(message)

    def _do_plot(self):
        'Plot the mesh and lauch the event handler.'
        gdim = self.gdim
        if gdim < 2:
            raise ValueError(
                'Meshes with geometrical dimension 1 not supported yet.')
        if V.num_sub_spaces() > 0:
            raise ValueError('No support for Mixed/Vector spaces yet.')

        # Notify about what ordering is used initially
        self.printer('Using %s ordering' % self.options['ordering'], 'green')

        # Turn on xkcd
        xkcd = False
        if self.options['xkcd']:
            plt.xkcd()
            xkcd = True
        self.fig = plt.figure()

        # Prapere axis labels
        if gdim > 2:
            ax = self.fig.gca(projection='3d')
            ax.set_xlabel('x' if xkcd else r'$x$')
            ax.set_ylabel('y' if xkcd else r'$y$')
            ax.set_zlabel('z' if xkcd else r'$z$')
        else:
            ax = self.fig.gca()
            ax.set_xlabel('x' if xkcd else r'$x$')
            if gdim > 1:
                ax.set_ylabel('y' if xkcd else r'$y$')

        # Make the axes available for all the methods
        self.axes = ax

        # First plot the mesh by drawing the edges
        mesh = self.mesh
        n_vertices = mesh.num_vertices()
        coordinates = mesh.coordinates().reshape((n_vertices, gdim))

        color = self.options['colors'][0]
        for edge in edges(mesh):
            vertex_indices = edge.entities(0)
            edge_coordinates = [[] for i in range(gdim)]
            for vertex_index in vertex_indices:
                for i in range(gdim):
                    edge_coordinates[i].append(coordinates[vertex_index, i])
            ax.plot(*edge_coordinates, color=color)

        # Make coordinates available for all methods
        self.coordinates = coordinates

        # Fix the limits for figure
        for i in range(gdim):
            xi_min, xi_max = coordinates[:, i].min(), coordinates[:, i].max()
            eval('ax.set_%slim(%g, %g)' % (chr(ord('x') + i), xi_min, xi_max))

        # Conect to event handler
        self.fig.canvas.mpl_connect('key_press_event', self._event_handler)
        plt.show()

    def _event_handler(self, event):
        'Call plotting method based on the pressed key.'
        if event.key in ['d', 'D', 'ctrl+d']:    # Plot dofs
            self._dof_plot(event)
        elif event.key in ['v', 'V', 'ctrl+v']:  # Plot vertices
            self._entity_plot(event, 'v', 'V', 0)
        elif event.key in ['c', 'C', 'ctrl+c']:  # Plot cells
            self._entity_plot(event, 'c', 'C', self.tdim)
        elif event.key in ['e', 'E', 'ctrl+e']:  # Plot edges
            self._entity_plot(event, 'e', 'E', 1)
        elif event.key in ['t', 'T', 'ctrl+t']:  # Plot facets
            self._entity_plot(event, 't', 'T', self.tdim - 1)

        # elif event.key == 'g': # Change the ordering #TODO How to sync the
        # options?
        # if self.options['ordering'] == 'local':    #how to sync clear screen
        #    self.options['ordering'] = 'global'
        #
        #  elif self.options['ordering'] == 'global':
        #    self.options['ordering'] = 'local'
        #
        # self.printer('Swithing to %s ordering' %\
        # self.options['ordering'], 'red')

        elif event.key == 'h':
            self._help()
        elif event.key in ['s', 'f', 'l']:
            pass  # Ignore the matplotlib active keys
        else:
            self.printer('Unhandled key pressed event ' + event.key)

    def _locate_event(self, event):
        'Find cells where the key press was initiated.'
        # Locate all the cells that intersect event point. If locating is done
        # by compute_first_collision the cells are difficult to mark.
        bb_tree = self.mesh.bounding_box_tree()
        # Get coordinates of the event point
        x_string = self.axes.format_coord(event.xdata, event.ydata)
        # Convert to numbers, hard time with regexp to capture minus sign
        try:
            x = map(float, [w.split('=')[1] for w in x_string.split(' ') if w])
            cell_indices = bb_tree.compute_entity_collisions(Point(*x))
        except ValueError:
            if self.mpi_size > 1:
                self.printer('Are you clicking outside the active figure?')
            cell_indices = []
        return cell_indices

    def _entity_plot(self, event, local_key, global_key, tdim):
        '''Universal mesh entity plotting function. Pressing local_key,
        single_entity_plot(...,tdim) function is used on cell under cursor.
        With global_key all the cells are target of local plotting.'''
        if event.key == local_key:
            if not self.showing_all_entities[tdim]:
                cell_indices = self._locate_event(event)
                for cell_index in cell_indices:
                    self._single_entity_plot(cell_index, tdim)
                self.fig.canvas.draw()

        elif event.key == global_key:
            if not self.showing_all_entities[tdim]:
                start = time.time()
                self.printer('Processing ...')
                cell_indices = xrange(self.num_cells)
                for cell_index in cell_indices:
                    self._single_entity_plot(cell_index, tdim)
                self.fig.canvas.draw()
                self.showing_all_entities[tdim] = True
                stop = time.time()
                self.printer('done in %.2f seconds.' % (stop - start))

        else:
            # Remove all the existing labels
            if len(self.entity_labels[tdim]):
                for label in self.entity_labels[tdim].itervalues():
                    label.set_visible(False)

                # Flag that work will have to be done again
                self.entity_labels[tdim] = {}
                self.showing_all_entities[tdim] = False
                self.fig.canvas.draw()

    def _single_entity_plot(self, cell_index, tdim):
        '''Loop through entities of tdim topological dimension that are in
        cell with index and plot their labels which are given by tdim-entity
        numbering scheme.'''
        self.mesh.init(self.tdim, tdim)
        # Dont compute cell-cell connectivity
        if self.tdim == tdim:
            entity_indices = [cell_index]
        else:
            entity_indices = Cell(self.mesh, cell_index).entities(tdim)

        color = self.options['colors'][tdim]
        labels = self.entity_labels[tdim]
        for entity_index in entity_indices:
            entity = MeshEntity(self.mesh, tdim, entity_index)
            x = entity.midpoint()
            x = [x[i] for i in range(self.gdim)]
            if self.options['ordering'] == 'global':
                if self.mpi_size:
                    args = x + [str(entity.global_index())]
                else:
                    args = x + [str(entity_index)]
            elif self.options['ordering'] == 'local':
                args = x + [str(entity_index)]
            if not(entity_index in labels):
                labels[entity_index] = self.axes.text(*args, color=color)

    def _dof_plot(self, event):
        '''For cell under cursor or all the cells, plot the locations of degrees
        of freedom and their labels. The label is order of dof in dof numbering
        scheme.'''
        if event.key == 'd':
            if not self.showing_all_dofs:
                # Locate cells with event and plot their dofs and labels
                cell_indices = self._locate_event(event)
                new_dofs = []
                new_labels = []
                dupl_labels = []
                for cell_index in cell_indices:
                    new_dofs_labels = self._single_dof_plot(cell_index)
                    new_dofs.extend(new_dofs_labels[0])
                    new_labels.extend(new_dofs_labels[1])
                    dupl_labels.extend(new_dofs_labels[2])
                # Scatter and text the new objects
                print 'what', new_dofs
                self._create_objects(new_dofs, new_labels, dupl_labels)
                self.fig.canvas.draw()

        elif event.key == 'D':
            if not self.showing_all_dofs:
                start = time.time()
                self.printer('Processing ...')
                cell_indices = xrange(self.num_cells)
                new_dofs = []
                new_labels = []
                dupl_labels = []
                for cell_index in cell_indices:
                    new_dofs_labels = self._single_dof_plot(cell_index)
                    new_dofs.extend(new_dofs_labels[0])
                    new_labels.extend(new_dofs_labels[1])
                    dupl_labels.extend(new_dofs_labels[2])
                # Scatter and text the new objects
                self.showing_all_dofs = True  # Flag that all plotting is done
                self._create_objects(new_dofs, new_labels, dupl_labels)
                self.fig.canvas.draw()
                stop = time.time()
                self.printer('done in %.2f seconds.' % (stop - start))
        else:
            # Remove all the existing dof labels and dofs
            for scatter_object in self.scatter_objects.itervalues():
                scatter_object.remove()

            for text_object in self.text_objects.itervalues():
                text_object.set_visible(False)

            # Flag that work will have to be done again
            self.dofs = {}
            self.dof_labels = {}
            self.scatter_objects = {}
            self.text_objects = {}
            self.showing_all_dofs = False
            self.fig.canvas.draw()

    def _single_dof_plot(self, cell_index):
        'Plot dof locations and labels for cell with given index.'
        dofmap = self.dofmap
        mesh = self.mesh
        cell_dofs = dofmap.cell_dofs(cell_index)
        dof_coordinates = dofmap.tabulate_coordinates(Cell(mesh, cell_index))
        ordering = self.options['ordering']

        new_dofs = []
        new_labels = []
        dupl_labels = []
        for i in range(len(cell_dofs)):
            x = dof_coordinates[i, :].tolist()
            key = ''.join(map(str, x))
            if not(key in self.dofs):
                self.dofs[key] = x  # Register position and index of new dof
                new_dofs.append(key)

            if ordering == 'global':
                label = str(cell_dofs[i])
            elif ordering == 'local':
                label = str(cell_dofs[i] - self.first_dof)

            if not(key in self.dof_labels):
                self.dof_labels[key] = label
                new_labels.append(key)
            else:
                old_label = self.dof_labels[key]
                print 'old', old_label, 'current', label, 's', old_label.find(label),
                if old_label.find(label) == -1:
                    label = ' '.join([old_label, label])
                    self.dof_labels[key] = label
                    dupl_labels.append(key)

        return (new_dofs, new_labels, dupl_labels)

    def _create_objects(self, new_dofs, new_labels, dupl_labels):
        'Put last new objects on the figure.'
        color = self.options['colors'][-1]
        s_objects = self.scatter_objects
        t_objects = self.text_objects
        axes = self.axes

        print 'new dofs', new_dofs
        print len(s_objects), len(self.scatter_objects)
        for dof_key in new_dofs:
            dof = self.dofs[dof_key]  # Dof is a position
            print 'plotting?', dof
            s_objects[dof_key] = self.axes.scatter(*dof, color=color, marker='o', s=40)
        print len(s_objects), len(self.scatter_objects)

        for dof_label_key in new_labels:
            # Combine position and label to make text
            dof = self.dofs[dof_label_key] # Dof is a position
            label = dof + [self.dof_labels[dof_label_key]]
            #t_objects[dof_label_key].set_visible(False)
            t_objects[dof_label_key] = axes.text(*label, color=color)

        for dof_label_key in dupl_labels:
            # Combine position and label to make text
            dof = self.dofs[dof_label_key] # Dof is a position
            label = dof + [self.dof_labels[dof_label_key]]
            t_objects[dof_label_key].set_visible(False)
            t_objects[dof_label_key] = axes.text(*label, color=color)

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    from dolfin import Rectangle, Circle, FunctionSpace, Mesh

    domain = Rectangle(-1, -1, 1, 1) - Circle(0, 0, 0.4)
    mesh = Mesh(domain, 5)

    V = FunctionSpace(mesh, 'DG', 1)
    dp = DofmapPlotter()
    dp(V)
