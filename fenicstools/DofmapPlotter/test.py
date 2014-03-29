from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from dolfin import Cell, Point, MeshEntity, edges, MPI, mpi_comm_world
from numpy import zeros

class DofmapPlotter(object):
  '''
  Plot map of degrees of freedom of finite element function space
  defined over mesh. To avoid cluttering of the screen the default plot
  shows only the mesh. Pressing the keys while mouse is in the figure window
  yields actions, see _help.
  '''
  def __init__(self, options=None):
    # Set dof plotting parameters
    self.dofs = []
    self.dof_labels = []
    self.showing_all_dofs = False

    # Set entity plotting parameters
    # Label for entities of topl. dimension 0, 1, 2, 3, and their flags
    self.entity_labels = {i : [] for i in range(4)}
    self.showing_all_entities = {i : False for i in range(4)}

    # Default options for colors of label of dofs and mesh entities with tdim
    self.options = {-1 : 'b', 0 : 'k', 1 : 'r', 2 : 'g', 3 : 'c'}
    
    # Rewrite plotting options if they are provided by used
    if options is not None:
      self.options.update(options)
  
  def __call__(self, V):
    'Plot degrees of freedom of function space over mesh.'
    try:
      # Get info about the function space
      self.mesh = V.mesh()
      self.dofmap = V.dofmap()
      self.gdim = self.mesh.geometry().dim()
      self.tdim = self.mesh.topology().dim()

    except AttributeError:
      print 'V is not a function space!'
      exit()
    
    # Do the plotting.
    self._do_plot()

  def _help(self):
    'Print help about key functionality.'
    message =\
    '''
                              KEY FUNCTIONALITY
    --------------------------------------------------------------------------
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
  -----------------------------------------------------------------------------
    * especially in 3D cursor position might not coincide with any cell. This
    triggers warning. You might try to zoom in to help you hit something.
    '''
    print message

  def _do_plot(self):
    'Plot the mesh and lauch the event handler.'
    gdim = self.gdim
    if gdim < 2:
      raise ValueError('Meshes with geometrical dimension 1 not supported yet.')
    if MPI.size(mpi_comm_world()) > 1:
      raise ValueError('Does not work in parallel yet.')
    fig = plt.figure()

    if gdim > 2:
      ax = fig.gca(projection='3d')
      ax.set_xlabel(r'$x$')
      ax.set_ylabel(r'$y$')
      ax.set_zlabel(r'$z$')
    else:
      ax = fig.gca()
      ax.set_xlabel(r'$x$')
      if gdim > 1:
        ax.set_ylabel(r'$y$')

    # Make the axes available for all the methods
    self.axes = ax

    # First plot the mesh by drawing the edges
    mesh = self.mesh
    n_vertices = mesh.num_vertices()
    coordinates = mesh.coordinates().reshape((n_vertices, gdim))
    
    color = self.options[0]
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
    fig.canvas.mpl_connect('key_press_event', self._event_handler) 
    plt.show() 

  def _event_handler(self, event):
    'Call plotting method based on the pressed key.'
    if event.key in ['d', 'D', 'ctrl+d']:    # plot dofs
      self._dof_plot(event)
    elif event.key in ['v', 'V', 'ctrl+v']:  # plot vertices
      self._entity_plot(event, 'v', 'V', 0)
    elif event.key in ['c', 'C', 'ctrl+c']:  # plot cells
      self._entity_plot(event, 'c', 'C', self.tdim)
    elif event.key in ['e', 'E', 'ctrl+e']:  # plot edges
      self._entity_plot(event, 'e', 'E', 1)
    elif event.key in ['t', 'T', 'ctrl+t']:  # plot facets
      self._entity_plot(event, 't', 'T', self.tdim-1)
    elif event.key in ['s', 'f']:
      pass # Ignore the matplotlib active keys
    elif event.key == 'h':
      self._help()
    else:
      print 'Unhandled key pressed event', event.key

  def _locate_event(self, event):
    'Find cells where the key press was initiated.'
    # Locate all the cells that intersect event point. If locating is done by
    # compute_first_collision the cells are difficult to mark.
    bb_tree = self.mesh.bounding_box_tree()
    # Get coordinates of the event point
    x_string = self.axes.format_coord(event.xdata, event.ydata)
    # Convert to numbers
    # FIXME replace by regexp?
    print x_string
    x = map(float, [word.split('=')[1] for word in x_string.split(' ') if word])
    cell_indices = bb_tree.compute_entity_collisions(Point(*x))
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
        plt.draw()
    elif event.key == global_key:
      if not self.showing_all_entities[tdim]:
        cell_indices = range(self.mesh.num_cells())
        for cell_index in cell_indices:
          self._single_entity_plot(cell_index, tdim)
        plt.draw()
        self.showing_all_entities[tdim] = True
    else:
      # Remove all the existing labels
      if len(self.entity_labels[tdim]):
        for label in self.entity_labels[tdim]:
          label.set_visible(False)

        # Flag that work will have to be done again
        self.entity_labels[tdim] = [] 
        self.showing_all_entities[tdim] = False
        plt.draw()

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

    color = self.options[tdim]
    for entity_index in entity_indices:
      entity = MeshEntity(self.mesh, tdim, entity_index)
      x = entity.midpoint()
      x = [x[i] for i in range(self.gdim)]
      args = x + [str(entity.global_index())]
      self.entity_labels[tdim].append(self.axes.text(*args, color=color))

  def _dof_plot(self, event):
    '''For cell under cursor or all the cells, plot the locations of degrees
    of freedom and their labels. The label is order of dof in dof numbering
    scheme.'''
    # FIXME the code is duplicit but D is done that way for speed. Consider
    # some unification
    if event.key == 'd':
      if not self.showing_all_dofs:
        # Locate cells with event and plot their dofs and labels
        dofmap = self.dofmap
        mesh = self.mesh
        cell_indices = self._locate_event(event)
        
        if len(cell_indices): # Found something
          for cell_index in cell_indices:
            cell_dofs = dofmap.cell_dofs(cell_index)
            dof_coordinates = dofmap.tabulate_coordinates(Cell(mesh, cell_index))
            
            for i  in range(len(cell_dofs)):
              x = dof_coordinates[i, :].tolist()
              self.dofs.append(self.axes.scatter(*x, color='b', marker='o', s=8))
              args = x + [str(cell_dofs[i])]
              self.dof_labels.append(self.axes.text(*args, color='b'))
          plt.draw()

    elif (event.key == 'D'):
      if not self.showing_all_dofs:
        # Run over all cells plotting their dofs + labels
        dofmap = self.dofmap
        mesh = self.mesh
        gdim = self.gdim
        ax = self.axes

        # Plot dof locations
        n_dofs = dofmap.global_dimension()
        dof_coordinates = dofmap.tabulate_all_coordinates(mesh)
        dof_coordinates = dof_coordinates.reshape((n_dofs, gdim))
        x = [dof_coordinates[:, i] for i in range(gdim)]
        self.dofs.append(ax.scatter(*x, color='b', marker='o', s=8))

        # Plot the labels
        for dof in range(n_dofs):
          args = dof_coordinates[dof, :].tolist() + [str(dof)]
          self.dof_labels.append(ax.text(*args, color='b'))

        self.showing_all_dofs = True # Flag that all plotting is done
        plt.draw()
    else:
      # Remove all the existing dof labels and dofs
      if len(self.dof_labels) and len(self.dofs):
        for dof in self.dofs:
          dof.remove()

        for dof_label in self.dof_labels:
          dof_label.set_visible(False)

        # Flag that work will have to be done again
        self.dofs = [] 
        self.dof_labels = []
        self.showing_all_dofs = False
        plt.draw()
      
# -----------------------------------------------------------------------------

if __name__ == '__main__':
  from dolfin import UnitIntervalMesh, Rectangle, Circle, FunctionSpace, Mesh
  #mesh = UnitIntervalMesh(10)

  domain = Rectangle(-1, -1, 1, 1) - Circle(0, 0, 0.4)
  mesh = Mesh(domain, 10)

  V = FunctionSpace(mesh, 'CG', 1)
  dp = DofmapPlotter()
  dp(V)


