from fenicstools import DofMapPlotter
from dolfin import *

# Define domain and create mesh
#domain2d = Rectangle(Point(-1, -1), Point(1, 1))
#domain3d = Box(-1, -1, -1, 1, 1, 1)
mesh = RectangleMesh(Point(0, 0), Point(1, 1), 10, 10)
#mesh = Mesh(domain2d, 3)

# Create function space
V = VectorElement('CR', mesh.ufl_cell(), 1)
Q = FiniteElement('DG', mesh.ufl_cell(), 0)
S = FiniteElement('DG', mesh.ufl_cell(), 1)
M = FunctionSpace(mesh, MixedElement([V, Q, S]))

# Create DofMapPlotter for the space
dmp = DofMapPlotter(M)

# See how many dofmaps can be plotted
n_dofmaps = dmp.num_dofmaps()
# M is represented by signature [gdim, 1, 1] so there are gdim + 2 dofmaps

# Create plot which will show all dofmaps. Use global ordering scheme for dofs
# and mesh entities. plot(order='local') to switch to local ordering scheme
dmp.plot()
#dmp.show()   # Comment out to
#exit()       # showcase other capabilities

# Create plot which will show only dofs of single dofmap
for i in range(n_dofmaps):
    dmp.plot(component=i)

# Plot dofmaps of first component of V and Q, S
dmp.plot(component=[0, n_dofmaps-2, n_dofmaps-1])

# Access dofmaps by subspaces
n_subspaces = dmp.num_subspaces()
for sub in range(n_subspaces):
    dmp.plot(sub=sub)

# Give plot xkcd flavor
dmp_xkcd = DofMapPlotter(M, options={'xkcd': True})
dmp_xkcd.plot()

# Show the plots
dmp.show()
