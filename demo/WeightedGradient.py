from dolfin import *
from fenicstools.WeightedGradient import *

mesh = UnitSquareMesh(10, 10)
V = FunctionSpace(mesh, 'CG', 1)
u = interpolate(Expression("x[0]+2*x[1]"), V)
wx = weighted_gradient_matrix(mesh, 0)
dux = Function(V)
dux.vector()[:] = wx * u.vector()
wy = weighted_gradient_matrix(mesh, 1)
duy = Function(V)
duy.vector()[:] = wy * u.vector()
plot(dux, title='One')
plot(duy, title='Two')
