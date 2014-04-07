from dolfin import *
from fenicstools import divergence_matrix

mesh = UnitSquareMesh(100, 100)
CG = VectorFunctionSpace(mesh, 'CG', 1)
u = interpolate(Expression(("sin(2*pi*x[0])", "cos(3*pi*x[1])")), CG)

C = divergence_matrix(u)
DG = FunctionSpace(mesh, 'DG', 0)
cc = Function(DG)
cc.vector()[:] = C * u.vector()
plot(cc, title='Gauss div')
plot(div(u), title="Projection")
