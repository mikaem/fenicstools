from dolfin import *
from fenicstools import divergence_matrix
from numpy import cos, pi

mesh = UnitSquareMesh(10, 10)
x = mesh.coordinates()
x[:] = (x - 0.5) * 2
x[:] = 0.5*(cos(pi*(x-1.) / 2.) + 1.)

CG = VectorFunctionSpace(mesh, 'CG', 1)
u = interpolate(Expression(("sin(2*pi*x[0])", "cos(3*pi*x[1])")), CG)

C = divergence_matrix(mesh)
DG = FunctionSpace(mesh, 'DG', 0)
cc = Function(DG)
cc.vector()[:] = C * u.vector()
plot(cc, title='Gauss div')
plot(div(u), title="Projection")

divu_dg = project(div(u), DG)

plot(divu_dg - cc, title="Difference between FV and FEM")

