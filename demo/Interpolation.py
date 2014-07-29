from dolfin import *
from fenicstools import interpolate_nonmatching_mesh

# Test for nonmatching mesh and FunctionSpace
mesh = UnitCubeMesh(16, 16, 16)
mesh2 = UnitCubeMesh(32, 32, 32)
V = FunctionSpace(mesh, 'CG', 1)
V2 = FunctionSpace(mesh2, 'CG', 1)

# Just create some random data to be used for probing
x0 = interpolate(Expression('x[0]'), V)
u = interpolate_nonmatching_mesh(x0, V2)

VV = VectorFunctionSpace(mesh, 'CG', 1)
VV2 = VectorFunctionSpace(mesh2, 'CG', 1)
v0 = interpolate(Expression(('x[0]', '2*x[1]', '3*x[2]')), VV)    
v = interpolate_nonmatching_mesh(v0, VV2)

plot(u)
plot(v)
interactive()

