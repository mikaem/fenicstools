from dolfin import *
from fenicstools.Probe import *

# Test the probe functions:
set_log_level(20)

mesh = UnitCubeMesh(16, 16, 16)
#mesh = UnitSquareMesh(10, 10)
V = FunctionSpace(mesh, 'CG', 1)
Vv = VectorFunctionSpace(mesh, 'CG', 1)
W = V * Vv

# Just create some random data to be used for probing
w0 = interpolate(Expression(('x[0]', 'x[1]', 'x[2]', 'x[1]*x[2]')), W)

x = array([[1.5, 0.5, 0.5], [0.2, 0.3, 0.4], [0.8, 0.9, 1.0]])
p = Probes(x.flatten(), W)
x = x*0.9 
p.add_positions(x.flatten(), W)
for i in range(6):
    p(w0)

print p.array(2, "testarray")         # dump snapshot 2
print p.array(filename="testarray")   # dump all snapshots
print p.dump("testarray")

