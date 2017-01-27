from fenicstools.LagrangianParticles import LagrangianParticles, RandomCircle
import matplotlib.pyplot as plt
#from dolfin import VectorFunctionSpace, interpolate, RectangleMesh, Expression, Point
from dolfin import *
from mpi4py import MPI as pyMPI

comm = pyMPI.COMM_WORLD

mesh = RectangleMesh(Point(0, 0), Point(1, 1), 10, 10)
particle_positions = RandomCircle([0.5, 0.75], 0.15).generate([100, 100])

V = VectorFunctionSpace(mesh, 'CG', 1)
u = interpolate(Expression(("-2*sin(pi*x[1])*cos(pi*x[1])*pow(sin(pi*x[0]),2)",
                            "2*sin(pi*x[0])*cos(pi*x[0])*pow(sin(pi*x[1]),2)"),
                            degree=3),
                V)
lp = LagrangianParticles(V)
lp.add_particles(particle_positions)

fig = plt.figure()
lp.scatter(fig)
fig.suptitle('Initial')

if comm.Get_rank() == 0:
    fig.show()

plt.ion()

save = False

dt = 0.01
for step in range(500):
    lp.step(u, dt=dt)

    lp.scatter(fig)
    fig.suptitle('At step %d' % step)
    fig.canvas.draw()

    if save: plt.savefig('img%s.png' % str(step).zfill(4))

    fig.clf()
