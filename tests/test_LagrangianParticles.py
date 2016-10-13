#!/usr/bin/env py.test
from fenicstools.LagrangianParticles import LagrangianParticles
from dolfin import VectorFunctionSpace, interpolate, UnitSquareMesh, Expression, info
import numpy as np


def test_LP():
    dt = 0.01

    # Initial
    mesh = UnitSquareMesh(20, 20)
    x = np.linspace(0.25, 0.75, 1000)
    y = 0.5*np.ones_like(x)
    x, y = np.r_[x, y], np.r_[y, x]
    particle_positions = np.c_[x, y]

    # At one dt one rigid body rotation around dt
    particle_positions_dt = particle_positions + dt*np.c_[-(y-0.5), (x-0.5)]

    V = VectorFunctionSpace(mesh, 'CG', 1)
    lp = LagrangianParticles(V)
    lp.add_particles(particle_positions, properties_d={'dt position': particle_positions_dt})

    # Time travel
    u = interpolate(Expression(('-(x[1]-0.5)', '(x[0]-0.5)'), degree=1), V)
    lp.step(u, dt=dt)

    e = [np.linalg.norm(p.position-p.properties['dt position']) < 1E-15 for p in lp]
    info('Has %d particles' % len(e))
    assert all(e)

# ---------------------------------------------------------------------------

if __name__ == '__main__': 
    import sys
    sys.exit(test_LP())
