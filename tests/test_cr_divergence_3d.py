'''
Run a convergence test of CR computed divergence in 3d.
'''
from dolfin import *
from fenicstools import cr_divergence
import numpy

u_exact = Expression(('sin(pi*x[0])', 'cos(2*pi*x[1])', 'exp(-x[2]*x[2])'))
divu_exact = \
Expression('pi*(cos(pi*x[0]) - 2*sin(2*pi*x[1])) - 2*exp(-x[2]*x[2])*x[2]')


def foo(mesh):
    # Compute the divergence with CR method
    cr_divu = cr_divergence(u_exact, mesh=mesh)

    # Represent the exact divergence on DG
    DG = FunctionSpace(mesh, 'DG', 0)
    divu = interpolate(divu_exact, DG)

    divu.vector().axpy(-1, cr_divu.vector())
    error_Loo = divu.vector().norm('linf')/DG.dim()
    error_L2 = divu.vector().norm('l2')/DG.dim()

    return mesh.hmin(), error_Loo, error_L2

# -----------------------------------------------------------------------------

mesh = UnitCubeMesh(8, 8, 8)
print '     h    |   rate_Loo  |  rate_L2 '
h_, eLoo_, eL2_ = foo(mesh)
for mesh in [UnitCubeMesh(N, N, N) for N in [16, 32, 64]]:
    h, eLoo, eL2 = foo(mesh)
    rate_Loo = numpy.log(eLoo/eLoo_)/numpy.log(h/h_)
    rate_L2 = numpy.log(eL2/eL2_)/numpy.log(h/h_)
    h_, eLoo_, eL2_ = h, eLoo, eL2
    print '  %.4f  |     %.2f    |   %.2f   ' % (h_, rate_Loo, rate_L2)
