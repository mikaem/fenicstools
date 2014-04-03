'''
Run a convergence test of CR computed divergence in 2d. Perform the test on
variety of meshes to see if the results are independent of the mesh quality.
'''
from dolfin import *
from fenicstools import cr_divergence
import numpy

u_exact = Expression(('sin(pi*x[0])', 'cos(2*pi*x[1])'))
divu_exact = Expression('pi*(cos(pi*x[0]) - 2*sin(2*pi*x[1]))')


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

rectangle0 = Rectangle(-1, -1, 1, 1)
rectangle1 = Rectangle(-0.25, -0.25, 0.25, 0.25)
circle = Circle(0, 0, 0.25)

domain0 = rectangle0
domain1 = rectangle0 - rectangle1
domain2 = rectangle0 - circle

domains = [domain0, domain1, domain2]
names = ['Rectangle', 'RectangleHole', 'CircleHole']
Ns = [10, 20, 40, 60, 80, 160, 240]
mesh_types = {name: [Mesh(domain, N) for N in Ns]
              for name, domain in zip(names, domains)}
mesh_types['UnitSquareMesh'] = [UnitSquareMesh(N, N) for N in Ns]

for mesh_type in mesh_types:
    print mesh_type
    meshes = mesh_types[mesh_type]
    print '     h    |   rate_Loo  |  rate_L2 | MeshQuality'
    mesh = meshes[0]
    h_, eLoo_, eL2_ = foo(mesh)
    for mesh in meshes[1:]:
        h, eLoo, eL2 = foo(mesh)
        rate_Loo = numpy.log(eLoo/eLoo_)/numpy.log(h/h_)
        rate_L2 = numpy.log(eL2/eL2_)/numpy.log(h/h_)
        h_, eLoo_, eL2_ = h, eLoo, eL2

        r_min, r_max = MeshQuality.radius_ratio_min_max(mesh)
        print '  %.4f  |     %.2f    |   %.2f   |    %.2f, %.2f' %\
        (h_, rate_Loo, rate_L2, r_min, r_max)
    print
