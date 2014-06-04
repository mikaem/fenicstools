'''
Run a convergence test with divergence computed by Gauss theorem. Perform the
test on variety of meshes in 2d to see how the results depend on the mesh
quality.
'''

from dolfin import *
from fenicstools import gauss_divergence
import numpy


def demo_2d_divergence(mesh, u_exact, divu_exact):
    # Compute the divergence with Gauss theorem
    cr_divu = gauss_divergence(u_exact, mesh=mesh)

    # Represent the exact divergence on DG
    rank = divu_exact.value_rank()
    if rank == 0:
        DG = FunctionSpace(mesh, 'DG', 0)
    elif rank in [1, 2]:
        DG = VectorFunctionSpace(mesh, 'DG', 0)
    divu = interpolate(divu_exact, DG)

    # Compute the l^oo and l^1 norms
    divu.vector().axpy(-1, cr_divu.vector())
    error_Loo = divu.vector().norm('linf')/DG.dim()
    error_L2 = divu.vector().norm('l2')/DG.dim()
    h = mesh.hmin()

    # Create global measure of h and errors
    comm = mpi_comm_world()
    h = MPI.min(comm, mesh.hmin())
    error_Loo = MPI.max(comm, error_Loo)
    error_L2 = MPI.max(comm, error_L2)

    return h, error_Loo, error_L2

# -----------------------------------------------------------------------------

# Create various meshes
rectangle0 = Rectangle(-1, -1, 1, 1)
rectangle1 = Rectangle(-0.25, -0.25, 0.25, 0.25)
circle = Circle(0, 0, 0.25)

domain0 = rectangle0
domain1 = rectangle0 - rectangle1
domain2 = rectangle0 - circle

domains = [domain0, domain1, domain2]
names = ['Rectangle', 'RectangleHole', 'CircleHole']
Ns = [20, 40, 60, 80, 160, 240]
mesh_types = {name: [Mesh(domain, N) for N in Ns]
              for name, domain in zip(names, domains)}
mesh_types['UnitSquareMesh'] = [UnitSquareMesh(N, N) for N in Ns]

# Create a dicionary of u, divu for u scalar, vector, tensor
test_cases = {}

u_exact = Expression('sin(pi*x[0]*x[1])')
divu_exact = Expression(('cos(pi*x[0]*x[1])*pi*x[1]',
                         'cos(pi*x[0]*x[1])*pi*x[0]'))
test_cases['Scalar'] = [u_exact, divu_exact]

u_exact = Expression(('sin(pi*x[0])', 'cos(2*pi*x[1])'))
divu_exact = Expression('pi*(cos(pi*x[0]) - 2*sin(2*pi*x[1]))')
test_cases['Vector'] = [u_exact, divu_exact]

u_exact = Expression((('sin(pi*x[0])', 'cos(pi*x[1])'),
                      ('-cos(pi*x[0])', 'sin(2*pi*x[1])')))
divu_exact = Expression(('cos(pi*x[0])*pi - sin(pi*x[1])*pi',
                         'sin(pi*x[0])*pi + cos(2*pi*x[1])*2*pi'))
test_cases['Tensor'] = [u_exact, divu_exact]

# Go through the combinations
for quantity in test_cases:
    print 'Computing divergence of', quantity
    for mesh_type in mesh_types:
        print mesh_type
        meshes = mesh_types[mesh_type]
        print '     h    |   rate_Loo  |  rate_L2 | MeshQuality'
        mesh = meshes[0]
        h_, eLoo_, eL2_ = demo_2d_divergence(mesh, *test_cases[quantity])
        for mesh in meshes[1:]:
            h, eLoo, eL2 = demo_2d_divergence(mesh, *test_cases[quantity])
            rate_Loo = numpy.log(eLoo/eLoo_)/numpy.log(h/h_)
            rate_L2 = numpy.log(eL2/eL2_)/numpy.log(h/h_)
            h_, eLoo_, eL2_ = h, eLoo, eL2

            r_min, r_max = MeshQuality.radius_ratio_min_max(mesh)
            print '  %.4f  |     %.2f    |   %.2f   |    %.2f, %.2f' %\
                (h_, rate_Loo, rate_L2, r_min, r_max)
        print
