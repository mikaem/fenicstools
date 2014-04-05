from math import log as ln

from dolfin import *
from fenicstools import weighted_gradient_matrix


set_log_level(WARNING)

# Exact u and its divergence
u_ = Expression(('sin(pi*x[0])', 'cos(2*pi*x[1])'))
divu_ = Expression('pi*(cos(pi*x[0]) - 2*sin(2*pi*x[1]))')


def divergence_test(mesh):
    V = FunctionSpace(mesh, 'CG', 1)
    W = VectorFunctionSpace(mesh, 'CG', 1)

    u = interpolate(u_, W)
    divu_p = project(div(u), V)                # projected divergence
    e_p = errornorm(divu_, divu_p, 'l2')

    divu_w = Function(V)
    d = mesh.geometry().dim()
    u_i = Function(V)
    for i in range(d):
        u_i.vector()[:] = u.vector()[W.sub(i).dofmap().dofs()]
        GRADi = weighted_gradient_matrix(mesh, i, 'CG', 1)
        # weighted interpolated divergence
        divu_w.vector()[:] += GRADi * u_i.vector()
    e_w = errornorm(divu_, divu_w, 'l2')

    return mesh.hmin(), e_p, e_w

# ------------------------------------------------------------------------------

if __name__ == '__main__':

    Ns = [2 ** i for i in range(3, 9)]
    meshes = [UnitSquareMesh(N, N) for N in Ns]

    h_, ep_, ew_ = divergence_test(meshes[0])
    for mesh in meshes[1:]:
        h, ep, ew = divergence_test(mesh)
        ratep = ln(ep / ep_) / ln(h / h_)
        ratew = ln(ew / ew_) / ln(h / h_)
        print h, ratep, ratew

        map
