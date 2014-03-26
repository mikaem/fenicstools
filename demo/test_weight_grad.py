from dolfin import *
from fenicstools import weighted_gradient_matrix
from math import log as ln

set_log_level(WARNING)

f = Expression('5*pi*pi*sin(2*pi*x[0])*sin(pi*x[1])')

u_ = Expression('sin(2*pi*x[0])*sin(pi*x[1])')
gradu_ = Expression(('cos(2*pi*x[0])*2*pi*sin(pi*x[1])',\
                     'sin(2*pi*x[0])*cos(pi*x[1])*pi'))

def foo(mesh, family, degree):
    V = FunctionSpace(mesh, family, 1)
    u = TrialFunction(V)
    v = TestFunction(V)

    a = inner(grad(u), grad(v))*dx
    L = inner(f, v)*dx
    bc = DirichletBC(V, Constant(0.), DomainBoundary())

    u = Function(V)
    solve(a == L, u, bc)

    VV = VectorFunctionSpace(mesh, family, degree=1 if family == 'CR' else degree)
    gradu = Function(VV)
    for i in range(VV.num_sub_spaces()):
        GRAD = weighted_gradient_matrix(mesh, i, family, degree)
        gradu.vector()[VV.sub(i).dofmap().dofs()] = GRAD*u.vector()

    h = mesh.hmin()
    e_l2 = errornorm(u_, u, 'l2')
    grade_l2 = errornorm(u_, u, 'h10')
    grade_w_l2 = errornorm(gradu_, gradu, 'l2')

    return h, e_l2, grade_l2, grade_w_l2

#----------------------------------------------------------------------

N = 1
mesh = UnitSquareMesh(8, 8)
h_, eL2_, eH1_, ew_ = foo(mesh, family='CR', degree=2)

for N in [16, 32, 64, 96, 128]:
    mesh = UnitSquareMesh(N, N)
    h, eL2, eH1, ew = foo(mesh, family='CR', degree=1)

    rateL2 = ln(eL2/eL2_)/ln(h/h_)

    rateH1 = ln(eH1/eH1_)/ln(h/h_)

    rateW = ln(ew/ew_)/ln(h/h_)

    print rateL2, rateH1, rateW
#  
#
