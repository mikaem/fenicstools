__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-12-13"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"
import os, inspect
from dolfin import info, compile_extension_module, Function, FunctionSpace, assemble, TrialFunction, TestFunction, dx, Matrix

fem_folder = os.path.abspath(os.path.join(inspect.getfile(inspect.currentframe()), "../fem"))
gradient_code = open(os.path.join(fem_folder, 'gradient_weight.cpp'), 'r').read()
compiled_gradient_module = compile_extension_module(code=gradient_code)

def weighted_gradient_matrix(mesh, i, family='CG', degree=1, constrained_domain=None):
    """Compute weighted gradient matrix

    The matrix allows you to compute the gradient of a P1 Function
    through a simple matrix vector product

    CG family:
        p_ is the pressure solution on CG1
        dPdX = weighted_gradient_matrix(mesh, 0, 'CG', degree)
        V = FunctionSpace(mesh, 'CG', degree)
        dpdx = Function(V)
        dpdx.vector()[:] = dPdX * p_.vector()

        The space for dpdx must be continuous Lagrange of some order

    CR family:
        p_ is the pressure solution on CR
        dPdX = weighted_gradient_matrix(mesh, 0, 'CR', 1)
        V = FunctionSpace(mesh, 'CR', 1)
        dpdx = Function(V)
        dpdx.vector()[:] = dPdX * p_.vector()

    """

    DG = FunctionSpace(mesh, 'DG', 0)

    if family == 'CG':
        # Source and Target spaces are CG_1 and CG_degree
        S = FunctionSpace(mesh, 'CG', 1, constrained_domain=constrained_domain)
        T = FunctionSpace(mesh, 'CG', degree, constrained_domain=constrained_domain)
    elif family == 'CR':
        if degree != 1:
            print '\033[1;37;34m%s\033[0m' % 'Ignoring degree'

        # Source and Target spaces are CR
        S = FunctionSpace(mesh, 'CR', 1, constrained_domain=constrained_domain)
        T = S
    else:
        raise ValueError('Only CG and CR families are allowed.')

    G = assemble(TrialFunction(DG)*TestFunction(T)*dx)
    dg = Function(DG)
    if isinstance(i, (tuple, list)):
        CC = []
        for ii in i:
            dP = assemble(TrialFunction(S).dx(ii)*TestFunction(DG)*dx)
            A = Matrix(G)
            Cp = compiled_gradient_module.compute_weighted_gradient_matrix(A, dP, dg)
            CC.append(Cp)
        return CC
    else:
        dP = assemble(TrialFunction(S).dx(i)*TestFunction(DG)*dx)        
        Cp = compiled_gradient_module.compute_weighted_gradient_matrix(G, dP, dg)
        #info(G, True)        
        #info(dP, True)        
        return Cp
