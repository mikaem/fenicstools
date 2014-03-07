__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-12-13"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"
import os, inspect
from dolfin import compile_extension_module, Function, FunctionSpace, assemble, TrialFunction, TestFunction, dx, Matrix

fem_folder = os.path.abspath(os.path.join(inspect.getfile(inspect.currentframe()), "../fem"))
gradient_code = open(os.path.join(fem_folder, 'gradient_weight.cpp'), 'r').read()
compiled_gradient_module = compile_extension_module(code=gradient_code)

def weighted_gradient_matrix(mesh, i, degree=1, constrained_domain=None):
    """Compute weighted gradient matrix
    
    The matrix allows you to compute the gradient of a P1 Function 
    through a simple matrix vector product
    
      p_ is the pressure solution on CG1
      dPdX = weighted_gradient_matrix(mesh, 0, degree)
      V = FunctionSpace(mesh, 'CG', degree)
      dpdx = Function(V)
      dpdx.vector()[:] = dPdX * p_.vector()
      
    The space for dpdx must be continuous Lagrange of some order
      
    """
    DG = FunctionSpace(mesh, 'DG', 0)
    CG = FunctionSpace(mesh, 'CG', degree, constrained_domain=constrained_domain)
    CG1 = FunctionSpace(mesh, 'CG', 1, constrained_domain=constrained_domain)
    C = assemble(TrialFunction(CG)*TestFunction(CG)*dx)
    G = assemble(TrialFunction(DG)*TestFunction(CG)*dx)
    dg = Function(DG)
    if isinstance(i, (tuple, list)):
        CC = []
        for ii in i:
            dP = assemble(TrialFunction(CG1).dx(ii)*TestFunction(DG)*dx)
            A = Matrix(G)
            Cp = compiled_gradient_module.compute_weighted_gradient_matrix(A, dP, dg)
            CC.append(Cp)
        return CC
    else:
        dP = assemble(TrialFunction(CG1).dx(i)*TestFunction(DG)*dx)
        Cp = compiled_gradient_module.compute_weighted_gradient_matrix(G, dP, dg)
        return Cp
