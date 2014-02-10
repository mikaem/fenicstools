__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2010-11-03"
__copyright__ = "Copyright (C) 2010 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

import ufl
from dolfin import *

__all__ = ['StreamFunction', 'StreamFunction3D']

def StreamFunction(u, bcs, use_strong_bc = False):
    """Stream function for a given general 2D velocity field.
    The boundary conditions are weakly imposed through the term
    
        inner(q, grad(psi)*n)*ds, 
    
    where grad(psi) = [-v, u] is set on all boundaries. 
    This should work for any collection of boundaries: 
    walls, inlets, outlets etc.    
    """
    # Check dimension
    if isinstance(u, ufl.tensors.ListTensor):
        mesh = u[0].function_space().mesh()
    else:
        mesh = u.function_space().mesh()
    if not mesh.topology().dim() == 2:
        error("Stream-function can only be computed in 2D.")

    #V   = u.function_space().sub(0)
    degree = u[0].ufl_element().degree() if isinstance(u, ufl.tensors.ListTensor) else \
             u.ufl_element().degree()
    V   = FunctionSpace(mesh, 'CG', degree)
    q   = TestFunction(V)
    psi = TrialFunction(V)
    n   = FacetNormal(mesh)
    a   = dot(grad(q), grad(psi))*dx
    #L   = dot(q, u[1].dx(0) - u[0].dx(1))*dx
    L   = dot(q, curl(u))*dx 
    
    if(use_strong_bc): 
        # Strongly set psi = 0 on entire domain. Used for drivencavity.
        bcu = [DirichletBC(V, Constant(0), DomainBoundary())]
    else:
        bcu=[]        
        L = L + q*(n[1]*u[0] - n[0]*u[1])*ds
        
    # Compute solution
    psi = Function(V)
    A = assemble(a)
    b = assemble(L)
    if not use_strong_bc: 
        normalize(b)  # Because we only have Neumann conditions
    [bc.apply(A, b) for bc in bcu]
    solve(A, psi.vector(), b, 'gmres', 'hypre_amg')
    if not use_strong_bc: 
        normalize(psi.vector())

    return psi
    
def StreamFunction3D(u, constrained_domain=None):
    """Stream function for a given 3D velocity field.
    The boundary conditions are weakly imposed through the term
    
        inner(q, grad(psi)*n)*ds, 
    
    where u = curl(psi) is used to fill in for grad(psi) and set 
    boundary conditions. This should work for walls, inlets, outlets, etc.
    """
    if isinstance(u, ufl.tensors.ListTensor):
        mesh = u[0].function_space().mesh()
    else:
        mesh = u.function_space().mesh()

    degree = u[0].ufl_element().degree() if isinstance(u, ufl.tensors.ListTensor) else \
             u.ufl_element().degree()
    V = VectorFunctionSpace(mesh, 'CG', degree, constrained_domain=constrained_domain)

    # Check dimension
    if not mesh.topology().dim() == 3:
        error("Function used only for 3D.")

    q = TestFunction(V)
    psi = TrialFunction(V)
    n = FacetNormal(mesh)

    a = inner(grad(q), grad(psi))*dx - inner(q, dot(n, grad(psi)))*ds
    L = inner(q, curl(u_))*dx - dot(q, cross(n, u_))*ds
    
    # Compute solution
    A = assemble(a)
    b = assemble(L)

    solver = KrylovSolver('gmres', 'petsc_amg')
    solver.parameters['monitor_convergence'] = True
    solver.parameters['maximum_iterations'] = 1000
    solver.parameters['nonzero_initial_guess'] = True
    #solver.parameters['absolute_tolerance'] = 1e-8
    #solver.parameters['relative_tolerance'] = 1e-8
    solver.set_operator(A)
    dim = V.dim()
    n_comp = V.num_sub_spaces() 
    null_vectors = []
    vv = Function(V)
    for i in range(n_comp):
        null_vec_i = vv.vector().copy()
        V.sub(i).dofmap().set(null_vec_i, 1) 
        null_vec_i *= 1.0/null_vec_i.norm("l2")
        null_vectors.append(null_vec_i)

    null_space = VectorSpaceBasis(null_vectors)
    solver.set_nullspace(null_space)
    null_space.orthogonalize(b)

    psi = Function(V)
    solver.solve(psi.vector(), b)

    return psi
    