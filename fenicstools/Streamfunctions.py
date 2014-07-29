__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2010-11-03"
__copyright__ = "Copyright (C) 2010 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

import ufl
from dolfin import *

__all__ = ['StreamFunction', 'StreamFunction3D']

def StreamFunction(u, bcs, mesh, use_strong_bc = False, degree=1):
    """Stream function for a given general 2D velocity field.
    The boundary conditions are weakly imposed through the term
    
        inner(q, grad(psi)*n)*ds, 
    
    where grad(psi) = [-v, u] is set on all boundaries. 
    This should work for any collection of boundaries: 
    walls, inlets, outlets etc.    
    """
    # Check dimension
    if not mesh.geometry().dim() == 2:
        error("Stream-function can only be computed in 2D.")

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
    psi_ = Function(V)

    a = inner(grad(q), grad(psi))*dx - inner(q, dot(n, grad(psi)))*ds
    L = inner(q, curl(u))*dx - dot(q, cross(n, u))*ds
    
    # Compute solution
    A0 = assemble(inner(grad(q), grad(psi))*dx)
    L0 = inner(q, curl(u))*dx - dot(q, cross(n, u))*ds + inner(q, dot(n, grad(psi_)))*ds
    A = assemble(a)
    b = assemble(L)

    solver = KrylovSolver('bicgstab', 'jacobi')
    solver.parameters['monitor_convergence'] = True
    solver.parameters['maximum_iterations'] = 1000
    solver.parameters['nonzero_initial_guess'] = True
    solver.parameters['preconditioner']['structure'] = "same"
    solver.parameters['error_on_nonconvergence'] = False
    #solver.parameters['absolute_tolerance'] = 1e-8
    #solver.parameters['relative_tolerance'] = 1e-8
    #solver.set_operator(A0)
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
    
    #error = 1.
    #i = 0
    #d_psi = Vector(psi_.vector())
    #while error > 1e-6 and i < 50:
        #i += 1
        #b = assemble(L0, tensor=b)
        #solver.set_nullspace(null_space)
        #null_space.orthogonalize(b)
        #d_psi[:] = psi_.vector()
        #solver.solve(psi_.vector(), b)  
        #d_psi.axpy(-1., psi_.vector())
        #error = norm(d_psi)
        #print i, " Error = ", error, norm(b)

    b = assemble(L, tensor=b)
    solver.set_operator(A)
    solver.set_nullspace(null_space)
    null_space.orthogonalize(b)
    solver.solve(psi_.vector(), b)
    
    return psi_
    