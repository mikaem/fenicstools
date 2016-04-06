from dolfin import *
from mpi4py import MPI as piMPI
import time
import ufl


class ClementInterpolant(object):
    '''
    This class implements efficient construction of Clement interpolant of an
    UFL-built expression. Here, the Clement interpolant is a CG_1 function over 
    mesh constructed in two steps (See Braess' Finite element book):
        1) For each mesh vertex xj let wj the union of cells that share the vertex 
           (i.e wj is the support of vj - the basis function of CG_1 function
           space such that vj(xj) = 1). Then Qj(expr) is an L2 projection of
           expr into constant field on wj.
        2) Set Ih(expr) = sum_j Qj(expr)vj.
    '''
    # Cannot interpolate expression with Arguments + things which are not well
    # defined at vertex
    black_listed = (ufl.Argument, ufl.MaxCellEdgeLength, ufl.MaxFacetEdgeLength,
                    ufl.MinCellEdgeLength, ufl.MinFacetEdgeLength,
                    ufl.FacetArea, ufl.FacetNormal, 
                    ufl.CellNormal, ufl.CellVolume)

    def __init__(self, expr, use_averaging=False):
        '''For efficient interpolation things are precomuputed here'''
        t0 = time.time()  
        # Analyze expr and raise if invalid
        terminals = _analyze_expr(expr)
        # Analyze shape and raise if expr cannot be represented
        _analyze_shape(expr.ufl_shape)
        shape = expr.ufl_shape
        # Extract mesh from expr operands and raise if it is not unique or missing
        mesh = _extract_mesh(terminals)
        # Compute things for constructing Q
        Q = FunctionSpace(mesh, 'DG', 0)
        q = TestFunction(Q)
        # Forms for L2 means [rhs]
        # Scalar, Vectors, Tensors are built from components
        # Translate expression into forms for individual components
        if len(shape) == 0: forms = [inner(expr, q)*dx]
        elif len(shape) == 1: forms = [inner(expr[i], q)*dx for i in range(shape[0])]
        else: forms = [inner(expr[i, j], q)*dx for i in range(shape[0]) for j in range(shape[1])]
        # Build averaging or summation operator for computing the interpolant
        # from L2 averaged components.
        V = FunctionSpace(mesh, 'CG', 1)
        volumes = assemble(inner(Constant(1), q)*dx)
        # Ideally we compute the averaging operator, then the interpolant is
        # simply A*component. I have not implemented this for backends other 
        # than PETSc. 
        is_petsc = parameters['linear_algebra_backend'] == 'PETSc'
        # FIXME: use_averaging is here to allow comparing avg vs sum and petsc
        if is_petsc and use_averaging:
            A = _construct_averaging_operator(V, volumes)
            patch_volumes = None
        # Then we go back to the old ways
        else:
            warning('Using summation!')
            A = _construct_summation_operator(V)
            # Precompute 'mass matrix inverse [lhs]
            patch_volumes = Function(V).vector()
            A.mult(volumes, patch_volumes)
            
            if is_petsc:
                patch_volumes = as_backend_type(patch_volumes)
                patch_volumes.vec().reciprocal()
            else:
                # Awkard poitwise inverse (inverting the mass matrix)
                patch_volumes.set_local(1./patch_volumes.get_local())
                patch_volumes.apply('insert')

        # Record time it takes to construct and also later the average call
        self.__init_time = time.time() - t0
        self.__ncalls, self.__total_call_time = 0., 0.
        # Collect stuff
        self.shape, self.V, self.A, self.patch_volumes, self.forms = \
                shape, V, A, patch_volumes, forms


    def __call__(self):
        '''Return the interpolant.'''
        shape, V, A, patch_volumes, forms = \
                self.shape, self.V, self.A, self.patch_volumes, self.forms
        
        self.__ncalls += 1
        t0 = time.time()
        # L2 means of comps to indiv. cells
        means = map(assemble, forms)
        # The interpolant (scalar, vector, tensor) is build from components
        components = []
        for mean in means:
            component = Function(V)
            # In case we have the averaging operator A*component computed the
            # final value. Otherwise it is rhs for L2 patch projection
            A.mult(mean, component.vector()) 
            # And to complete the interpolant we need to apply the precomputed
            # mass matrix inverse
            if not self.patch_volumes is None: component.vector()[:] *= patch_volumes

            components.append(component)

        # Finalize the interpolant
        # Scalar has same space as component
        if len(shape) == 0: 
            uh = components.pop()
        # Other ranks
        else:
            mesh = V.mesh()
            W = VectorFunctionSpace(mesh, 'CG', 1, dim=shape[0]) if len(shape) == 1 else\
                TensorFunctionSpace(mesh, 'CG', 1, shape=shape)
            uh = Function(W)
            assign(uh, components)
            # NOTE: assign might not use apply correctly. see 
            # https://bitbucket.org/fenics-project/dolfin/issues/587/functionassigner-does-not-always-call
            # So just to be sure
            uh.vector().apply('insert')

        self.__total_call_time += time.time() - t0

        return uh

    def timings(self, mpiop='avg', verbose=False):
        '''
        Statistics for construction and ncalls averaged time spent in __call__.
        In parallel these values are MPI.OP-ed reduced on all processes. 
        '''
        comm = self.V.mesh().mpi_comm().tompi4py()
        data = [self.__init_time, self.__total_call_time/self.__ncalls]

        ops = {'sum': piMPI.SUM, 'avg': piMPI.SUM, 'min': piMPI.MIN, 'max': piMPI.MAX}
        op = ops[mpiop]
        
        data = comm.allreduce(data, op=op)
        if mpiop == 'avg': data = [v/comm.size for v in data]

        if comm.rank == 0 and verbose: 
            GREEN = '\033[1;37;32m%s\033[0m'
            print '---- Clement Interpolant(stats for %d procs) ----' % comm.size
            print 'Construction time [s]              ', GREEN % ('%g' % data[0])
            print 'MPI-%s time per call [s](%d calls)' % (mpiop, self.__ncalls),  GREEN % ('%g' % data[1])
            print

        return data

# Workers--

def _analyze_expr(expr):
    '''
    A valid expr for Clement interpolation is defined only in terms of pointwise
    operations on finite element functions.
    '''
    # Elliminate forms
    if isinstance(expr, ufl.Form): raise ValueError('Expression is a form')
    # Elliminate expressions build from Trial/Test functions, FacetNormals 
    terminals = [t for t in ufl.algorithms.traverse_unique_terminals(expr)]
    if any(isinstance(t, ClementInterpolant.black_listed) for t in terminals):
        raise ValueError('Invalid expression (e.g. has Arguments as operand)')
    # At this point the expression is valid
    return terminals


def _analyze_shape(shape):
    '''
    The shape of expr that UFL can build is arbitrary but we only support
    scalar, rank-1 and rank-2(square) tensors.
    '''
    is_valid = len(shape) < 3 and (shape[0] == shape[1] if len(shape) == 2 else True)
    if not is_valid:
        raise ValueError('Interpolating Expr does not result rank-0, 1, 2 function')


def _extract_mesh(terminals):
    '''Get the common mesh of operands that make the expression.'''
    pairs = []
    for t in terminals:
        try: 
            mesh = t.function_space().mesh()
            pairs.append((mesh.id(), mesh))
        except AttributeError: 
            pass
    ids = set(id_ for id_, _ in pairs)
    # Unique mesh
    if len(ids) == 1: return pairs.pop()[1]
    # Mesh of Nones of multiple
    raise ValueError('Failed to extract mesh: Operands with no or different meshes')


def _construct_summation_operator(V):
    '''
    Summation matrix has the following properties: It is a map from DG0 to CG1.
    It has the same sparsity pattern as the mass matrix and in each row the nonzero
    entries are 1. Finally let v \in DG0 then (A*v)_i is the sum of entries of v
    that live on the support of i-th basis function of CG1.
    '''
    mesh = V.mesh()
    Q = FunctionSpace(mesh, 'DG', 0)
    q = TrialFunction(Q)
    v = TestFunction(V)
    tdim = mesh.topology().dim()
    K = CellVolume(mesh)
    dX = dx(metadata={'form_compiler_parameters': {'quadrature_degree': 1,
                                                   'quadrature_scheme': 'vertex'}})
    # This is a nice trick which uses properties of the vertex quadrature to get
    # only ones as nonzero entries.
    # NOTE: Its is designed spec for CG1. In particular does not work CG2 etc so
    # for such spaces a difference construction is required, e.g. rewrite nnz
    # entries of mass matric V, Q to 1. That said CG2 is the highest order where
    # clement interpolation makes sense. With higher ordered the dofs that are
    # interior to cell (or if there are multiple dofs par facet interior) are
    # assigned the same value.
    A = assemble((1./K)*Constant(tdim+1)*inner(v, q)*dX)

    return A


def _construct_averaging_operator(V, c):
    '''
    If b is the vectors of L^2 means of some u on the mesh, v is the vector
    of cell volumes and A is the summation oparotr then x=(Ab)/(Ac) are the
    coefficient of Clement interpolant of u in V. Here we construct an operator
    B such that x = Bb.
    '''
    assert parameters['linear_algebra_backend'] == 'PETSc'
    
    A = _construct_summation_operator(V)
    Ac = Function(V).vector()
    A.mult(c, Ac)
    # 1/Ac
    Ac = as_backend_type(Ac).vec()
    Ac.reciprocal()     
    # Scale rows
    mat = as_backend_type(A).mat()
    mat.diagonalScale(L=Ac)

    return A


def clement_interpolate(expr, with_CI=False):
    '''
    A free function for construting Clement interpolant of an expr. This is
    done by creating instance of ClementInterpolant and applying it. The
    instance is not cached. The function is intended for one time interpolation.
    However, should you decide to do the repeated interpolation use with_CI=True
    to return the interpolated function along with the ClementInterpolant
    instance. 
    '''
    ci = ClementInterpolant(expr)
    return (ci(), ci) if with_CI else ci()
