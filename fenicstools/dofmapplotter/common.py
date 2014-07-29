__author__ = 'Miroslav Kuchta <mirok@math.uio.no>'
__date__ = '2014-04-23'
__copyright__ = 'Copyright (C) 2013 ' + __author__
__license__ = 'GNU Lesser GPL version 3 or any later version'

import inspect
from os.path import abspath, join
from dolfin import compile_extension_module


def dmt_number_entities(mesh, tdim):
    'Number (global) mesh entities of topological dimension.'
    folder = abspath(join(inspect.getfile(inspect.currentframe()), '../cpp'))
    code = open(join(folder, 'dmt.cpp'), 'r').read()
    compiled_module = compile_extension_module(code=code)
    return compiled_module.dmt_number_entities(mesh, tdim)


class ParallelColorPrinter(object):

    '''Print color messages proceeded by the info about process number
    (and figure number).'''

    def __init__(self, mpi_rank, mpi_size, fig_num=None):
        self.mpi_rank = mpi_rank
        self.mpi_size = mpi_size
        self.fig_num = fig_num

        self.color_templates = {'blue': '\033[1;37;34m%s%s%s\033[0m',
                                'green': '\033[1;37;32m%s%s%s\033[0m',
                                'red': '\033[1;37;31m%s%s%s\033[0m',
                                'yellow': '\033[1;33;33m%s%s%s\033[0m',
                                'cyan': '\033[0;36;36m%s%s%s\033[0m',
                                'pink': '\033[1;31;31m%s%s%s\033[0m'}

    def __call__(self, string, color='', line_break=True):
        '''Print string in given color. With line_break=False the function
        behaves like `print string,`.'''
        if self.mpi_size > 1:
            mpi_string = 'Process number %d, ' % self.mpi_rank
        else:
            mpi_string = ''

        if self.fig_num is not None:
            fig_string = 'Figure %d : ' % self.fig_num
        else:
            fig_string = ''

        if color in self.color_templates:
            template = self.color_templates[color]
        else:
            template = '%s%s%s'

        print template % (mpi_string, fig_string, string),

        if line_break:
            print ''


def signature(V):
    '''Compute signature = nested list representation, of the function space.
    S, scalar space, --> 0 (number of subspaces)
    V, vector space in 3D, --> 3 (number or subspaces)
    T, tensor space in 3D, --> 9 (number of subspaces)
    M = [S, V, T] --> [0, 3, 9]
    N = [M, V] --> [[0, 3, 9], 3]'''
    n_sub = V.num_sub_spaces()
    if n_sub == 0:
        return n_sub
    else:
        # Catch vector and tensor spaces
        if sum(V.sub(i).num_sub_spaces() for i in range(n_sub)) == 0:
            return n_sub
        # Proper mixed space
        else:
            n_subs = [0] * n_sub
            for i in range(n_sub):
                Vi = V.sub(i)
                n_subs[i] = signature(Vi)
            return n_subs


def extract_dofmaps(V):
    '''Extract dofmap of every component of V.
    S, space whose signature is 0 has only one dofmap
    V, space whose signature is 3, has 3 dofmaps
    M=[S, V], space whose signature is [0, 3] has four dofmaps, M.sub(0) and
    (M.sub(0)).sub(i), i = 0, 1, 2'''
    signature_ = signature(V)
    if type(signature_) is int:
        if signature_ == 0:
            return [V.dofmap()]
        else:
            return [V.sub(i).dofmap() for i in range(signature_)]
    else:
        if type(signature_) is list:
            dofmaps = []
            for i in range(len(signature_)):
                Vi = V.sub(i)
                dofmaps += extract_dofmaps(Vi)
            return dofmaps


def flat_signature(V):
    '''Compute the flat signature of V, i.e. flatten the nested list signature
    into a single list. V can be a FunctionSpace of nested list signature.
    [[0, 3, 9, ], 3] --> [0, 3, 9, 3]'''
    if is_function_space(V):
        return flat_signature(signature(V))
    else:
        if type(V) is int:
            return [V]
        else:
            if type(V) is list:
                if all(isinstance(x, int) for x in V):
                    return V
                else:
                    return flat_signature(V[0]) + flat_signature(V[1:])


def bounds(V):
    '''Access extracted dofmaps of V. To get dofmaps of i-th component,
    loop over bounds[i]:bounds[i+1].'''
    # With function space compute signate
    if is_function_space(V):
        signature_ = signature(V)
    # Assume signature and continue
    else:
        signature_ = V
    signature_ = flat_signature(signature_)
    trans = lambda x: x if x else x + 1
    signature_ = map(trans, signature_)
    limits = partial_sum(signature_)
    limits = [0] + limits

    return limits


def subspace_index(flat_index, bounds_=None):
    '''Compute subspace index. Let M be vector space in 2d and have flat
    signature [0, 2] so that there are 2 dofmaps that can be extraced but
    they all belong to the same subspace. Thus the subspace index is [0, 0].'''
    # If arg if FunctionSpace get subspace indices of all dofmaps
    if is_function_space(flat_index):
        b = bounds(flat_index)
        f = range(b[-1])
        return subspace_index(f, b)
    else:
        assert bounds_ is not None

        if type(flat_index) is int:
            for i in range(len(bounds_)-1):
                if bounds_[i] <= flat_index < bounds_[i+1]:
                    return [i]
            return [-1]
        elif type(flat_index) is list:
            partial = lambda index : subspace_index(index, bounds_)
            return sum(map(partial, flat_index), [])


def partial_sum(xs):
    'Partial sum of a list.'
    sums = [0] * len(xs)
    for i in range(len(xs)):
        sums[i] = xs[i] + sum(xs[:i])
    return sums


def x_to_str(x):
    'Convert coordinate point to string'
    return ''.join(map(str, x.tolist()))


def is_function_space(V):
    'Check if V is function space.'
    try:
        V.dofmap()
        return True
    except:
        return False

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    from dolfin import *

    mesh = UnitSquareMesh(2, 2)

    S = FunctionSpace(mesh, 'CG', 1)
    V = VectorFunctionSpace(mesh, 'CG', 1)
    T = TensorFunctionSpace(mesh, 'CG', 1)
    M = MixedFunctionSpace([S, V, T])

    SS = MixedFunctionSpace([S, S])
    N = MixedFunctionSpace([V, SS])

    V = VectorFunctionSpace(mesh, 'CR', 1)
    Q = FunctionSpace(mesh, 'DG', 0)
    M = MixedFunctionSpace([V, Q])

    domain2d = Rectangle(-1, -1, 1, 1)
    domain3d = Box(-1, -1, -1, 1, 1, 1)

    mesh = Mesh(domain2d, 3)
    V = VectorFunctionSpace(mesh, 'CR', 1)
    Q = FunctionSpace(mesh, 'DG', 0)
    S = FunctionSpace(mesh, 'DG', 1)
    M = MixedFunctionSpace([V, Q, S])

    _dofmaps = extract_dofmaps(M)
    _bounds = bounds(M)
    print 'Signature of M:', signature(M)
    print 'Flat signature of M:', flat_signature(M)
    print 'Bounds of M:', _bounds
    print 'Sub. indices for comps [0, 1]:', subspace_index([0, 1], _bounds)
    print 'Sub. indices for all comps:', subspace_index(M)

    for i in range(len(_bounds) - 1):
        first = _bounds[i]
        last = _bounds[i + 1]
        for j in range(first, last):
            print _dofmaps[j].dofs()
