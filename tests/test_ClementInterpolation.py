#!/usr/bin/env py.test
from __future__ import division
from dolfin import *
import fenicstools.ClementInterpolation as ci
import numpy as np


def test_analyze_extract():
    '''Test quering expressions for Clement interpolation.'''
    mesh = UnitSquareMesh(40, 40)
    V = FunctionSpace(mesh, 'CG', 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Function(V)
    g = Expression('x[0]', degree=1)
    c = Constant(1)
    n = FacetNormal(mesh)
    e = Function(FunctionSpace(mesh, 'CG', 2))
    x = Function(FunctionSpace(UnitIntervalMesh(100), 'CG', 2))
    
    # Raises
    expressions = (u, v, inner(u, v), inner(f, v), dot(grad(f), n),
                   inner(grad(f), grad(v)), inner(f, f)*dx, inner(f, f)*ds)
    count = 0
    for expr in expressions:
        try: ci._analyze_expr(expr)
        except ValueError:
            count += 1
    assert len(expressions) == count

    # Pass analysis
    expressions = (f, grad(f), inner(f, f) + inner(grad(f), grad(f)), inner(f, g)+c, 
                   grad(f)[0], f+g, inner(f, g), c+e, inner(grad(e), grad(f)),
                   x+e, c, g)
    terminals = map(ci._analyze_expr, expressions)

    assert all(ci._extract_mesh(term) for i, term in enumerate(terminals[:9]))

    # Fails to extract
    count = 0
    for term in terminals[9:]:
        try: ci._extract_mesh(term)
        except ValueError:
            count += 1
    assert 3 == count


def test_parallel(mesh=None):
    '''Test if clement interpolation works in parallel'''
    meshes = (IntervalMesh(100, -1, 2),
              RectangleMesh(Point(-1, -2), Point(2, 4), 10, 10),
              BoxMesh(Point(0, 0, 0), Point(1, 2, 3), 3, 3, 3))
   
    # Run across all dims
    if mesh is None: 
        return [test_parallel(mesh) for mesh in range(len(meshes))]

    mesh = meshes[mesh]
    V = FunctionSpace(mesh, 'DG', 0)
    gdim = mesh.geometry().dim()
    lhs = interpolate(Expression('std::abs(%s)' % '+'.join(['x[%d]' % i for i in range(gdim)]),
                                 degree=1),
                      V)
    # Compute first the interpolant
    uh, CI = ci.clement_interpolate(lhs, True)
    uh_values = uh.vector().array()
    
    # Check the logic manually
    V = uh.function_space()
    dofmap = V.dofmap()
    first, last = dofmap.ownership_range()
    v2d = vertex_to_dof_map(V)

    tdim = mesh.topology().dim()
    mesh.init(0, tdim)
    offproc_bA, offproc_dof = [], []
    my_incomplete_dofs = set([])
    for vertex in vertices(mesh):
        # It is alway meaning full to compute locally
        patch_cells = [Cell(mesh, index) for index in vertex.entities(tdim)]
        volumes = [cell.volume() for cell in patch_cells]
        midpoints = np.array([[cell.midpoint()[i] for i in range(gdim)]
                              for cell in patch_cells])
        b = sum(lhs(mp)*volume for mp, volume in zip(midpoints, volumes))
        A = sum(volumes)
    
        local_dof = v2d[vertex.index()]
        global_dof = dofmap.local_to_global_index(local_dof)
        is_owned = first <= global_dof < last
        # If the vertex is not shared the final answer can be computed and the
        # owner can compure the value with uh
        if not vertex.is_shared():
            assert is_owned
            value0 = b/A
            assert abs(uh_values[local_dof]-value0) < 1E-14
        
    # The way the offprocess things are handled is not supposed to be efficient
        else:
            # Each process collects global dof, b, A
            offproc_bA.append([b, A])
            offproc_dof.append(global_dof)
            # Record dofs which the process will check later
            if is_owned: 
                my_incomplete_dofs.add(global_dof)
                # This what we will use to look up the value
                assert global_dof - first == local_dof

    # Communicate
    comm = mesh.mpi_comm().tompi4py()

    offproc_dof = np.array(offproc_dof)
    offproc_dof = comm.allgather(offproc_dof)

    offproc_bA = np.array(offproc_bA).flatten()
    offproc_bA = comm.allgather(offproc_bA)

    # Now the the process that own the dof can look up b, A from all the
    # processes sum them and compute the final result. Finally do the comparison
    offproc_dof = [vec.tolist() for vec in offproc_dof]
    offproc_bA = [vec.reshape((-1, 2)) for vec in offproc_bA]
    for dof in my_incomplete_dofs:
        bA = np.zeros(2)
        # Look up
        for rank in range(comm.size):
            try:
                # Add if found
                index = offproc_dof[rank].index(dof)
                bA += offproc_bA[rank][index]

                del offproc_dof[rank][index]
                offproc_bA[rank] = np.delete(offproc_bA[rank], index, 0)
            except ValueError:
                pass
        # Final
        b, A = bA
        value0 = b/A
        # Compare
        assert abs(uh_values[dof-first]-value0) < 1E-14

    CI.timings()


def test_parallel_avg(mesh=None):
    '''Test logic of averaging operator'''
    meshes = (IntervalMesh(100, -1, 2),
              RectangleMesh(Point(-1, -2), Point(2, 4), 10, 10),
              BoxMesh(Point(0, 0, 0), Point(1, 2, 3), 3, 3, 3))
   
    # Run across all dims
    if mesh is None: 
        return [test_parallel_avg(mesh) for mesh in range(len(meshes))]

    mesh = meshes[mesh]
    V = FunctionSpace(mesh, 'CG', 1)
    A = ci._construct_averaging_operator(V)
    # Shape
    Q = FunctionSpace(mesh, 'DG', 0)
    assert (A.size(0), A.size(1)) == (V.dim(), Q.dim())
    # All nonzero values are 1
    entries = np.unique(A.array().flatten())
    assert all(near(e, 1, 1E-14) for e in entries[np.abs(entries) > 1E-14])

    # The action on a vector of cell volumes should give vector volumes of
    # supports of CG1 functions
    q = TestFunction(Q)
    volumes = assemble(inner(Constant(1), q)*dx)
    patch_volumes = Function(V).vector()
    A.mult(volumes, patch_volumes)

    patch_volumes = patch_volumes.array()
    # Check the logic manually
    dofmap = V.dofmap()
    first, last = dofmap.ownership_range()
    v2d = vertex_to_dof_map(V)

    tdim = mesh.topology().dim()
    mesh.init(0, tdim)
    offproc_b, offproc_dof = [], []
    my_incomplete_dofs = set([])
    for vertex in vertices(mesh):
        # It is alway meaning full to compute locally
        b = sum(Cell(mesh, index).volume() for index in vertex.entities(tdim))
    
        local_dof = v2d[vertex.index()]
        global_dof = dofmap.local_to_global_index(local_dof)
        is_owned = first <= global_dof < last
        # If the vertex is not shared the final answer can be computed and the
        # owner can compute the value with ci
        if not vertex.is_shared():
            assert is_owned
            value0 = b
            assert abs(patch_volumes[local_dof]-value0) < 1E-14
        
    # The way the offprocess things are handled is not supposed to be efficient
        else:
            # Each process collects global dof, b
            offproc_b.append(b)
            offproc_dof.append(global_dof)
            # Record dofs which the process will check later
            if is_owned: 
                my_incomplete_dofs.add(global_dof)
                # This what we will use to look up the value
                assert global_dof - first == local_dof

    # Communicate
    comm = mesh.mpi_comm().tompi4py()

    offproc_dof = np.array(offproc_dof)
    offproc_dof = comm.allgather(offproc_dof)

    offproc_b = np.array(offproc_b)
    offproc_b = comm.allgather(offproc_b)

    # Now the the process that own the dof can look up b from all the
    # processes sum them and compute the final result. Finally do the comparison
    offproc_dof = [vec.tolist() for vec in offproc_dof]
    offproc_b = [vec.tolist() for vec in offproc_b]
    for dof in my_incomplete_dofs:
        b = 0.
        # Look up
        for rank in range(comm.size):
            try:
                # Add if found
                index = offproc_dof[rank].index(dof)
                b += offproc_b[rank][index]

                del offproc_dof[rank][index]
                del offproc_b[rank][index]
            except ValueError:
                pass
        # Final
        value0 = b
        # Compare
        assert abs(patch_volumes[dof-first]-value0) < 1E-14

if __name__ == '__main__':
    # test_analyze_extract()
    # test_parallel_avg() 
    test_parallel()
