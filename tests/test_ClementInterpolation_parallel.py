from dolfin import *
from fenicstools.ClementInterpolation import clement_interpolate
import numpy as np


def test_parallel():
    '''Test if clement interpolation works in parallel'''
    mesh = UnitSquareMesh(30, 30)
    V = FunctionSpace(mesh, 'DG', 0)
    lhs = interpolate(Expression('std::abs(x[0]+x[1])', degree=1), V)
    # Compute first the interpolant
    ci = clement_interpolate(lhs)
    ci_values = ci.vector().array()
    
    # Check the logic manually
    V = ci.function_space()
    dofmap = V.dofmap()
    first, last = dofmap.ownership_range()
    v2d = vertex_to_dof_map(V)

    mesh.init(0, 2)
    offproc_bA, offproc_dof = [], []
    my_incomplete_dofs = set([])
    for vertex in vertices(mesh):
        # It is alway meaning full to compute locally
        patch_cells = [Cell(mesh, index) for index in vertex.entities(2)]
        volumes = [cell.volume() for cell in patch_cells]
        midpoints = np.array([[cell.midpoint().x(), cell.midpoint().y()]
                              for cell in patch_cells])
        b = sum(lhs(mp)*volume for mp, volume in zip(midpoints, volumes))
        A = sum(volumes)
    
        local_dof = v2d[vertex.index()]
        global_dof = dofmap.local_to_global_index(local_dof)
        is_owned = first <= global_dof < last
        # If the vertex is not shared the final answer can be computed and the
        # owner can compure the value with ci
        if not vertex.is_shared():
            assert is_owned
            value0 = b/A
            assert abs(ci_values[local_dof]-value0) < 1E-14
        
    # The way the offprocess things are handled is not supposed to be efficient
        else:
            # Each process collects global dof, b, A
            offproc_bA.append([b, A])
            offproc_dof.append(global_dof)
            # Record dofs which the process will check later
            if is_owned: my_incomplete_dofs.add(global_dof)

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
        assert abs(ci_values[dof-first]-value0) < 1E-14

# ----------------------------------------------------------------------------

if __name__ == '__main__':
    test_parallel()
