import warnings
from fenicstools.Streamfunctions import StreamFunction, StreamFunction3D

try:
    from fenicstools.Probe import Probe, Probes, StatisticsProbe, StatisticsProbes
except:
    warnings.warn("Probe/Probes/StatisticsProbe/StatisticsProbes not installed")

try:
    from fenicstools.StructuredGrid import StructuredGrid, ChannelGrid
except:
    warnings.warn("StructuredGrid/ChannelGrid not installed")

try:
    from fenicstools.WeightedGradient import weighted_gradient_matrix, compiled_gradient_module
except:
    warnings.warn("weighted_gradient_matrix/compiled_gradient_module not installed")
    
try:
    from fenicstools.common import getMemoryUsage, SetMatrixValue
except:
    warnings.warn("getMemoryUsage/SetMatrixValue not installed")

try:
    from fenicstools.GaussDivergence import gauss_divergence, divergence_matrix
except:
    warnings.warn("gauss_divergence/divergence_matrix not installed")

try:
    from fenicstools.Interpolation import interpolate_nonmatching_mesh, interpolate_nonmatching_mesh_any
except:
    warnings.warn("interpolate_nonmatching_mesh/interpolate_nonmatching_mesh_any not installed")

try:
    from fenicstools.DofMapPlotter import DofMapPlotter
except:
    warnings.warn("DofMapPlotter not installed") # Probably missing dependency

try:
    from fenicstools.ClementInterpolation import clement_interpolate
except:
    warnings.warn("ClementInterpolation not installed")

try:
    from fenicstools.LagrangianParticles import LagrangianParticles
except:
    warnings.warn("Lagrangian particles not installed")
