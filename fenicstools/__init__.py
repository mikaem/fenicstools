import warnings
from Streamfunctions import StreamFunction, StreamFunction3D

try:
    from Probe import Probe, Probes, StatisticsProbe, StatisticsProbes
except:
    warnings.warn("Probe/Probes/StatisticsProbe/StatisticsProbes not installed")

try:
    from StructuredGrid import StructuredGrid, ChannelGrid
except:
    warnings.warn("StructuredGrid/ChannelGrid not installed")

try:
    from WeightedGradient import weighted_gradient_matrix, compiled_gradient_module
except:
    warnings.warn("weighted_gradient_matrix/compiled_gradient_module not installed")
    
try:
    from common import getMemoryUsage, SetMatrixValue
except:
    warnings.warn("getMemoryUsage/SetMatrixValue not installed")

try:
    from GaussDivergence import gauss_divergence, divergence_matrix
except:
    warnings.warn("gauss_divergence/divergence_matrix not installed")

try:
    from Interpolation import interpolate_nonmatching_mesh, interpolate_nonmatching_mesh_any
except:
    warnings.warn("interpolate_nonmatching_mesh/interpolate_nonmatching_mesh_any not installed")

try:
    from DofMapPlotter import DofMapPlotter
except:
    warnings.warn("DofMapPlotter not installed") # Probably missing dependency

try:
    from ClementInterpolation import clement_interpolate
except:
    warnings.warn("ClementInterpolation not installed")
