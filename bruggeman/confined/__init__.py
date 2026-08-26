"""B: Confined groundwater (solutions 700-800)

Submodules:

- bruggeman.confined.flow1d: One dimensional flow in continuous multi-layer systems (solutions 100)
- bruggeman.confined.flow2d
   - bruggeman.confined.flow2d.radial: Two-dimensional radial flow in continuous multi-layer systems (solutions 200)
   - bruggeman.confined.flow2d.general: General two-dimensional flow in continuous multi-layer systems (solutions 300)
"""

# ruff: noqa F401
from bruggeman.confined import flow1d, flow2d
from bruggeman.confined.flow1d import *
from bruggeman.confined.flow2d import *
