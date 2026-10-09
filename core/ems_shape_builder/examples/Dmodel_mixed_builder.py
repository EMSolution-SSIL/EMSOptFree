"""
Dmodel_mixed_builder.py
The MIT License (MIT)
Copyright © 2025 Science Solutions International Laboratory, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the “Software”), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""  # noqa: N999

# ruff: noqa: F401
from enum import IntEnum

import numpy as np

from emsopt_engine.interface.ems_shape_builder_interface import EMSShapeBuilderInterface
from emsopt_engine.registry import ems_shape_builder


class MagnetTypeCategory(IntEnum):
    NdFeB = 0
    FerriteMagnet = 1


class DmodelMixedBuilder(EMSShapeBuilderInterface):
    def __init__(self) -> None:
        super().__init__()
        try:
            from eMotorSolution.CheckPoints.Rotor.IPMSM.IPM_HoleMagnet51 import IPM_HoleMagnet51Data
            from eMotorSolution.Project import Project
        except ModuleNotFoundError as e:
            msg = "eMotorSolution API not found. Please install it to use ems_shape_builder classes."
            raise ModuleNotFoundError(msg) from e

    def get_variable_dimension(self) -> int:
        """Get variable dimension

        Returns:
            int: dimension
        """
        return 5

    def update_shape(self, parameters: np.ndarray, project: object) -> bool:
        """update shape from parameters

        Args:
            parameters (np.ndarray): K-dimensional vector (K: number of parameters for shape design)
            project (object): eMotorSolution Project instance. For detail, please refer to eMotorSolution's API website.

        Returns:
            bool: True if succeeded, False otherwise

        Raises:
            TypeError: if the hole magnet type is not IPM_HoleMagnet51Data
        """
        from eMotorSolution.CheckPoints.Rotor.IPMSM.IPM_HoleMagnet51 import IPM_HoleMagnet51Data

        hole_magnet = project.rotor.hole_magnet.collection[0]
        try:
            hole_magnet.set_magnet_material(MagnetTypeCategory(int(parameters[0])).name)
        except ValueError as e:
            msg = f"First parameter must be 0 or 1. Got: {int(parameters[0])}"
            raise ValueError(msg) from e
        # set coil turns
        winding = project.stator.winding
        winding.set_turns_per_coil(int(parameters[1]))
        # set continuous geometric parameters
        hole_magnet.set_W0(parameters[2] + 0.2, "mm")  # magnet length with margin
        hole_magnet.set_W3(parameters[2], "mm")  # magnet length
        hole_magnet.set_H0(parameters[3], "mm")  # position
        hole_magnet.set_H2(parameters[4], "mm")  # magnet width
        hole_magnet.set_W1(50.0, "mm")
        hole_magnet.set_W2(0.1, "mm")  # margin / 2.0
        hole_magnet.set_H1(0.5, "mm")
        project.mesh.no_hole_mesh = True
        return hole_magnet.validate()["status"]


@ems_shape_builder("Dmodel_mixed_builder")
def build_Dmodel_mixed_builder() -> EMSShapeBuilderInterface:
    """shape builder for Dmodel_mixed_opt project"""
    return DmodelMixedBuilder()
