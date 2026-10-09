"""
HoleMagnet55_builder.py
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

import numpy as np

from emsopt_engine.interface.ems_shape_builder_interface import EMSShapeBuilderInterface
from emsopt_engine.registry import ems_shape_builder


class HoleMagnet55Builder(EMSShapeBuilderInterface):
    def __init__(self) -> None:
        super().__init__()
        try:
            from eMotorSolution.CheckPoints.Rotor.IPMSM.IPM_HoleMagnet55 import IPM_HoleMagnet55Data
            from eMotorSolution.Project import Project
        except ModuleNotFoundError as e:
            msg = "eMotorSolution API not found. Please install it to use ems_shape_builder classes."
            raise ModuleNotFoundError(msg) from e

    def get_variable_dimension(self) -> int:
        """Get variable dimension

        Returns:
            int: dimension
        """
        return 10

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
        from eMotorSolution.CheckPoints.Rotor.IPMSM.IPM_HoleMagnet55 import IPM_HoleMagnet55Data

        hole_magnet = project.rotor.hole_magnet.collection[0]
        if not isinstance(hole_magnet, IPM_HoleMagnet55Data):
            msg = f"IPM_HoleMagnet55Data is expected for rotor magnet. Got: {type(hole_magnet)}"
            raise TypeError(msg)
        # set parameters
        hole_magnet.set_W0(parameters[0], "mm")
        hole_magnet.set_W1(parameters[1], "mm")
        hole_magnet.set_W2(parameters[2], "mm")
        hole_magnet.set_W3(parameters[3], "mm")
        hole_magnet.set_W4(parameters[4], "mm")
        hole_magnet.set_H0(parameters[5], "mm")
        hole_magnet.set_H1(parameters[6], "mm")
        hole_magnet.set_H2(parameters[7], "mm")
        hole_magnet.set_H3(parameters[8], "mm")
        hole_magnet.set_H4(parameters[9], "mm")

        return hole_magnet.validate()["status"]


@ems_shape_builder("HoleMagnet55")
def build_HoleMagnet55_builder() -> EMSShapeBuilderInterface:
    """shape builder for magnet of Dmodel"""
    return HoleMagnet55Builder()
