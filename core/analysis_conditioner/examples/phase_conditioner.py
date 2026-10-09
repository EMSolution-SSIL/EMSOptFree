"""
phase_conditioner.py
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
"""

import numpy as np

from emsopt_engine.interface.analysis_conditioner_interface import AnalysisConditionerInterface
from emsopt_engine.registry import analysis_conditioner


class PhaseConditioner(AnalysisConditionerInterface):
    def __init__(self, case_names: list[str], scale: float = 1) -> None:
        self.case_names = case_names
        self.scale = scale

    def get_variable_dimension(self) -> int:
        """Get variable dimension

        Returns:
            int: dimension
        """
        return len(self.case_names)

    def condition_analysis_case(self, parameters: np.ndarray, input_json: dict, case_name: str) -> dict:
        """condition analysis case by modifing input_json

        Args:
            parameters (np.ndarray): K-dimensional vector (K: number of parameters)
            input_json (dict): pyemsol input json data
            case_name (str): Analysis case name

        Returns:
            Modified input_json
        """
        modified = input_json.copy()
        for i in range(len(self.case_names)):
            if case_name == self.case_names[i]:
                for d in modified["18_Time_Function"]:
                    if "PHASE" in d:
                        d["PHASE"] += parameters[i] * self.scale
        return modified


@analysis_conditioner("phase_conditioner")
def build_phase_conditioner(case_names: list[str], scale: float = 1.0) -> AnalysisConditionerInterface:
    """Phase conditioner. Change phase according to optimization variable and scale.

    Args:
        case_names (list[str]): case names to apply phase conditioning.
        scale (float): scale of phase angle. Resultant phase angle [deg] will be: += scale * parameter

    Returns:
        AnalysisConditionerInterface: _description_
    """
    return PhaseConditioner(case_names, scale)
