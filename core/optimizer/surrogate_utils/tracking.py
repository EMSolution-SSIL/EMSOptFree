"""
tracking.py
The MIT License (MIT)
Copyright © 2026 Science Solutions International Laboratory, Inc.

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

from collections.abc import Mapping

from emsopt_engine.individual import Individual
from utils.individuals_io import SURROGATE_INFO_ATTRIBUTE


def set_surrogate_info(individual: Individual, **fields: object) -> None:
    """Merge canonical surrogate bookkeeping fields into an individual."""
    info = dict(getattr(individual, SURROGATE_INFO_ATTRIBUTE, {}))
    info.update({key: value for key, value in fields.items() if value is not None})
    setattr(individual, SURROGATE_INFO_ATTRIBUTE, info)


def mark_surrogate_prediction(
    individual: Individual,
    predicted_objectives: list[float],
    predicted_label_values: Mapping[str, float] | None = None,
) -> None:
    """Record that the current individual state came from surrogate inference."""
    set_surrogate_info(
        individual,
        evaluation_method="surrogate",
        surrogate_used=True,
        true_evaluation_used=False,
        inference_failed=False,
        predicted_value=list(predicted_objectives),
        predicted_label_values=dict(predicted_label_values) if predicted_label_values is not None else None,
    )


def mark_true_evaluation(
    individual: Individual,
    *,
    predicted_value: list[float] | None = None,
    true_label_values: Mapping[str, float] | None = None,
) -> None:
    """Record the final true-evaluation result and keep any prediction trace."""
    fields: dict[str, object] = {
        "true_evaluation_used": True,
        "true_value": list(individual.metrics.objectives),
    }
    if true_label_values is not None:
        fields["true_label_values"] = dict(true_label_values)
    if predicted_value is None:
        fields["evaluation_method"] = "true"
        fields["surrogate_used"] = False
    else:
        fields["evaluation_method"] = "surrogate_then_true"
        fields["surrogate_used"] = True
        fields["predicted_value"] = predicted_value
        fields["inference_failed"] = False
    set_surrogate_info(individual, **fields)
