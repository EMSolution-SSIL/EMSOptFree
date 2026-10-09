"""
factory.py
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

from collections.abc import Callable

from core.optimizer.surrogate_models.protocol import SurrogateProtocol

SurrogateResolver = Callable[..., SurrogateProtocol]


def build_default_mlp_surrogate(dim: int, num_obj: int, seed: int | None) -> SurrogateProtocol:
    try:
        from core.optimizer.surrogate_models.mlp_pytorch import MLPSurrogate
    except ImportError as exc:
        raise ImportError("TorchSurrogate requires PyTorch. Install it separately to use this feature.") from exc

    num_nodes = [dim, 512, 256, 128, num_obj]
    return MLPSurrogate(num_nodes, seed=seed)


def build_default_random_forest_surrogate(
    dim: int,  # noqa: ARG001
    num_obj: int,  # noqa: ARG001
    seed: int | None,
    n_estimators: int = 200,
    max_depth: int | None = None,
) -> SurrogateProtocol:
    from core.optimizer.surrogate_models.random_forest import RandomForestSurrogate

    return RandomForestSurrogate(n_estimators=n_estimators, max_depth=max_depth, random_state=seed)


# you can register your own surrogate model and its factory function,
# and use it by giving the name in optimization.yaml config file.
_SURROGATE_MODEL_REGISTRY: dict[str, SurrogateResolver] = {
    "mlp": build_default_mlp_surrogate,
    "random_forest": build_default_random_forest_surrogate,
}


def validate_surrogate_model(surrogate: object) -> SurrogateProtocol:
    required_methods = ("is_ready", "fit", "update", "predict")
    missing = [name for name in required_methods if not callable(getattr(surrogate, name, None))]
    if missing:
        msg = f"surrogate must implement SurrogateProtocol. Missing callable methods: {', '.join(missing)}"
        raise TypeError(msg)
    return surrogate


def resolve_surrogate_model(*, surrogate_model: str, **kwargs: dict) -> SurrogateProtocol:
    if surrogate_model in _SURROGATE_MODEL_REGISTRY:
        return validate_surrogate_model(_SURROGATE_MODEL_REGISTRY[surrogate_model](**kwargs))
    msg = f"Unknown surrogate model name: {surrogate_model}"
    raise ValueError(msg)
