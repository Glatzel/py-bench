import os

import numpy as np
import pytest
import torch

group = "Channel merge: "


def data_figures():
    if os.getenv("CI"):
        return [0]
    return range(5, 9)


@pytest.fixture(params=data_figures(), scope="module")
def audio(request):
    rng = np.random.default_rng(1337)

    return request.param, rng.random((10**request.param, 2), dtype=np.float32)


def test_numpy(benchmark, audio):
    def foo(data: np.ndarray):
        data.mean(1)

    benchmark.group = group + f"10^{audio[0]}"
    benchmark.name = "numpy"
    benchmark(foo, audio[1])


def test_numpy2(benchmark, audio):
    def foo(data: np.ndarray):
        np.mean(data, axis=1)

    benchmark.group = group + f"10^{audio[0]}"
    benchmark.name = "numpy2"
    benchmark(foo, audio[1])


def test_torch(benchmark, audio):
    def foo(data: torch.Tensor):
        data.mean(1)

    benchmark.group = group + f"10^{audio[0]}"
    benchmark.name = "torch"
    benchmark(foo, torch.from_numpy(audio[1]))


def test_torch2(benchmark, audio):
    def foo(data: torch.Tensor):
        torch.mean(data, dim=1)

    benchmark.group = group + f"10^{audio[0]}"
    benchmark.name = "torch2"
    benchmark(foo, torch.from_numpy(audio[1]))
