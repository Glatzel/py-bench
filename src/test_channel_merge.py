import os

import numpy as np
import pytest
import torch

group = "Channel merge: "


def data_figures():
    if os.getenv("CI"):
        return [1]
    else:  # pragma: nocover
        return range(5, 9)


@pytest.fixture(params=data_figures(), scope="module")
def audio(request):
    rng = np.random.default_rng(1337)
    return request.param, rng.random((10**request.param, 2), dtype=np.float32)


# channel, wave
def test_numpy_method_cw(benchmark, audio):
    def foo(data: np.ndarray):
        data.mean(1)

    benchmark.group = group + f"10^{audio[0]}"
    benchmark.name = "numpy method cw"
    benchmark(foo, audio[1])


def test_numpy_function_cw(benchmark, audio):
    def foo(data: np.ndarray):
        np.mean(data, axis=1)

    benchmark.group = group + f"10^{audio[0]}"
    benchmark.name = "numpy function cw"
    benchmark(foo, audio[1])


def test_torch_method_cw(benchmark, audio):
    def foo(data: torch.Tensor):
        data.mean(1)

    benchmark.group = group + f"10^{audio[0]}"
    benchmark.name = "torch method cw"
    benchmark(foo, torch.from_numpy(audio[1]))


def test_torch_function_cw(benchmark, audio):
    def foo(data: torch.Tensor):
        torch.mean(data, dim=1)

    benchmark.group = group + f"10^{audio[0]}"
    benchmark.name = "torch function cw"
    benchmark(foo, torch.from_numpy(audio[1]))


#  wave, channel
def test_numpy_method_wc(benchmark, audio):
    def foo(data: np.ndarray):
        data.mean(1)

    benchmark.group = group + f"10^{audio[0]}"
    benchmark.name = "numpy method wc"
    benchmark(foo, audio[1].T)


def test_numpy_function_wc(benchmark, audio):
    def foo(data: np.ndarray):
        np.mean(data, axis=1)

    benchmark.group = group + f"10^{audio[0]}"
    benchmark.name = "numpy function wc"
    benchmark(foo, audio[1].T)


def test_torch_method_wc(benchmark, audio):
    def foo(data: torch.Tensor):
        data.mean(1)

    benchmark.group = group + f"10^{audio[0]}"
    benchmark.name = "torch method wc"
    benchmark(foo, torch.from_numpy(audio[1].T))


def test_torch_function_wc(benchmark, audio):
    def foo(data: torch.Tensor):
        torch.mean(data, dim=1)

    benchmark.group = group + f"10^{audio[0]}"
    benchmark.name = "torch function wc"
    benchmark(foo, torch.from_numpy(audio[1].T))
