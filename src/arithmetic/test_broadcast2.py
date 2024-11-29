import os

import numexpr as ne
import numpy as np
import pytest


def data_figures():
    if os.getenv("CI"):
        return [0]
    else:  # pragma: nocover
        return range(9)


@pytest.fixture(params=data_figures(), scope="module")
def sample_data(request):
    n = 10**request.param
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    z = np.empty(n, dtype=np.float64)
    return request.param, x, y, z


group = "Round2 "


def test_np2(benchmark, sample_data):
    def foo(x, y, z):
        np.sin(x, out=x)
        np.power(x, 2.0, out=x)
        np.cos(y, out=y)
        np.power(y, 2.0, out=y)
        np.add(x, y, out=z)

    benchmark.group = group + f"10^{sample_data[0]}"
    benchmark(foo, *sample_data[1:])


def test_ne2(benchmark, sample_data):
    sin = np.sin  # noqa: F841
    cos = np.cos  # noqa: F841

    def foo(x, y, z):
        ne.evaluate("sin(x)**2 + cos(y)**2", out=z)

    benchmark.group = group + f"10^{sample_data[0]}"
    benchmark(foo, *sample_data[1:])


def test_torch2(benchmark, sample_data):
    import torch

    def foo(x, y, z):
        torch.sin(x, out=x)
        torch.pow(x, 2.0, out=x)
        torch.cos(y, out=y)
        torch.pow(y, 2.0, out=y)
        torch.add(x, y, out=z)

    benchmark.group = group + f"10^{sample_data[0]}"
    benchmark(
        foo,
        torch.from_numpy(sample_data[1]),
        torch.from_numpy(sample_data[2]),
        torch.from_numpy(sample_data[3]),
    )
