import os

import numexpr as ne
import numpy as np
import pytest
import torch


def data_figures():
    if os.getenv("CI"):
        return [0]
    else:  # pragma: nocover
        return range(9)


@pytest.fixture(params=data_figures(), scope="module")
def sample_data(request):
    n = 10**request.param
    x = np.linspace(0, 1, n, dtype=np.float64)
    y = np.linspace(0, 1, n, dtype=np.float64)
    z = np.empty(n, dtype=np.float64)
    return request.param, x, y, z


group = "Round1 "


def test_np1(benchmark, sample_data):
    def foo(x, y, z):
        np.multiply(2.0, y, out=y)
        np.multiply(4.0, x, out=x)
        np.add(x, y, out=z)

    benchmark.group = group + f"10^{sample_data[0]}"
    benchmark.name = "numpy"
    benchmark(foo, *sample_data[1:])


def test_ne1(benchmark, sample_data):
    def foo(x, y, z):
        ne.evaluate("2*y + 4*x", out=z)

    benchmark.group = group + f"10^{sample_data[0]}"
    benchmark.name = "numexpr"
    benchmark(foo, *sample_data[1:])


def test_torch1(benchmark, sample_data):
    def foo(x, y, z):
        torch.mul(2.0, y, out=y)
        torch.mul(4.0, x, out=x)
        torch.add(x, y, out=z)

    benchmark.name = "torch"

    benchmark.group = group + f"10^{sample_data[0]}"
    benchmark(
        foo,
        torch.from_numpy(sample_data[1]),
        torch.from_numpy(sample_data[2]),
        torch.from_numpy(sample_data[3]),
    )
