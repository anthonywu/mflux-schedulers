import pytest

import mlx.core as mx
from mflux.config.config import Config
from mflux.config.model_config import ModelConfig
from mflux.config.runtime_config import RuntimeConfig
from mflux.contrib.schedulers import euler_discrete_scheduler


@pytest.fixture
def example_runtime_config():
    return RuntimeConfig(
        Config(
            # these are the only attributes relevant to schedulers
            num_inference_steps=14,
            width=1024,
            height=1024,
            scheduler="linear",
        ),
        ModelConfig.dev(),  # requires_sigma_shift=True
    )


def test_euler_discrete_scheduler(example_runtime_config):
    sched = euler_discrete_scheduler.EulerDiscreteScheduler(example_runtime_config)
    assert isinstance(sched.sigmas, mx.array)
    assert sched.sigmas.dtype == mx.float32
    assert len(sched.sigmas) == example_runtime_config.num_inference_steps + 1
    print(sched.sigmas)
