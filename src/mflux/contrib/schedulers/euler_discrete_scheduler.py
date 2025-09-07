import math
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union, TYPE_CHECKING

import mlx.core as mx
from mflux.schedulers.base_scheduler import BaseScheduler

from .interp import mx_interp

if TYPE_CHECKING:
    from mflux.config.runtime_config import RuntimeConfig


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@dataclass
class EulerDiscreteSchedulerOutput:
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`mx.array` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
        pred_original_sample (`mx.array` of shape `(batch_size, num_channels, height, width)` for images):
            The predicted denoised sample `(x_{0})` based on the model output from the current timestep.
            `pred_original_sample` can be used to preview progress or for guidance.
    """

    prev_sample: mx.array
    pred_original_sample: Optional[mx.array] = None


# Copied from diffusers.schedulers.scheduling_ddpm.betas_for_alpha_bar
def betas_for_alpha_bar(
    num_diffusion_timesteps,
    max_beta=0.999,
    alpha_transform_type="cosine",
):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of
    (1-beta) over time from t = [0,1].

    Contains a function alpha_bar that takes an argument t and transforms it to the cumulative product of (1-beta) up
    to that part of the diffusion process.


    Args:
        num_diffusion_timesteps (`int`): the number of betas to produce.
        max_beta (`float`): the maximum beta to use; use values lower than 1 to
                     prevent singularities.
        alpha_transform_type (`str`, *optional*, default to `cosine`): the type of noise schedule for alpha_bar.
                     Choose from `cosine` or `exp`

    Returns:
        betas (`mx.array`): the betas used by the scheduler to step the model outputs
    """
    if alpha_transform_type == "cosine":

        def alpha_bar_fn(t):
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    elif alpha_transform_type == "exp":

        def alpha_bar_fn(t):
            return math.exp(t * -12.0)

    else:
        raise ValueError(f"Unsupported alpha_transform_type: {alpha_transform_type}")

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
    return mx.array(betas, dtype=mx.float32)


# Copied from diffusers.schedulers.scheduling_ddim.rescale_zero_terminal_snr
def rescale_zero_terminal_snr(betas):
    """
    Rescales betas to have zero terminal SNR Based on https://huggingface.co/papers/2305.08891 (Algorithm 1)


    Args:
        betas (`mx.array`):
            the betas that the scheduler is being initialized with.

    Returns:
        `mx.array`: rescaled betas with zero terminal SNR
    """
    # Convert betas to alphas_bar_sqrt
    alphas = 1.0 - betas
    alphas_cumprod = mx.cumprod(alphas, axis=0)
    alphas_bar_sqrt = alphas_cumprod.sqrt()

    # Store old values.
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()

    # Shift so the last timestep is zero.
    alphas_bar_sqrt -= alphas_bar_sqrt_T

    # Scale so the first timestep is back to the old value.
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

    # Convert alphas_bar_sqrt to betas
    alphas_bar = alphas_bar_sqrt**2  # Revert sqrt
    alphas = alphas_bar[1:] / alphas_bar[:-1]  # Revert cumprod
    alphas = mx.concatenate([alphas_bar[0:1], alphas])
    betas = 1 - alphas

    return betas


class EulerDiscreteScheduler(BaseScheduler):
    """
    A wrapper class for the default Euler Discrete SchedulerImplementation that implementes the interface
    between mflux's RuntimeConfig and a Scheduler implementation with more/other args
    """

    def __init__(self, runtime_config: "RuntimeConfig"):
        self.impl = SchedulerImplementation()  # all default args for now
        self.impl.set_timesteps(runtime_config.num_inference_steps)
        self.scale_model_input = self.impl.scale_model_input

    @property
    def sigmas(self):
        return self.impl.sigmas

    def step(
        self, model_output: mx.array, timestep: int, sample: mx.array, **kwargs
    ) -> mx.array:
        """
        Perform one Euler discrete denoising step.

        Args:
            model_output: The noise prediction from the transformer
            timestep: Current timestep index (0 to num_inference_steps-1)
            sample: Current latent representation
            **kwargs: Additional scheduler parameters (s_churn, s_tmin, s_tmax, s_noise)

        Returns:
            Updated latents after one Euler discrete step
        """
        # Convert timestep index to actual timestep value from the schedule
        timestep_value = self.impl.timesteps[timestep]
        result = self.impl.step(
            model_output=model_output,
            timestep=timestep_value,
            sample=sample,
            return_dict=True,  # Ensure we get the dataclass output
            **kwargs,
        )

        # Return just the denoised sample for mflux
        return result.prev_sample


@dataclass
class SchedulerImplementation:
    num_train_timesteps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02
    beta_schedule: str = "linear"
    trained_betas: Optional[Union[mx.array, List[float]]] = None
    prediction_type: str = "epsilon"
    interpolation_type: str = "linear"
    use_karras_sigmas: Optional[bool] = False
    use_exponential_sigmas: Optional[bool] = False
    use_beta_sigmas: Optional[bool] = False
    sigma_min: Optional[float] = None
    sigma_max: Optional[float] = None
    timestep_spacing: str = "linspace"
    timestep_type: str = "discrete"  # can be "discrete" or "continuous"
    steps_offset: int = 0
    rescale_betas_zero_snr: bool = False
    final_sigmas_type: str = "zero"  # can be "zero" or "sigma_min"

    timesteps: mx.array = field(default_factory=lambda: mx.array([]))
    betas: mx.array = field(default_factory=lambda: mx.array([]))
    alphas: mx.array = field(default_factory=lambda: mx.array([]))
    alphas_cumprod: mx.array = field(default_factory=lambda: mx.array([]))
    _sigmas: mx.array = field(init=False, repr=False)

    num_inference_steps: Optional[int] = None
    is_scale_input_called: bool = False

    order = 1

    def __post_init__(self):
        if (
            self.use_beta_sigmas
            and self.use_exponential_sigmas
            and self.use_karras_sigmas
        ):
            raise ValueError(
                "Only one of `use_beta_sigmas`, `use_exponential_sigmas`, `use_karras_sigmas` can be used."
            )

        if self.trained_betas is not None:
            self.betas = mx.array(self.trained_betas, dtype=mx.float32)
        elif self.beta_schedule == "linear":
            self.betas = mx.linspace(
                self.beta_start,
                self.beta_end,
                self.num_train_timesteps,
                dtype=mx.float32,
            )
        elif self.beta_schedule == "scaled_linear":
            self.betas = (
                mx.linspace(
                    self.beta_start**0.5,
                    self.beta_end**0.5,
                    self.num_train_timesteps,
                    dtype=mx.float32,
                )
                ** 2
            )
        elif self.beta_schedule == "squaredcos_cap_v2":
            self.betas = betas_for_alpha_bar(self.num_train_timesteps)
        else:
            raise NotImplementedError(
                f"{self.beta_schedule} is not implemented for {self.__class__}"
            )

        if self.rescale_betas_zero_snr:
            self.betas = rescale_zero_terminal_snr(self.betas)

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = mx.cumprod(self.alphas, axis=0)

        if self.rescale_betas_zero_snr:
            self.alphas_cumprod[-1] = 2**-24

        sigmas = ((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5
        sigmas = sigmas[::-1]
        timesteps = mx.linspace(
            0, self.num_train_timesteps - 1, self.num_train_timesteps, dtype=mx.float32
        )[::-1]
        timesteps = mx.array(timesteps).astype(mx.float32)

        if (
            self.timestep_type == "continuous"
            and self.prediction_type == "v_prediction"
        ):
            self.timesteps = mx.array([0.25 * sigma.log() for sigma in sigmas])
        else:
            self.timesteps = timesteps

        self.sigmas = mx.concatenate([sigmas, mx.zeros(1)])

        self._step_index = None
        self._begin_index = None

    @property
    def init_noise_sigma(self):
        max_sigma = (
            max(self.sigmas) if isinstance(self.sigmas, list) else self.sigmas.max()
        )
        if self.timestep_spacing in ["linspace", "trailing"]:
            return max_sigma

        return (max_sigma**2 + 1) ** 0.5

    @property
    def step_index(self):
        return self._step_index

    @property
    def begin_index(self):
        return self._begin_index

    def set_begin_index(self, begin_index: int = 0):
        self._begin_index = begin_index

    def scale_model_input(
        self, sample: mx.array, timestep: Union[float, mx.array]
    ) -> mx.array:
        if self.step_index is None:
            self._init_step_index(timestep)

        sigma = self.sigmas[self.step_index]
        sample = sample / ((sigma**2 + 1) ** 0.5)

        self.is_scale_input_called = True
        return sample

    def set_timesteps(
        self,
        num_inference_steps: int = None,
        device: Union[str, mx.Device] = None,
        timesteps: Optional[List[int]] = None,
        sigmas: Optional[List[float]] = None,
    ):
        if timesteps is not None and sigmas is not None:
            raise ValueError("Only one of `timesteps` or `sigmas` should be set.")
        if num_inference_steps is None and timesteps is None and sigmas is None:
            raise ValueError(
                "Must pass exactly one of `num_inference_steps` or `timesteps` or `sigmas`."
            )
        if num_inference_steps is not None and (
            timesteps is not None or sigmas is not None
        ):
            raise ValueError(
                "Can only pass one of `num_inference_steps` or `timesteps` or `sigmas`."
            )
        if timesteps is not None and self.use_karras_sigmas:
            raise ValueError("Cannot set `timesteps` with `use_karras_sigmas = True`.")
        if timesteps is not None and self.use_exponential_sigmas:
            raise ValueError(
                "Cannot set `timesteps` with `use_exponential_sigmas = True`."
            )
        if timesteps is not None and self.use_beta_sigmas:
            raise ValueError("Cannot set `timesteps` with `use_beta_sigmas = True`.")
        if (
            timesteps is not None
            and self.timestep_type == "continuous"
            and self.prediction_type == "v_prediction"
        ):
            raise ValueError(
                "Cannot set `timesteps` with `timestep_type = 'continuous'` and `prediction_type = 'v_prediction'`."
            )

        if num_inference_steps is None:
            num_inference_steps = (
                len(timesteps) if timesteps is not None else len(sigmas) - 1
            )
        self.num_inference_steps = num_inference_steps

        if sigmas is not None:
            log_sigmas = mx.log(
                mx.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
            )
            sigmas = mx.array(sigmas).astype(mx.float32)
            timesteps = mx.array(
                [self._sigma_to_t(sigma, log_sigmas) for sigma in sigmas[:-1]]
            )

        else:
            if timesteps is not None:
                timesteps = mx.array(timesteps).astype(mx.float32)
            else:
                if self.timestep_spacing == "linspace":
                    timesteps = mx.linspace(
                        0,
                        self.num_train_timesteps - 1,
                        num_inference_steps,
                        dtype=mx.float32,
                    )[::-1]
                    timesteps = mx.array(timesteps)
                elif self.timestep_spacing == "leading":
                    step_ratio = self.num_train_timesteps // self.num_inference_steps
                    timesteps = mx.array(
                        (mx.arange(0, num_inference_steps) * step_ratio).round()[::-1]
                    ).astype(mx.float32)
                    timesteps += self.steps_offset
                elif self.timestep_spacing == "trailing":
                    step_ratio = self.num_train_timesteps / self.num_inference_steps
                    timesteps = mx.array(
                        (mx.arange(self.num_train_timesteps, 0, -step_ratio)).round()
                    ).astype(mx.float32)
                    timesteps -= 1
                else:
                    raise ValueError(
                        f"{self.timestep_spacing} is not supported. Please make sure to choose one of 'linspace', 'leading' or 'trailing'."
                    )

            sigmas = mx.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
            log_sigmas = mx.log(sigmas)
            if self.interpolation_type == "linear":
                sigmas = mx_interp(timesteps, mx.arange(0, len(sigmas)), sigmas)
            elif self.interpolation_type == "log_linear":
                sigmas = mx.exp(
                    mx.linspace(
                        mx.log(sigmas[-1]), mx.log(sigmas[0]), num_inference_steps + 1
                    )
                )
            else:
                raise ValueError(
                    f"{self.interpolation_type} is not implemented. Please specify interpolation_type to either"
                    " 'linear' or 'log_linear'"
                )

            if self.use_karras_sigmas:
                sigmas = self._convert_to_karras(
                    in_sigmas=sigmas, num_inference_steps=self.num_inference_steps
                )
                timesteps = mx.array(
                    [self._sigma_to_t(sigma, log_sigmas) for sigma in sigmas]
                )

            elif self.use_exponential_sigmas:
                sigmas = self._convert_to_exponential(
                    in_sigmas=sigmas, num_inference_steps=num_inference_steps
                )
                timesteps = mx.array(
                    [self._sigma_to_t(sigma, log_sigmas) for sigma in sigmas]
                )

            elif self.use_beta_sigmas:
                raise NotImplementedError(
                    "mlx does not provide scipy.stats.beta.ppf reference in diffusers implementation. Consider later."
                )

            if self.final_sigmas_type == "sigma_min":
                sigma_last = (
                    (1 - self.alphas_cumprod[0]) / self.alphas_cumprod[0]
                ) ** 0.5
            elif self.final_sigmas_type == "zero":
                sigma_last = 0
            else:
                raise ValueError(
                    f"`final_sigmas_type` must be one of 'zero', or 'sigma_min', but got {self.final_sigmas_type}"
                )

            sigmas = mx.concatenate([sigmas, mx.array([sigma_last])]).astype(mx.float32)

        sigmas = mx.array(sigmas).astype(mx.float32)

        if (
            self.timestep_type == "continuous"
            and self.prediction_type == "v_prediction"
        ):
            self.timesteps = mx.array([0.25 * sigma.log() for sigma in sigmas[:-1]])
        else:
            self.timesteps = mx.array(timesteps.astype(mx.float32))

        self._step_index = None
        self._begin_index = None
        self.sigmas = sigmas

    def _sigma_to_t(self, sigma, log_sigmas):
        log_sigma = mx.log(mx.maximum(sigma, 1e-10))
        dists = log_sigma - mx.expand_dims(log_sigmas, 1)
        low_idx = mx.cumsum((dists >= 0), axis=0).argmax(axis=0)
        low_idx = mx.clip(low_idx, a_min=0, a_max=log_sigmas.shape[0] - 2)
        high_idx = low_idx + 1
        low = log_sigmas[low_idx]
        high = log_sigmas[high_idx]
        w = (low - log_sigma) / (low - high)
        w = mx.clip(w, a_min=0, a_max=1)
        t = (1 - w) * low_idx + w * high_idx
        t = t.reshape(sigma.shape)
        return t

    def _convert_to_karras(self, in_sigmas: mx.array, num_inference_steps) -> mx.array:
        sigma_min = (
            self.sigma_min if self.sigma_min is not None else in_sigmas[-1].item()
        )
        sigma_max = (
            self.sigma_max if self.sigma_max is not None else in_sigmas[0].item()
        )

        rho = 7.0
        ramp = mx.linspace(0, 1, num_inference_steps)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return sigmas

    def _convert_to_exponential(
        self, in_sigmas: mx.array, num_inference_steps: int
    ) -> mx.array:
        sigma_min = (
            self.sigma_min if self.sigma_min is not None else in_sigmas[-1].item()
        )
        sigma_max = (
            self.sigma_max if self.sigma_max is not None else in_sigmas[0].item()
        )

        sigmas = mx.exp(
            mx.linspace(math.log(sigma_max), math.log(sigma_min), num_inference_steps)
        )
        return sigmas

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        mask = schedule_timesteps == timestep
        indices = [i for i in range(len(mask)) if bool(mask[i])]
        pos = 1 if len(indices) > 1 else 0
        return indices[pos]

    def _init_step_index(self, timestep):
        if self.begin_index is None:
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def step(
        self,
        model_output: mx.array,
        timestep: Union[float, mx.array],
        sample: mx.array,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        return_dict: bool = True,
    ) -> Union[EulerDiscreteSchedulerOutput, Tuple]:
        if isinstance(timestep, int):
            raise ValueError(
                (
                    "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                    " `EulerDiscreteScheduler.step()` is not supported. Make sure to pass"
                    " one of the `scheduler.timesteps` as a timestep."
                ),
            )

        if not self.is_scale_input_called:
            logger.warning(
                "The `scale_model_input` function should be called before `step` to ensure correct denoising. "
                "See `StableDiffusionPipeline` for a usage example."
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        sample = sample.astype(mx.float32)
        sigma = self.sigmas[self.step_index]
        gamma = (
            min(s_churn / (len(self.sigmas) - 1), 2**0.5 - 1)
            if s_tmin <= sigma <= s_tmax
            else 0.0
        )
        sigma_hat = sigma * (gamma + 1)

        if gamma > 0:
            noise = mx.random.normal(model_output.shape, dtype=model_output.dtype)
            eps = noise * s_noise
            sample = sample + eps * (sigma_hat**2 - sigma**2) ** 0.5

        if (
            self.prediction_type == "original_sample"
            or self.prediction_type == "sample"
        ):
            pred_original_sample = model_output
        elif self.prediction_type == "epsilon":
            pred_original_sample = sample - sigma_hat * model_output
        elif self.prediction_type == "v_prediction":
            pred_original_sample = model_output * (-sigma / (sigma**2 + 1) ** 0.5) + (
                sample / (sigma**2 + 1)
            )
        else:
            raise ValueError(
                f"prediction_type given as {self.prediction_type} must be one of `epsilon`, or `v_prediction`"
            )

        derivative = (sample - pred_original_sample) / sigma_hat
        dt = self.sigmas[self.step_index + 1] - sigma_hat
        prev_sample = sample + derivative * dt
        prev_sample = prev_sample.astype(model_output.dtype)
        self._step_index += 1

        if not return_dict:
            return (
                prev_sample,
                pred_original_sample,
            )

        return EulerDiscreteSchedulerOutput(
            prev_sample=prev_sample, pred_original_sample=pred_original_sample
        )

    def add_noise(
        self,
        original_samples: mx.array,
        noise: mx.array,
        timesteps: mx.array,
    ) -> mx.array:
        sigmas = self.sigmas.astype(original_samples.dtype)
        schedule_timesteps = self.timesteps.astype(mx.float32)
        timesteps = timesteps.astype(mx.float32)

        if self.begin_index is None:
            step_indices = [
                self.index_for_timestep(t, schedule_timesteps) for t in timesteps
            ]
        elif self.step_index is not None:
            step_indices = [self.step_index] * timesteps.shape[0]
        else:
            step_indices = [self.begin_index] * timesteps.shape[0]

        sigma = mx.flatten(sigmas[step_indices])
        while len(sigma.shape) < len(original_samples.shape):
            sigma = mx.expand_dims(sigma, -1)

        noisy_samples = original_samples + noise * sigma
        return noisy_samples

    def get_velocity(
        self, sample: mx.array, noise: mx.array, timesteps: mx.array
    ) -> mx.array:
        if isinstance(timesteps, int):
            raise ValueError(
                (
                    "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                    " `EulerDiscreteScheduler.get_velocity()` is not supported. Make sure to pass"
                    " one of the `scheduler.timesteps` as a timestep."
                ),
            )

        schedule_timesteps = self.timesteps.astype(mx.float32)
        timesteps = timesteps.astype(mx.float32)

        step_indices = [
            self.index_for_timestep(t, schedule_timesteps) for t in timesteps
        ]
        alphas_cumprod = self.alphas_cumprod
        sqrt_alpha_prod = mx.flatten(alphas_cumprod[step_indices] ** 0.5)
        while len(sqrt_alpha_prod.shape) < len(sample.shape):
            sqrt_alpha_prod = mx.expand_dims(sqrt_alpha_prod, -1)

        sqrt_one_minus_alpha_prod = mx.flatten(
            (1 - alphas_cumprod[step_indices]) ** 0.5
        )
        while len(sqrt_one_minus_alpha_prod.shape) < len(sample.shape):
            sqrt_one_minus_alpha_prod = mx.expand_dims(sqrt_one_minus_alpha_prod, -1)

        velocity = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample
        return velocity

    def __len__(self):
        return self.num_train_timesteps
