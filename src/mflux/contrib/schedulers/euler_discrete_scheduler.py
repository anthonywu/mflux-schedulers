import mlx.core as mx

from mflux.schedulers.base_scheduler import BaseScheduler

class EulerDiscreteScheduler(BaseScheduler):

    @property
    def sigmas(self) -> mx.array:
        """
        The sigma schedule for the diffusion process.
        """
        raise NotImplementedError("Schedulers must implement sigmas")

    def scale_model_input(self, latents: mx.array, t: int) -> mx.array:
        """
        Scale the denoising model input.
        """
        return NotImplementedError("todo")
