import unittest
import mlx.core as mx
import numpy as np
import torch
from diffusers.schedulers.scheduling_euler_discrete import (
    EulerDiscreteScheduler as DiffusersEulerDiscreteScheduler,
)
from mflux.contrib.schedulers.euler_discrete_scheduler import (
    SchedulerImplementation as MFluxEulerDiscreteScheduler,
)


def to_numpy(tensor):
    if isinstance(tensor, mx.array):
        return np.array(tensor)
    elif isinstance(tensor, torch.Tensor):
        return tensor.cpu().numpy()
    return tensor


class TestEulerDiscreteScheduler(unittest.TestCase):
    def assert_tensors_equal(self, mflux_tensor, diffusers_tensor, atol=1e-5):
        mflux_np = to_numpy(mflux_tensor)
        diffusers_np = to_numpy(diffusers_tensor)
        self.assertTrue(
            np.allclose(mflux_np, diffusers_np, atol=atol),
            f"Tensors not equal:\nMFlux: {mflux_np}\nDiffusers: {diffusers_np}",
        )

    def get_schedulers(self, **kwargs):
        mflux_scheduler = MFluxEulerDiscreteScheduler(**kwargs)
        diffusers_scheduler = DiffusersEulerDiscreteScheduler(**kwargs)
        return mflux_scheduler, diffusers_scheduler

    def get_dummy_data(self, shape=(1, 4, 64, 64)):
        sample_mflux = mx.random.normal(shape)
        sample_torch = torch.from_numpy(to_numpy(sample_mflux))
        return sample_mflux, sample_torch

    def test_initialization(self):
        mflux_scheduler, diffusers_scheduler = self.get_schedulers()
        self.assert_tensors_equal(mflux_scheduler.betas, diffusers_scheduler.betas)
        self.assert_tensors_equal(mflux_scheduler.alphas, diffusers_scheduler.alphas)
        self.assert_tensors_equal(
            mflux_scheduler.alphas_cumprod, diffusers_scheduler.alphas_cumprod
        )
        self.assert_tensors_equal(mflux_scheduler.sigmas, diffusers_scheduler.sigmas)

    def test_set_timesteps(self):
        for num_inference_steps in [5, 10, 50]:
            with self.subTest(num_inference_steps=num_inference_steps):
                mflux_scheduler, diffusers_scheduler = self.get_schedulers()
                mflux_scheduler.set_timesteps(num_inference_steps)
                diffusers_scheduler.set_timesteps(num_inference_steps)
                self.assert_tensors_equal(
                    mflux_scheduler.timesteps, diffusers_scheduler.timesteps
                )
                self.assert_tensors_equal(
                    mflux_scheduler.sigmas, diffusers_scheduler.sigmas
                )

    def test_step_prediction_types(self):
        for prediction_type in ["epsilon", "sample", "v_prediction"]:
            with self.subTest(prediction_type=prediction_type):
                num_inference_steps = 10
                mflux_scheduler, diffusers_scheduler = self.get_schedulers(
                    prediction_type=prediction_type, beta_schedule="scaled_linear"
                )
                mflux_scheduler.set_timesteps(num_inference_steps)
                diffusers_scheduler.set_timesteps(num_inference_steps)

                # Set seeds for deterministic noise generation
                mx.random.seed(42)
                torch.manual_seed(42)
                np.random.seed(42)
                
                sample_mflux, sample_torch = self.get_dummy_data()
                model_output_mflux, model_output_torch = self.get_dummy_data()

                for i, t in enumerate(diffusers_scheduler.timesteps):
                    mflux_t = mflux_scheduler.timesteps[i]

                    scaled_sample_mflux = mflux_scheduler.scale_model_input(
                        sample_mflux, mflux_t
                    )
                    scaled_sample_torch = diffusers_scheduler.scale_model_input(
                        sample_torch, t
                    )

                    # Test with gamma=0 (no stochastic noise) for deterministic comparison
                    mflux_output = mflux_scheduler.step(
                        model_output_mflux, mflux_t, scaled_sample_mflux, s_churn=0.0
                    )
                    diffusers_output = diffusers_scheduler.step(
                        model_output_torch, t, scaled_sample_torch, s_churn=0.0
                    )

                    self.assert_tensors_equal(
                        mflux_output.prev_sample,
                        diffusers_output.prev_sample,
                        atol=1e-4,
                    )
                    # The pred_original_sample is not always available in diffusers' implementation
                    if diffusers_output.pred_original_sample is not None:
                        self.assert_tensors_equal(
                            mflux_output.pred_original_sample,
                            diffusers_output.pred_original_sample,
                            atol=1e-4,
                        )

                    sample_mflux = mflux_output.prev_sample
                    sample_torch = diffusers_output.prev_sample

    def test_stochastic_behavior(self):
        """Test that stochastic sampling (s_churn > 0) produces different outputs each time."""
        # Create separate scheduler instances to avoid state interference
        mflux_scheduler1, _ = self.get_schedulers(prediction_type="epsilon")
        mflux_scheduler1.set_timesteps(4)
        mflux_scheduler2, _ = self.get_schedulers(prediction_type="epsilon") 
        mflux_scheduler2.set_timesteps(4)
        
        sample_mflux, _ = self.get_dummy_data()
        model_output_mflux, _ = self.get_dummy_data()
        mflux_t = mflux_scheduler1.timesteps[0]
        
        # Two calls with s_churn > 0 should produce different outputs due to random noise
        output1 = mflux_scheduler1.step(model_output_mflux, mflux_t, sample_mflux, s_churn=0.1)
        output2 = mflux_scheduler2.step(model_output_mflux, mflux_t, sample_mflux, s_churn=0.1)
        
        # Outputs should be different due to stochastic noise
        self.assertFalse(
            mx.allclose(output1.prev_sample, output2.prev_sample, atol=1e-6),
            "Stochastic sampling should produce different outputs"
        )
        
        # Test deterministic behavior with s_churn=0
        mflux_scheduler3, _ = self.get_schedulers(prediction_type="epsilon")
        mflux_scheduler3.set_timesteps(4)
        mflux_scheduler4, _ = self.get_schedulers(prediction_type="epsilon")
        mflux_scheduler4.set_timesteps(4)
        
        output3 = mflux_scheduler3.step(model_output_mflux, mflux_t, sample_mflux, s_churn=0.0)
        output4 = mflux_scheduler4.step(model_output_mflux, mflux_t, sample_mflux, s_churn=0.0)
        
        self.assertTrue(
            mx.allclose(output3.prev_sample, output4.prev_sample, atol=1e-6),
            "Deterministic sampling should produce identical outputs"
        )

    def test_add_noise(self):
        mflux_scheduler, diffusers_scheduler = self.get_schedulers()
        mflux_scheduler.set_timesteps(10)
        diffusers_scheduler.set_timesteps(10)

        original_samples_mflux, original_samples_torch = self.get_dummy_data()
        noise_mflux, noise_torch = self.get_dummy_data()

        timesteps_mflux = mflux_scheduler.timesteps[:1]
        timesteps_torch = torch.from_numpy(to_numpy(timesteps_mflux))

        noisy_samples_mflux = mflux_scheduler.add_noise(
            original_samples_mflux, noise_mflux, timesteps_mflux
        )
        noisy_samples_torch = diffusers_scheduler.add_noise(
            original_samples_torch, noise_torch, timesteps_torch
        )

        self.assert_tensors_equal(noisy_samples_mflux, noisy_samples_torch, atol=1e-4)

    def test_karras_sigmas(self):
        num_inference_steps = 10
        mflux_scheduler, diffusers_scheduler = self.get_schedulers(
            use_karras_sigmas=True
        )
        mflux_scheduler.set_timesteps(num_inference_steps)
        diffusers_scheduler.set_timesteps(num_inference_steps)

        self.assert_tensors_equal(mflux_scheduler.sigmas, diffusers_scheduler.sigmas)


if __name__ == "__main__":
    unittest.main()
