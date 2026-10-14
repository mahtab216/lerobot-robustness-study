import gymnasium as gym
import numpy as np

class PerturbedPushTWrapper(gym.ObservationWrapper):
    """Gym wrapper applying camera shift, lighting changes, and observation noise."""
    def __init__(self, env, brightness=1.0, noise_std=0.0, camera_shift=(0, 0)):
        super().__init__(env)
        self.brightness = brightness
        self.noise_std = noise_std
        self.camera_shift = camera_shift

    def observation(self, obs):
        if isinstance(obs, dict) and "pixels" in obs:
            img = obs["pixels"].astype(np.float32)
            # 1. Apply Lighting Shift
            img = np.clip(img * self.brightness, 0, 255)
            # 2. Apply Spatial Camera Shift (Translation)
            if any(self.camera_shift):
                img = np.roll(img, shift=self.camera_shift, axis=(0, 1))
            # 3. Apply Gaussian Pixel Noise
            if self.noise_std > 0:
                noise = np.random.normal(0, self.noise_std * 255.0, img.shape)
                img = np.clip(img + noise, 0, 255)
            obs["pixels"] = img.astype(np.uint8)
        return obs
