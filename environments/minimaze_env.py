import time
import gymnasium as gym
from gymnasium import spaces
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
import numpy as np

class MovementObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(MovementObsWrapper, self).__init__(env)
        self.prev_agent_pos = None
        self.prev_agent_dir = None

        # Extend the observation space to include the movement and direction change
        original_image_space = self.observation_space
        flattened_image_size = np.prod(original_image_space.shape)

        # Extend the observation space to include the flattened image, movement, and direction change
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(flattened_image_size + 2 + 1,),  # Flattened image + 2 for movement + 1 for dir_change
            dtype=np.float32
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_agent_pos = self.env.unwrapped.agent_pos
        self.prev_agent_dir = self.env.unwrapped.agent_dir
        return self._get_obs(obs), info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        return self._get_obs(obs), reward, done, truncated, info

    def _get_obs(self, obs):
        current_pos = self.env.unwrapped.agent_pos
        current_dir = self.env.unwrapped.agent_dir

        # Calculate movement and direction change
        movement = np.array([current_pos[0] - self.prev_agent_pos[0], current_pos[1] - self.prev_agent_pos[1]], dtype=np.float32)
        dir_change = (current_dir - self.prev_agent_dir + 4) % 4  # Normalize to [0, 3] range

        if dir_change == 3:
            dir_change = -1  # Represent left turn as -1
        elif dir_change == 2:
            dir_change = 0  # Represent no change as 0
        elif dir_change == 1:
            dir_change = 1  # Represent right turn as 1

        # Flatten the image observation and normalize it
        flattened_image = obs.astype(np.float32).flatten() / 255.0

        # Concatenate all parts into a 1D vector
        final_obs = np.concatenate((flattened_image, movement, [dir_change])).astype(np.float32)

        # Update the previous position and direction
        self.prev_agent_pos = current_pos
        self.prev_agent_dir = current_dir

        return final_obs


class MinigridMaze:
    def __init__(self, env_name, realtime_mode = False):
        
        # Set the environment rendering mode
        self._realtime_mode = realtime_mode
        render_mode = "human" if realtime_mode else "rgb_array"
            
        self._env = gym.make(env_name, agent_view_size = 3, tile_size=4, render_mode=render_mode)
        # Decrease the agent's view size to raise the agent's memory challenge
        # On MiniGrid-Memory-S7-v0, the default view size is too large to actually demand a recurrent policy.
        self._env = RGBImgPartialObsWrapper(self._env, tile_size=4)
        self._env = ImgObsWrapper(self._env)
        self._env = MovementObsWrapper(self._env)
        # Define the observation space based on the final wrapped environment
        self._observation_space = self._env.observation_space

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        # This reduces the agent's action space to the only relevant actions (rotate left/right, move forward)
        # to solve the Minigrid-Memory environment.
        return spaces.Discrete(3)

    def reset(self):
        self._rewards = []
        obs, info = self._env.reset(seed=np.random.randint(0, 99))
        return obs

    def step(self, action):
        obs, reward, done, truncated, info = self._env.step(action[0])
        self._rewards.append(reward)
        if done or truncated:
            info = {"reward": sum(self._rewards),
                    "length": len(self._rewards)}
        else:
            info = None
        return obs, reward, done or truncated, info

    def render(self):
        img = self._env.render()
        time.sleep(0.5)
        return img

    def close(self):
        self._env.close()

if __name__ == "__main__":
    env = MinigridMaze("MiniGrid-MemoryS9-v0")
    obs = env.reset()
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)
    print("Initial observation:", obs)
    print("Initial observation shape:", obs['image'].shape)
    print("Step 1")
    obs, reward, done, info = env.step([2])
    print("Observation:", obs)
    print("Reward:", reward)
    print("Done:", done)
    print("Info:", info)
    print("Step 2")
    obs, reward, done, info = env.step([1])
    print("Observation:", obs)
    print("Reward:", reward)
    print("Done:", done)
    print("Info:", info)
    env.render()
    env.close()
