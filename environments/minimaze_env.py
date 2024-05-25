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
        self.observation_space = spaces.Dict({
            'image': original_image_space,
            'movement': spaces.Box(low=-1, high=1, shape=(2,), dtype=np.int32),
            'dir_change': spaces.Discrete(3),  # Assuming direction can be -1 (left), 0 (same), 1 (right)
        })

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
        movement = np.array([current_pos[0] - self.prev_agent_pos[0], current_pos[1] - self.prev_agent_pos[1]], dtype=np.int32)
        dir_change = (current_dir - self.prev_agent_dir + 4) % 4  # Normalize to [0, 3] range
        
        if dir_change == 3:
            dir_change = -1  # Represent left turn as -1
        elif dir_change == 2:
            dir_change = 0  # Represent no change as 0
        elif dir_change == 1:
            dir_change = 1  # Represent right turn as 1

        # Create the observation dict with the new information
        observation = {
            'image': obs,
            'movement': movement,
            'dir_change': dir_change
        }

        # Update the previous position and direction
        self.prev_agent_pos = current_pos
        self.prev_agent_dir = current_dir

        return observation


class MinigridMaze:
    def __init__(self, env_name, realtime_mode = False):
        
        # Set the environment rendering mode
        self._realtime_mode = realtime_mode
        render_mode = "human" if realtime_mode else "rgb_array"
            
        self._env = gym.make(env_name, agent_view_size = 3, tile_size=28, render_mode=render_mode)
        # Decrease the agent's view size to raise the agent's memory challenge
        # On MiniGrid-Memory-S7-v0, the default view size is too large to actually demand a recurrent policy.
        self._env = RGBImgPartialObsWrapper(self._env, tile_size=28)
        self._env = ImgObsWrapper(self._env)
        self._env = MovementObsWrapper(self._env)
        self._observation_space = spaces.Dict({
            'image': spaces.Box(low=0, high=1.0, shape=(3, 84, 84), dtype=np.float32),
            'movement': spaces.Box(low=-1, high=1, shape=(2,), dtype=np.int32),
            'dir_change': spaces.Discrete(3),
        })

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
        obs, _ = self._env.reset(seed=np.random.randint(0, 99))
        obs['image'] = obs['image'].astype(np.float32) / 255.
        # To conform PyTorch requirements, the channel dimension has to be first.
        obs['image'] = np.swapaxes(obs['image'], 0, 2)
        obs['image'] = np.swapaxes(obs['image'], 2, 1)
        
        return obs

    def step(self, action):
        obs, reward, done, truncated, info = self._env.step(action[0])
        self._rewards.append(reward)
        obs['image'] = obs['image'].astype(np.float32) / 255.
        if done or truncated:
            info = {"reward": sum(self._rewards),
                    "length": len(self._rewards)}
        else:
            info = None
        # To conform PyTorch requirements, the channel dimension has to be first.
        obs['image'] = np.swapaxes(obs['image'], 0, 2)
        obs['image'] = np.swapaxes(obs['image'], 2, 1)
        
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
