from typing import Optional, Union, Tuple
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import gym
from tqdm import tqdm
import matplotlib.pyplot as plt

from networks import PixelWiseQNetwork, argmax3d
from utils import ReplayBuffer, plot_predictions, plot_curves


class DQNAgent:
    def __init__(self,
                 env,
                 gamma: float,
                 learning_rate: float,
                 buffer_size: int,
                 batch_size: int,
                 initial_epsilon: float,
                 final_epsilon: float,
                 update_method: str='standard',
                 exploration_fraction: float=0.9,
                 target_network_update_freq: int=1000,
                 seed: int=0,
                 device: Union[str, torch.device]='cpu',
                ) -> None:
        self.env = env

        self.gamma = gamma
        self.batch_size = batch_size
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.exploration_fraction = exploration_fraction
        self.target_network_update_freq = target_network_update_freq
        self.update_method = update_method

        self.buffer = ReplayBuffer(buffer_size,
                                   env.observation_space.shape,
                                   env.action_space.shape)

        self.device = device
        img_shape = (3, self.env.img_size, self.env.img_size)
        self.network = PixelWiseQNetwork(img_shape).to(device)
        self.target_network = PixelWiseQNetwork(img_shape).to(device)
        self.hard_target_update()

        self.optim = torch.optim.Adam(self.network.parameters(),
                                      lr= learning_rate)

        np.random.seed(seed)
        torch.manual_seed(seed)
        if device == 'cuda':
            torch.cuda.manual_seed(seed)

    def train(self, num_steps: int, plotting_freq: int=0) -> None:
        '''Train q-function for given number of environment steps using
        q-learning with e-greedy action selection

        Parameters
        ----------
        num_steps
            number of environment steps
        plotting_freq
            interval (in env steps) between plotting of training data, if 0
            then never plots.
        '''
        rewards_data = []
        success_data = []
        loss_data = []

        episode_count = 0
        episode_rewards = 0
        s = self.env.reset()

        pbar = tqdm(range(1, num_steps+1))
        for step in pbar:
            progress_fraction = step/(self.exploration_fraction*num_steps)
            epsilon = self.compute_epsilon(progress_fraction)
            a = self.select_action(s, epsilon)
            print(a)

            sp, r, done, info = self.env.step(a)
            episode_rewards += r

            self.buffer.add_transition(s=s, a=a, r=r, sp=sp, d=done)

            # optimize
            if len(self.buffer) > self.batch_size:
                loss = self.optimize()
                loss_data.append(loss)
                if len(loss_data) % self.target_network_update_freq == 0:
                    self.hard_target_update()

            s = sp.copy()
            if done:
                s = self.env.reset()
                print(info['success'])
                rewards_data.append(episode_rewards)
                success_data.append(info['success'])

                episode_rewards = 0
                episode_count += 1

                avg_success = np.mean(success_data[-min(episode_count, 50):])
                pbar.set_description(f'Success = {avg_success:.1%}')

            if plotting_freq > 0 and step % plotting_freq == 0:
                batch = self.buffer.sample(self.batch_size)
                imgs = self.prepare_batch(*batch)[0]

                with torch.no_grad():
                    q_map_pred = self.network(imgs)
                    actions = argmax3d(q_map_pred)
                plot_predictions(imgs, q_map_pred, actions)
                plot_curves(rewards_data, success_data, loss_data)
                plt.show()

        return rewards_data, success_data, loss_data

    def optimize(self) -> float:
        '''Optimizes q-network by minimizing td-loss on a batch sampled from
        replay buffer

        Returns
        -------
        mean squared td-loss across batch
        '''
        batch = self.buffer.sample(self.batch_size)
        s, a, r, sp, d = self.prepare_batch(*batch)

        q_map_pred = self.network(s)
        q_pred = q_map_pred[np.arange(len(s)), 0, a[:,0], a[:,1]]

        if self.update_method == 'standard':
            with torch.no_grad():
                q_map_next = self.target_network(sp)
                q_next = torch.max( torch.flatten(q_map_next, 1), dim=1)[0]
                q_target = r + self.gamma * q_next * (1-d)

        elif self.update_method == 'double':
            with torch.no_grad():
                q_map_next = self.target_network(sp)
                q_next = torch.flatten(q_map_next, torch.argmax(torch.flatten(q_map_next, 1), dim=1)[0])
                q_target = r + self.gamma * q_next * (1-d)
                

        assert q_pred.shape == q_target.shape
        self.optim.zero_grad()
        loss = self.network.compute_loss(q_pred, q_target)
        loss.backward()

        nn.utils.clip_grad_norm_(self.network.parameters(), 10)
        self.optim.step()

        return loss.item()

    def prepare_batch(self, s: np.ndarray, a: np.ndarray,
                      r: np.ndarray, sp: np.ndarray, d: np.ndarray,
                     ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        '''Converts components of transition from numpy arrays to tensors
        that are ready to be passed to q-network.  Make sure you send tensors
        to the right device!

        Parameters
        ----------
        s : array of state images, dtype=np.uint8, shape=(B, H, W, C)
        a : array of actions, dtype=np.int8, shape=(B, 2)
        r : array of rewards, dtype=np.float32, shape=(B,)
        sp : array of next state images, dtype=np.uint8, shape=(B, H, W, C)
        d : array of done flags, dtype=np.bool, shape=(B,)

        Returns
        ----------
        s : tensor of state images, dtype=torch.float32, shape=(B, C, H, W)
        a : tensor of actions, dtype=torch.long, shape=(B, 2)
        r : tensor of rewards, dtype=torch.float32, shape=(B,)
        sp : tensor of next state images, dtype=torch.float32, shape=(B, C, H, W)
        d : tensor of done flags, dtype=torch.float32, shape=(B,)
        '''



        s_permuted = (torch.from_numpy(s).to(device=self.device, dtype=torch.float32)/255).permute([0,3,1,2])
        sp_permuted = (torch.from_numpy(sp).to(device=self.device, dtype=torch.float32)/255).permute([0,3,1,2])


        s = torch.tensor(s_permuted, dtype=torch.float32).to(self.device)
        a = torch.tensor(a, dtype=torch.long).to(self.device)
        r = torch.tensor(r, dtype=torch.float32).to(self.device)
        sp = torch.tensor(sp_permuted, dtype=torch.float32).to(self.device)
        d = torch.tensor(d, dtype=torch.float32).to(self.device)


        return s, a, r, sp, d

    def select_action(self, state: np.ndarray, epsilon: float=0.) -> np.ndarray:
        '''Returns action based on e-greedy action selection.  With probability
        of epsilon, choose random action in environment action space, otherwise
        select argmax of q-function at given state

        Returns
        -------
        pixel action (px, py), dtype=int
        '''
        if np.random.random() < epsilon:
            return self.env.action_space.sample()
        else:
            return self.policy(state)

    def policy(self, state: np.ndarray) -> np.ndarray:
        '''Policy is the argmax over actions of the q-function at the given
        state. You will need to convert state to tensor on the device (similar
        to `prepare_batch`), then use `network.predict`.  Make sure to convert
        back to cpu before converting to numpy

        Returns
        -------
        pixel action (px, py); shape=(2,); dtype=int
        '''
        t = (torch.from_numpy(state).to(device=self.device, dtype=torch.float32)/255).permute([2,0,1]).unsqueeze(0)
        action = self.network.predict(t).squeeze().cpu().numpy()
        return action

    def compute_epsilon(self, fraction: float) -> float:
        '''Calculate epsilon value based on linear annealing schedule

        Parameters
        ----------
        fraction
            fraction of exploration time steps that have been taken
        '''
        fraction = np.clip(fraction, 0., 1.)
        return (1-fraction) * self.initial_epsilon \
                    + fraction * self.final_epsilon

    def hard_target_update(self):
        '''Update target network by copying weights from online network'''
        self.target_network.load_state_dict(self.network.state_dict())

    def save_network(self, dest: str='q_network.pt'):
        torch.save(self.network.state_dict(), dest)

    def load_network(self, model_path: str, map_location: str='cpu'):
        self.network.load_state_dict(torch.load(model_path,
                                                map_location=map_location))
        self.hard_target_update()


if __name__ == "__main__":
    from grasping_env import TopDownGraspingEnv
    env = TopDownGraspingEnv(render=False)

    agent = DQNAgent(env= env,
                     gamma= 0.50,
                     learning_rate= 1e-4,
                     buffer_size= 250,
                     batch_size= 64,
                     initial_epsilon= 0.3,
                     final_epsilon=0.01,
                     update_method='standard',
                     exploration_fraction=0.9,
                     target_network_update_freq= 200,
                     seed= 1,
                     device= 'cpu')

    agent.train(1000, 500)
