# your_env_module.py

import gymnasium as gym
from grid2op.gym_compat import DiscreteActSpace, MultiDiscreteActSpace
from gymnasium.spaces import MultiDiscrete
import grid2op
from grid2op import gym_compat
from grid2op.Parameters import Parameters
from grid2op.Action import PlayableAction
from grid2op.Observation import CompleteObservation
from grid2op.Reward import L2RPNReward, N1Reward, CombinedScaledReward
from lightsim2grid import LightSimBackend
from gymnasium.spaces import Discrete

class Gym2OpEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self._backend = LightSimBackend()
        self._env_name = "l2rpn_case14_sandbox"  # DO NOT CHANGE
        action_class = PlayableAction
        observation_class = CompleteObservation
        reward_class = CombinedScaledReward

        # DO NOT CHANGE Parameters
        # See https://grid2op.readthedocs.io/en/latest/parameters.html
        # Setup parameters
        p = Parameters()
        p.MAX_SUB_CHANGED = 4
        p.MAX_LINE_STATUS_CHANGED = 4

        # Make grid2op env
        self._g2op_env = grid2op.make(
            self._env_name, backend=self._backend, test=False,
            action_class=action_class, observation_class=observation_class,
            reward_class=reward_class, param=p
        )

        ##########
        # REWARD #
        ##########
        # NOTE: This reward should not be modified when evaluating RL agent
        # See https://grid2op.readthedocs.io/en/latest/reward.html
        cr = self._g2op_env.get_reward_instance()
        cr.addReward("N1", N1Reward(), 1.0)
        cr.addReward("L2RPN", L2RPNReward(), 1.0)
        cr.initialize(self._g2op_env)

        self._gym_env = gym_compat.GymEnv(self._g2op_env)
        self.observation_space = self._gym_env.observation_space
        self.setup_actions()

    def setup_observations(self):
        return
        #  See Grid2Op 'getting started' notebooks for guidance
        #  - Notebooks: https://github.com/rte-france/Grid2Op/tree/master/getting_started

    def setup_actions(self):
        # self.action_space = MultiDiscrete(
        #     #[2] * 57 + [2] * 20 + [3] * 57 + [3] * 20
        #     [3] * 57 + [3] * 20
        # )
        # print("Flattened action space setup complete.")
        act_attr_to_keep = ["set_line_status", "set_bus"]
        self._gym_env.action_space = DiscreteActSpace(self._g2op_env.action_space, attr_to_keep=act_attr_to_keep)
        self.action_space = Discrete(self._gym_env.action_space.n)


    def reset(self, seed=None):
        return self._gym_env.reset(seed=seed, options=None)

    def step(self, action):
        # unflatten the action into original components and then combine them
        # combined_action = {
        #     #'change_bus': action[:57],
        #     #'change_line_status': action[57:77],
        #     #'set_bus': action[77:134],
        #     #'set_line_status': action[134:]
        #     'set_bus': action[:57],
        #     'set_line_status': action[57:]
        # }
        return self._gym_env.step(action)

    def render(self):
        return self._gym_env.render()
