# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Cartpole balancing environment.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-MyBuddy-Direct-v0",
    entry_point=f"{__name__}.mybuddy_env:MyBuddyEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.mybuddy_env:MyBuddyEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CartpolePPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_trpo_cfg_entry_point": f"{agents.__name__}:skrl_trpo_cfg.yaml",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_amp_cfg.yaml",
        "skrl_sac_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
        "sb3_ppo_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "sb3_sac_cfg_entry_point": f"{agents.__name__}:sb3_sac_cfg.yaml",
        "sb3_trpo_cfg_entry_point": f"{agents.__name__}:sb3_trpo_cfg.yaml",
        "sb3_rppo_cfg_entry_point": f"{agents.__name__}:sb3_rppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-MyBuddy-Direct-SAC-v0",
    entry_point=f"{__name__}.mybuddy_env_sac:MyBuddyEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.mybuddy_env_sac:MyBuddyEnvCfg",
        "sb3_sac_cfg_entry_point": f"{agents.__name__}:sb3_sac_cfg.yaml"
    },
)

# gym.register(
#     id="Isaac-Cartpole-RGB-Camera-Direct-v0",
#     entry_point=f"{__name__}.cartpole_camera_env:CartpoleCameraEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.cartpole_camera_env:CartpoleRGBCameraEnvCfg",
#         "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_camera_ppo_cfg.yaml",
#         "skrl_cfg_entry_point": f"{agents.__name__}:skrl_camera_ppo_cfg.yaml",
#     },
# )

# gym.register(
#     id="Isaac-Cartpole-Depth-Camera-Direct-v0",
#     entry_point=f"{__name__}.cartpole_camera_env:CartpoleCameraEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.cartpole_camera_env:CartpoleDepthCameraEnvCfg",
#         "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_camera_ppo_cfg.yaml",
#         "skrl_cfg_entry_point": f"{agents.__name__}:skrl_camera_ppo_cfg.yaml",
#     },
# )
