# import time
# from typing import Optional, Tuple, Union
#
# import gym
# import numpy as np
# from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.type_aliases import GymStepReturn
#
#
# class TimeLimitMonitor(Monitor):
#     """
#     Does the same as stable_baselines3.common.monitor.Monitor but takes care of environments without TimeLimitWrapper
#     """
#
#     def __init__(self,
#                  env: gym.Env,
#                  filename: Optional[str] = None,
#                  allow_early_resets: bool = True,
#                  reset_keywords: Tuple[str, ...] = (),
#                  info_keywords: Tuple[str, ...] = ()):
#         super().__init__(env, filename, allow_early_resets, reset_keywords, info_keywords)
#
#         self._max_episode_steps = env.spec.max_episode_steps  # or 1000
#
#     def step(self, action: Union[np.ndarray, int]) -> GymStepReturn:
#         # Monitor for done flag
#         observation, reward, done, info = super().step(action)
#
#         # Monitor when max episode steps are reached.
#         # When done=True this information is already provided.
#         if len(self.rewards) >= self._max_episode_steps and not done:
#             ep_rew = sum(self.rewards)
#             ep_len = len(self.rewards)
#             ep_info = {"r": round(ep_rew, 6), "l": ep_len, "t": round(time.time() - self.t_start, 6)}
#             for key in self.info_keywords:
#                 ep_info[key] = info[key]
#             self.episode_returns.append(ep_rew)
#             self.episode_lengths.append(ep_len)
#             self.episode_times.append(time.time() - self.t_start)
#             ep_info.update(self.current_reset_info)
#             if self.results_writer:
#                 self.results_writer.write_row(ep_info)
#             info["episode"] = ep_info
#
#             # normally, we would require a reset on done, however for infinite horizon (i.e. no TimiLimitWrapper),
#             # we just want the stats returned, after the fixed trajectory length
#             # Hence, we manually reset the rewards here instead
#             self.rewards = []
#
#         return observation, reward, done, info
