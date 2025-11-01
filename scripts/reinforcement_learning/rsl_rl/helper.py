# simple_step_dumper.py
import os
import torch

class SimpleStepDumper:
    """
    Dumb, reliable per-step saver.
    Writes one file per step using torch.save(..) with a plain dict:
        {'obs': obs, 'rewards': rewards, 'dones': dones, 'info': info}

    Notes:
    - Uses torch.save (i.e., Python pickle underneath).
    - Saves CPU copies of tensors so GPU memory isn’t held by the files.
    """
    def __init__(self, out_dir: str, prefix: str = "step"):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.prefix = prefix

    def _cpuify(self, x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu()
        if isinstance(x, (list, tuple)):
            return type(x)(self._cpuify(t) for t in x)
        if isinstance(x, dict):
            return {k: self._cpuify(v) for k, v in x.items()}
        return x

    def save(self, step: int, obs, actions, rewards, dones, info):
        payload = {
            "obs":     self._cpuify(obs),
            "actions": self._cpuify(actions),
            "rewards": self._cpuify(rewards),
            "dones":   self._cpuify(dones),
            "info":    self._cpuify(info),
        }
        path = os.path.join(self.out_dir, f"{self.prefix}-{step:08d}.pt")
        # print(f"[INFO]: Saving step data to: {path}")
        torch.save(payload, path)
        return path


class VectorizedEpisodeBuffer:
    """
    Holds per-env episode buffers and dumps only successful ones.
    Works with vectorized environments (N envs in parallel).
    """

    def __init__(self, dumper, num_envs=None, success_key="success", max_len=2000):
        self.dumper = dumper
        self.success_key = success_key
        self.max_len = max_len
        self.global_step = 0
        self.num_envs = None
        self.episode_ids = []
        self.buffers = []

        if num_envs is not None:
            self._init_structures(num_envs)

    def _init_structures(self, num_envs):
        if num_envs <= 0:
            raise ValueError("num_envs must be a positive integer.")
        self.num_envs = num_envs
        self.episode_ids = [0] * num_envs
        self.buffers = [[] for _ in range(num_envs)]

    def _cpuify(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu()
        if isinstance(obj, dict):
            return {k: self._cpuify(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(self._cpuify(v) for v in obj)
        return obj

    def _infer_batch_size(self, *objs):
        for obj in objs:
            size = self._infer_batch_size_from_obj(obj)
            if size is not None:
                return size
        return None

    def _infer_batch_size_from_obj(self, obj):
        if obj is None:
            return None
        if isinstance(obj, torch.Tensor):
            if obj.ndim == 0:
                return None
            return obj.shape[0]
        if isinstance(obj, (list, tuple)):
            if len(obj) == 0:
                return None
            return len(obj)
        if isinstance(obj, dict):
            for value in obj.values():
                size = self._infer_batch_size_from_obj(value)
                if size is not None:
                    return size
        return None

    def _split_batch(self, obj):
        if self.num_envs is None:
            raise RuntimeError("VectorizedEpisodeBuffer is not initialized with num_envs.")

        if obj is None:
            return [None] * self.num_envs

        if isinstance(obj, torch.Tensor):
            if obj.ndim == 0:
                # scalar broadcast
                return [obj.clone() for _ in range(self.num_envs)]
            return [obj[i] for i in range(self.num_envs)]

        if isinstance(obj, (list, tuple)):
            if len(obj) == self.num_envs:
                return list(obj)
            if len(obj) == 1:
                return [obj[0]] * self.num_envs
            raise ValueError(f"Cannot split batch of length {len(obj)} for {self.num_envs} environments.")

        if isinstance(obj, dict):
            per_env = [dict() for _ in range(self.num_envs)]
            for key, value in obj.items():
                split_val = self._split_batch(value)
                for i in range(self.num_envs):
                    per_env[i][key] = split_val[i]
            return per_env

        return [obj for _ in range(self.num_envs)]

    def _ensure_initialized(self, *objs):
        if self.num_envs is not None:
            return

        inferred = self._infer_batch_size(*objs)
        if inferred is None:
            raise ValueError(
                "Unable to determine the number of environments automatically. "
                "Please provide 'num_envs' when constructing VectorizedEpisodeBuffer."
            )
        self._init_structures(inferred)

    def add(self, obs, acts, rews, dones, infos):
        """
        obs, acts, rews, dones: tensors/arrays/batched dicts with first dim = num_envs
        infos: list[dict] or dict of batched entries
        """
        obs = self._cpuify(obs)
        acts = self._cpuify(acts)
        rews = self._cpuify(rews)
        dones = self._cpuify(dones)
        infos = self._cpuify(infos)

        self._ensure_initialized(obs, acts, rews, dones, infos)

        obs_items = self._split_batch(obs)
        act_items = self._split_batch(acts)
        rew_items = self._split_batch(rews)
        done_items = self._split_batch(dones)
        info_items = self._split_batch(infos)

        for i in range(self.num_envs):
            if len(self.buffers[i]) >= self.max_len:
                continue
            self.buffers[i].append(
                dict(
                    obs=obs_items[i],
                    actions=act_items[i],
                    rewards=rew_items[i],
                    dones=done_items[i],
                    info=info_items[i],
                )
            )

    def flush_done(self, dones):
        """Flush all envs whose done=True. Dumps only those with success=True."""
        dones = self._cpuify(dones)
        if self.num_envs is None:
            raise RuntimeError(
                "VectorizedEpisodeBuffer has not been initialized. "
                "Call 'add' at least once or provide 'num_envs' on construction before flushing."
            )

        if isinstance(dones, torch.Tensor):
            dones_iter = [bool(v.item()) for v in dones.reshape(-1)[: self.num_envs]]
        elif isinstance(dones, (list, tuple)):
            dones_iter = [bool(v) for v in dones[: self.num_envs]]
        else:
            dones_iter = [bool(dones) for _ in range(self.num_envs)]

        for i, done_flag in enumerate(dones_iter):
            if not bool(done_flag):
                continue
            ep = self.buffers[i]
            if not ep:
                continue
            success = False
            last_info = ep[-1]["info"]
            if isinstance(last_info, dict):
                success = bool(last_info.get(self.success_key, False))

            if success:
                print(f"[SUCCESS] Env {i} → dumping {len(ep)} steps (ep#{self.episode_ids[i]})")
                for s in ep:
                    self.dumper.save(
                        step=self.global_step,
                        obs=s["obs"],
                        actions=s["actions"],
                        rewards=s["rewards"],
                        dones=s["dones"],
                        info=s["info"],
                    )
                    self.global_step += 1
            else:
                print(f"[FAIL] Env {i} → skipped {len(ep)} steps (ep#{self.episode_ids[i]})")

            # reset buffer for next episode
            self.buffers[i] = []
            self.episode_ids[i] += 1

# self.goal_poses = torch.tensor([
#                                 [6.07, 3.34],   #Door
#                                 [(5.93, 0.1), (5.93, -1.92)],  #Working table
#                                 [0.63, -3.7],   #TV
#                                 [5.28, -0.56],  #Working chair black
#                                 [5.28, -1.44],  #Working chair orange
#                                 [4.4, 3.34],     #Mirror
#                                 [-3.98, -3.56],  #Front Trash can below fire extinguisher
#                                 [6.25, 1.48],   #Plant near door
#                                 [-2.74, 2.9],   #Plant near sofa
#                                 [6.00, -2.53],  #Floor Lamp
#                                 [(4.715, -3.16), (4.27, -3.2), (5.05, -3.2)], #Printer_table
#                                 [5.12, -3.15],  #Monitor near printer
#                                 [(3.08, 2.65), (2.27, 2.65), (1.34, 2.65)],    # 13: Wardrobe (Wall Cabinet)
#                                 [-4.01, 0.83],   # 14: Small Plant Right
#                                 [-4.01, -1.35],   # 15: Small Plant Left
#                                 [-0.14, 1.36],      # Green Cup
#                                 [-0.6, 1.369],  # Red Cup
#                                 [0.15, 2.05], # Green Pouf
#                                 [-1.45, 2.05], # Red Pouf
#                                 ], device=self.device, dtype=torch.float32
#                                 ) 