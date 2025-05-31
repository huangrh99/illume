import torch
from collections import OrderedDict

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


import torch
import torch.distributed as dist


class SimpleDistributedEMA:
    def __init__(self, model, decay=0.9999, device=None, distributed=False):
        """
        :param model: The model to track with EMA.
        :param decay: EMA decay rate.
        :param device: Device to store EMA parameters (defaults to 'cuda' if available).
        :param distributed: If True, use distributed mode (shard EMA state); otherwise, track the full state.
        """
        self.decay = decay
        self.distributed = distributed
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.distributed and dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

        self._init_state(model)

    def _init_state(self, model):
        """Initialize the EMA state from the model's state_dict, distributing by numel."""
        full_state = model.state_dict()
        if self.distributed and self.world_size > 1:
            keys = sorted(full_state.keys())
            param_numel = {k: full_state[k].numel() for k in keys}  # 计算每个参数的 numel
            total_numel = sum(param_numel.values())  # 总 numel
            target_numel_per_rank = total_numel / self.world_size
            rank_numel = [0] * self.world_size  # 记录每个 rank 当前分配的 numel 总量
            rank_keys = [[] for _ in range(self.world_size)]  # 记录每个 rank 分配到的 keys

            sorted_keys_by_numel = sorted(keys, key=lambda k: param_numel[k],
                                          reverse=True)  # 可以选择按 numel 降序排序，或者按名字排序 (这里先用名字排序)
            # sorted_keys_by_numel = sorted(keys) # 使用名字排序，更简单

            for key in sorted_keys_by_numel:  # 遍历排序后的 keys
                best_rank = -1
                min_numel = float('inf')
                for rank_id in range(self.world_size):  # 找到当前 numel 最小的 rank
                    if rank_numel[rank_id] < min_numel:
                        min_numel = rank_numel[rank_id]
                        best_rank = rank_id
                rank_keys[best_rank].append(key)  # 将 key 分配给 numel 最小的 rank
                rank_numel[best_rank] += param_numel[key]  # 更新该 rank 的 numel 总量

            local_keys = rank_keys[self.rank]  # 获取当前 rank 的 keys
            self.ema_state = {k: full_state[k].detach().clone().to(self.device) for k in local_keys}

            # 打印每个 rank 分配到的 numel 总量，用于调试和观察
            print(f"Rank {self.rank} allocated numel: {rank_numel[self.rank]:.2e}, keys: {len(local_keys)}")
            if self.rank == 0:
                total_allocated_numel = sum(rank_numel)
                print(f"Total allocated numel: {total_allocated_numel:.2e}, target: {total_numel:.2e}")


        else:
            # Non-distributed mode: store the full state.
            self.ema_state = {k: v.detach().clone().to(self.device) for k, v in full_state.items()}
    @torch.no_grad()
    def update(self, model, decay=None):
        """Update the EMA parameters: ema = decay * ema + (1 - decay) * param."""
        decay = decay if decay is not None else self.decay
        state = model.state_dict()
        for k in self.ema_state.keys():
            self.ema_state[k].mul_(decay).add_(state[k].detach().to(self.device), alpha=1 - decay)

    def gather(self):
        """
        Gather EMA states from all processes (only applies in distributed mode).
        :return: A dictionary with the complete EMA state.
        """
        if not self.distributed or self.world_size == 1:
            return self.ema_state

        gathered = [None for _ in range(self.world_size)]
        dist.all_gather_object(gathered, self.ema_state)
        merged = {}
        for part in gathered:
            merged.update(part)
        return merged

    def state_dict(self):
        """Return the complete EMA state dictionary."""
        return self.gather()

    def load_state_dict(self, state_dict):
        """
        Load the EMA state from a given dictionary.
        This will correctly distribute parameters based on the current mode.

        :param state_dict: The full EMA state dictionary (gathered from all ranks).
        """
        if self.distributed and self.world_size > 1:
            keys = sorted(state_dict.keys())
            total = len(keys)
            per_rank = total // self.world_size
            remainder = total % self.world_size

            start = 0
            for r in range(self.rank):
                start += per_rank + (1 if r < remainder else 0)
            count = per_rank + (1 if self.rank < remainder else 0)
            local_keys = keys[start: start + count]

            self.ema_state = {k: state_dict[k].to(self.device) for k in local_keys}
        else:
            self.ema_state = {k: v.to(self.device) for k, v in state_dict.items()}

    def set_mode(self, distributed, model=None):
        """
        Switch between distributed and non-distributed modes.
        This will reinitialize the EMA state from the provided model if given.

        :param distributed: Boolean flag for distributed mode.
        :param model: The model to reinitialize the EMA state from (if provided).
        """
        self.distributed = distributed
        if self.distributed and dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

        if model:
            self._init_state(model)

    def parameters(self):
        """
        Return an iterator over EMA parameters (like a model's .parameters()).
        """
        return (param for param in self.ema_state.values())