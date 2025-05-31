import contextlib
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Sampler
import math
from torch.optim.lr_scheduler import LambdaLR



from transformers.configuration_utils import PretrainedConfig
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor
from transformers.hyperparameter_search import ALL_HYPERPARAMETER_SEARCH_BACKENDS, default_hp_search_backend
from transformers.integrations import hp_params
from transformers.integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint, is_deepspeed_available
from transformers.integrations.tpu import tpu_spmd_dataloader
from transformers.modelcard import TrainingSummary
from transformers.modeling_utils import PreTrainedModel, load_sharded_checkpoint

from transformers.trainer import DEFAULT_CALLBACKS, TRAINER_STATE_NAME

from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    ExportableState,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.trainer_pt_utils import (
    DistributedTensorGatherer,
    EvalLoopContainer,
    IterableDatasetShard,
    LabelSmoother,
    LayerWiseDummyOptimizer,
    LengthGroupedSampler,
    SequentialDistributedSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    find_batch_size,
    get_model_param_count,
    get_module_class_from_name,
    get_parameter_names,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_xla_mesh_reduce,
    reissue_pt_warnings,
    remove_dummy_checkpoint,
)
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalLoopOutput,
    EvalPrediction,
    HPSearchBackend,
    HubStrategy,
    IntervalStrategy,
    PredictionOutput,
    RemoveColumnsCollator,
    TrainerMemoryTracker,
    TrainOutput,
    check_target_module_exists,
    default_compute_objective,
    denumpify_detensorize,
    enable_full_determinism,
    find_executable_batch_size,
    get_last_checkpoint,
    has_length,
    neftune_post_forward_hook,
    number_of_arguments,
    seed_worker,
    set_seed,
    speed_metrics,
)
from transformers.training_args import OptimizerNames, ParallelMode, TrainingArguments
from transformers.utils import (
    ADAPTER_CONFIG_NAME,
    ADAPTER_SAFE_WEIGHTS_NAME,
    ADAPTER_WEIGHTS_NAME,
    CONFIG_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    XLA_FSDPV2_MIN_VERSION,
    PushInProgress,
    PushToHubMixin,
    can_return_loss,
    find_labels,
    is_accelerate_available,
    is_apex_available,
    is_bitsandbytes_available,
    is_datasets_available,
    is_galore_torch_available,
    is_in_notebook,
    is_ipex_available,
    is_lomo_available,
    is_peft_available,
    is_safetensors_available,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_compile_available,
    is_torch_mlu_available,
    is_torch_neuroncore_available,
    is_torch_npu_available,
    is_torch_xla_available,
    logging,
    strtobool,
)
from transformers.utils.quantization_config import QuantizationMethod

from torch.utils.data import DataLoader, Dataset, IterableDataset, RandomSampler, SequentialSampler
from packaging import version

from transformers import __version__

from transformers.trainer import _is_peft_model
from transformers.trainer import _get_fsdp_ckpt_kwargs


if is_accelerate_available():
    from accelerate import Accelerator, skip_first_batches
    from accelerate import __version__ as accelerate_version
    from accelerate.utils import (
        DistributedDataParallelKwargs,
        DistributedType,
        GradientAccumulationPlugin,
        is_mlu_available,
        is_mps_available,
        is_npu_available,
        is_torch_version,
        is_xpu_available,
        load_fsdp_model,
        load_fsdp_optimizer,
        save_fsdp_model,
        save_fsdp_optimizer,
    )

    DATA_SAMPLERS = [RandomSampler]
    if version.parse(accelerate_version) > version.parse("0.23.0"):
        from accelerate.data_loader import SeedableRandomSampler

        DATA_SAMPLERS += [SeedableRandomSampler]

    if is_deepspeed_available():
        from accelerate.utils import DeepSpeedSchedulerWrapper

if is_accelerate_available("0.28.0"):
    from accelerate.utils import DataLoaderConfiguration



# from .base_trainer import BaseTrainer as Trainer
from transformers.trainer import (
    Trainer,
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    logger,
    is_torch_xla_available,
    is_apex_available,
)

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

if is_apex_available():
    from apex import amp

if is_sagemaker_mp_enabled():
    from transformers.trainer_pt_utils import smp_forward_backward

from typing import List, Optional, Dict, Union, Any

from torch.utils.data import DataLoader
from transformers.utils import is_datasets_available
from transformers.trainer_utils import seed_worker
if is_datasets_available():
    import datasets

from ..dist_utils import synchronize
from ..utils import rank0_print
from collections import defaultdict


def safe_globals():
    # Starting from version 2.4 PyTorch introduces a check for the objects loaded
    # with torch.load(weights_only=True). Starting from 2.6 weights_only=True becomes
    # a default and requires allowlisting of objects being loaded.
    # See: https://github.com/pytorch/pytorch/pull/137602
    # See: https://pytorch.org/docs/stable/notes/serialization.html#torch.serialization.add_safe_globals
    # See: https://github.com/huggingface/accelerate/pull/3036
    if version.parse(torch.__version__).release < version.parse("2.6").release:
        return contextlib.nullcontext()

    np_core = np._core if version.parse(np.__version__) >= version.parse("2.0.0") else np.core
    allowlist = [np_core.multiarray._reconstruct, np.ndarray, np.dtype]
    # numpy >1.25 defines numpy.dtypes.UInt32DType, but below works for
    # all versions of numpy
    allowlist += [type(np.dtype(np.uint32))]

    return torch.serialization.safe_globals(allowlist)


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


class ILLUMETrainer(Trainer):

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if self.args.mm_projector_lr is not None:
                projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name]
                if self.args.mm_vision_tower_lr is not None:
                    vision_tower_parameters = [
                        name for name, _ in opt_model.named_parameters() if "vision_tower" in name]
                    optimizer_grouped_parameters = [
                        {
                            "params": [
                                p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in projector_parameters and n not in vision_tower_parameters and p.requires_grad)
                            ],
                            "weight_decay": self.args.weight_decay,
                        },
                        {
                            "params": [
                                p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in projector_parameters and n in vision_tower_parameters and p.requires_grad)
                            ],
                            "weight_decay": self.args.weight_decay,
                            "lr": self.args.mm_vision_tower_lr,
                        },
                        {
                            "params": [
                                p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in projector_parameters and n not in vision_tower_parameters and p.requires_grad)
                            ],
                            "weight_decay": 0.0,
                        },
                        {
                            "params": [
                                p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in projector_parameters and n in vision_tower_parameters and p.requires_grad)
                            ],
                            "weight_decay": 0.0,
                            "lr": self.args.mm_vision_tower_lr,
                        },
                        {
                            "params": [
                                p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                            ],
                            "weight_decay": self.args.weight_decay,
                            "lr": self.args.mm_projector_lr,
                        },
                        {
                            "params": [
                                p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                            ],
                            "weight_decay": 0.0,
                            "lr": self.args.mm_projector_lr,
                        },
                    ]
                else:
                    optimizer_grouped_parameters = [
                        {
                            "params": [
                                p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in projector_parameters and p.requires_grad)
                            ],
                            "weight_decay": self.args.weight_decay,
                        },
                        {
                            "params": [
                                p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in projector_parameters and p.requires_grad)
                            ],
                            "weight_decay": 0.0,
                        },
                        {
                            "params": [
                                p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                            ],
                            "weight_decay": self.args.weight_decay,
                            "lr": self.args.mm_projector_lr,
                        },
                        {
                            "params": [
                                p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                            ],
                            "weight_decay": 0.0,
                            "lr": self.args.mm_projector_lr,
                        },
                    ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def _save_checkpoint(self, model, trial, metrics=None):
        super(ILLUMETrainer, self)._save_checkpoint(model, trial, metrics)

        # Save all trainable states. i.e., saving the params in vision encoder in pretrain stage.
        from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)

        # weight_to_save = get_trainable_state_maybe_zero_3(self.model.named_parameters())

        if self.args.local_rank == 0 or self.args.local_rank == -1:
            os.makedirs(output_dir, exist_ok=True)
            # self.model.config.save_pretrained(output_dir)
            # torch.save(weight_to_save, os.path.join(output_dir, f'trainables.bin'))

            # all ranks need to upload their rng files.
            try:
                import moxing as mox
                if hasattr(self.args, 'remote_work_dir'):
                    remote_checkpoint_dir=os.path.join(self.args.remote_work_dir, checkpoint_folder)
                    mox.file.copy_parallel(output_dir,
                                           remote_checkpoint_dir)
                    rank0_print(f"Moxing: copy cehckpoint dir from {output_dir} to {remote_checkpoint_dir}.")
            except Exception as e:
                rank0_print(f"Checkpoint saving in {output_dir}")

        def get_free_space(path):
            if not os.path.exists(path):
                print(f"{path} dose not exist.")
                return 0
            info = os.statvfs(path)
            free_size = info.f_bsize * info.f_bavail / 1024 ** 3  # GB
            return free_size

        synchronize()
        free_space = get_free_space("/cache")
        rank0_print(f'Free space:{free_space:.2f} GB)')
        # if free_space < 380:
        #     raise RuntimeError

        torch.cuda.empty_cache()

    def _load_rng_state(self, checkpoint):
        # Load RNG states from `checkpoint`
        if checkpoint is None:
            return

        if self.args.world_size > 1:
            process_index = self.args.process_index
            rng_file = os.path.join(checkpoint, f"rng_state_{process_index}.pth")
            if not os.path.isfile(rng_file):
                logger.info(
                    f"Didn't find an RNG file for process {process_index}, if you are resuming a training that "
                    "wasn't launched in a distributed fashion, reproducibility is not guaranteed."
                )
                return
        else:
            rng_file = os.path.join(checkpoint, "rng_state.pth")
            if not os.path.isfile(rng_file):
                logger.info(
                    "Didn't find an RNG file, if you are resuming a training that was launched in a distributed "
                    "fashion, reproducibility is not guaranteed."
                )
                return

        with safe_globals():
            checkpoint_rng_state = torch.load(rng_file)
        random.setstate(checkpoint_rng_state["python"])
        np.random.set_state(checkpoint_rng_state["numpy"])
        torch.random.set_rng_state(checkpoint_rng_state["cpu"])
        if torch.cuda.is_available():
            if self.args.parallel_mode == ParallelMode.DISTRIBUTED:
                torch.cuda.random.set_rng_state_all(checkpoint_rng_state["cuda"])
            else:
                try:
                    torch.cuda.random.set_rng_state(checkpoint_rng_state["cuda"])
                except Exception as e:
                    logger.info(
                        f"Didn't manage to set back the RNG states of the GPU because of the following error:\n {e}"
                        "\nThis won't yield the same results as if the training had not been interrupted."
                    )
        if is_torch_xla_available():
            xm.set_rng_state(checkpoint_rng_state["xla"])
        if is_torch_npu_available():
            if self.args.parallel_mode == ParallelMode.DISTRIBUTED:
                torch.npu.random.set_rng_state_all(checkpoint_rng_state["npu"])
            else:
                try:
                    torch.npu.random.set_rng_state(checkpoint_rng_state["npu"])
                except Exception as e:
                    logger.info(
                        f"Didn't manage to set back the RNG states of the NPU because of the following error:\n {e}"
                        "\nThis won't yield the same results as if the training had not been interrupted."
                    )
        if is_torch_mlu_available():
            if self.args.parallel_mode == ParallelMode.DISTRIBUTED:
                torch.mlu.random.set_rng_state_all(checkpoint_rng_state["mlu"])
            else:
                try:
                    torch.mlu.random.set_rng_state(checkpoint_rng_state["mlu"])
                except Exception as e:
                    logger.info(
                        f"Didn't manage to set back the RNG states of the MLU because of the following error:\n {e}"
                        "\nThis won't yield the same results as if the training had not been interrupted."
                    )

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        super(ILLUMETrainer, self)._save(output_dir, state_dict)
