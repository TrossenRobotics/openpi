import dataclasses
import functools
import logging
from logging import config
import platform
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders
import os
import sys

def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_huggingface(config: _config.TrainConfig):
    """Initialize Hugging Face authentication if token is provided."""
    # Get token from config or environment variable
    token = config.huggingface_token
    if token is None:
        token = os.getenv("HUGGINGFACE_TOKEN")
        print(f"Token was retrieved from environment variable")
    
    if token is not None:
        try:
            from huggingface_hub import login
            # Log in programmatically
            login(token=token, add_to_git_credential=False)
            logging.info("Successfully logged in to Hugging Face")
            print(f"Logged in successfully")
        except ImportError:
            print(f"huggingface_hub not available, skipping Hugging Face login")
            logging.warning("huggingface_hub not available, skipping Hugging Face login")
        except Exception as e:
            print(f"Failed to login to Hugging Face: {e}")
            logging.warning(f"Failed to login to Hugging Face: {e}")


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    os.makedirs("/opt/ml/output/wandb", exist_ok=True)
    os.environ["WANDB_DIR"] = "/opt/ml/output/wandb"
    # Login with API key if provided (for environments like SageMaker)
    api_key = config.wandb_api_key
    if api_key is None:
        # Try to get API key from environment variable
        api_key = os.getenv("WANDB_API_KEY")
    
    if api_key is not None:
        wandb.login(key=api_key)

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_huggingface(config)
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # Log images from first batch to sanity check.
    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
        disable=not sys.stdout.isatty(),
        file=sys.stdout,
    )

    infos = []
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            logging.info(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            logging.info(f"Logged training metrics to WandB at step {step}")
            logging.info(f"reduced_info: {reduced_info}")
            infos = []
        batch = next(data_iter)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            # Log checkpoint saving.
            logging.info(f"Saving checkpoint at step {step}")
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


def filter_sagemaker_args(argv):
    """
    Filter out SageMaker-specific arguments while preserving legitimate training arguments.
    
    SageMaker often adds extra arguments that can interfere with tyro's CLI parsing.
    This function identifies and removes those unwanted arguments while preserving
    all legitimate training arguments.
    """
    import re
    
    # Common SageMaker arguments to filter out (both --key and --key=value formats)
    sagemaker_patterns = [
        r'^--model-dir(=.*)?$',           # SageMaker model directory
        r'^--model_dir(=.*)?$', 
        r'^--sm-model-dir(=.*)?$',        # SageMaker model directory (alternate)
        r'^--output-data-dir(=.*)?$',     # SageMaker output directory
        r'^--output_data_dir(=.*)?$',
        r'^--channel-.*$',                # SageMaker input channels
        r'^--training(=.*)?$',            # SageMaker training flag
        r'^--hosts(=.*)?$',               # SageMaker distributed training
        r'^--current-host(=.*)?$',        # SageMaker current host
        r'^--current_host(=.*)?$',
        r'^--num-gpus(=.*)?$',            # SageMaker GPU count
        r'^--num_gpus(=.*)?$',
        r'^--num-cpus(=.*)?$',            # SageMaker CPU count  
        r'^--num_cpus(=.*)?$',
        r'^--instance-groups(=.*)?$',     # SageMaker instance groups
        r'^--instance_groups(=.*)?$',
        r'^--network-interface-name(=.*)?$',  # SageMaker network interface
        r'^--network_interface_name(=.*)?$',
    ]
    
    # Known legitimate training arguments (both --key and --key=value formats)
    training_arg_patterns = [
        r'^--exp-name(=.*)?$',
        r'^--exp_name(=.*)?$',
        r'^--overwrite$',
        r'^--resume$', 
        r'^--wandb-enabled(=.*)?$',
        r'^--wandb_enabled(=.*)?$',
        r'^--wandb-api-key(=.*)?$',
        r'^--wandb_api_key(=.*)?$',
        r'^--huggingface-token(=.*)?$',
        r'^--huggingface_token(=.*)?$',
        r'^--batch-size(=.*)?$',
        r'^--batch_size(=.*)?$',
        r'^--num-train-steps(=.*)?$',
        r'^--num_train_steps(=.*)?$',
        r'^--log-interval(=.*)?$',
        r'^--log_interval(=.*)?$',
        r'^--save-interval(=.*)?$',
        r'^--save_interval(=.*)?$',
        r'^--seed(=.*)?$',
    ]
    
    filtered_args = []
    skip_next = False
    
    for i, arg in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
            
        # Always preserve the script name and config name (first few args)
        if i < 3:  # ['python', 'scripts/train.py', 'config_name']
            filtered_args.append(arg)
            continue
            
        # Check if this is a SageMaker argument to filter out
        is_sagemaker_arg = any(re.match(pattern, arg) for pattern in sagemaker_patterns)
        
        if is_sagemaker_arg:
            # Skip this argument and its value (only if it doesn't contain '=' and next arg is a value)
            if '=' not in arg and i + 1 < len(argv) and not argv[i + 1].startswith('--'):
                skip_next = True
            continue
            
        # Check if this is a known training argument
        is_training_arg = any(re.match(pattern, arg) for pattern in training_arg_patterns)
        
        if is_training_arg:
            # Preserve training arguments and their values
            filtered_args.append(arg)
            # If it's a separate key-value pair (not --key=value), preserve the value too
            if '=' not in arg and i + 1 < len(argv) and not argv[i + 1].startswith('--'):
                filtered_args.append(argv[i + 1])
                skip_next = True
        elif not arg.startswith('--'):
            # This is a value for a previous argument (already handled above)
            filtered_args.append(arg)
        else:
            # Unknown argument - log and discard for safety
            logging.info(f"Discarding unknown argument: {arg}")
            # Skip its value if it has one
            if i + 1 < len(argv) and not argv[i + 1].startswith('--'):
                skip_next = True
            
    return filtered_args


if __name__ == "__main__":

    
    # Log the original sys.argv for debugging
    logging.basicConfig(level=logging.INFO)
    logging.info(f"Original sys.argv: {sys.argv}")

    # Filter out SageMaker-specific arguments while preserving legitimate training arguments
    filtered_argv = filter_sagemaker_args(sys.argv)
    sys.argv = filtered_argv

    main(_config.cli())
