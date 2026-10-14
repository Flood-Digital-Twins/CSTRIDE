from .runtime import ROOT, load_config, parse_args, load_state_dict
import argparse
import os
import random
from time import time

import numpy as np
import torch
import torch.distributed as dist
from .tracking import wandb
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler

from .soap import SOAP
from .chicago import (
    ChicagoTrajectoryDataset,
    add_chicago_input_noise,
    apply_chicago_sensor_dropout,
    build_chicago_sensor_map,
    build_sensor_sequences,
    chicago_include_precipitation_input,
    chicago_split_indices,
    chicago_use_terrain_conditioning,
    compute_chicago_snapshot_weights,
    compute_chicago_spatial_probs,
    fit_chicago_normalizers,
    load_chicago_manifest,
    load_chicago_sensors,
    extend_chicago_sensor_map,
    make_windows,
    get_chicago_output_size,
    resolve_chicago_observables,
    resolve_chicago_targets,
    save_chicago_sensors,
)
from .models import CSTRIDE
from .train import save_training_results, train_model


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_worker_init_fn(base_seed, rank):
    def _seed_worker(worker_id):
        worker_seed = int(base_seed + rank * 1000 + worker_id)
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return _seed_worker


def _load_initial_checkpoint(model, checkpoint_path, device):
    state_dict = load_state_dict(checkpoint_path, device)

    # Be tolerant to checkpoints saved through DDP and/or torch.compile wrappers.
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k[len("_orig_mod."):]: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    print(f"Initialized model weights from checkpoint: {checkpoint_path}")


def _resolve_run_location(default_save_path, init_checkpoint):
    if init_checkpoint is None:
        return default_save_path, None, False

    ckpt_path = os.path.abspath(init_checkpoint)
    run_dir = os.path.dirname(ckpt_path)
    run_name = os.path.basename(run_dir)
    save_path = os.path.dirname(run_dir)
    return save_path, run_name, True


def _filter_ranks_by_memory(device, min_free_gb=4.0):
    global_rank = dist.get_rank()
    global_world = dist.get_world_size()

    free_bytes, _ = torch.cuda.mem_get_info(device)
    free_gb = free_bytes / (1024 ** 3)
    is_valid = free_gb >= min_free_gb

    flag = torch.tensor([int(is_valid)], dtype=torch.int32, device=device)
    gathered = [torch.zeros(1, dtype=torch.int32, device=device) for _ in range(global_world)]
    dist.all_gather(gathered, flag)
    valid_ranks = [r for r, v in enumerate(gathered) if v.item() == 1]

    if not valid_ranks:
        raise RuntimeError("No GPU has sufficient free memory. Aborting.")

    valid_group = dist.new_group(valid_ranks)
    if global_rank not in valid_ranks:
        print(
            f"[DDP] rank {global_rank} ({device}): only {free_gb:.1f} GB free "
            f"(< {min_free_gb:.1f} GB required) - skipping."
        )
        dist.destroy_process_group()
        raise SystemExit(0)

    valid_rank = valid_ranks.index(global_rank)
    valid_world = len(valid_ranks)
    return valid_group, valid_world, valid_rank, valid_ranks


def main(config, args):
    is_ddp = args.ddp
    if is_ddp:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        min_free_gb = config["training"].get("min_free_memory_gb", 40.0)
        valid_group, world_size, rank, valid_ranks = _filter_ranks_by_memory(device, min_free_gb)
        is_main = rank == 0
    else:
        device = torch.device(config["basics"]["device"] if torch.cuda.is_available() else "cpu")
        valid_group = None
        world_size = 1
        rank = 0
        valid_ranks = [0]
        is_main = True

    base_seed = int(config["basics"]["random_seed"])
    _seed_everything(base_seed)
    rank_seed = base_seed + rank
    worker_init_fn = _make_worker_init_fn(base_seed, rank)

    if config["basics"].get("scaler", "global") != "global":
        raise ValueError("Chicago scripts currently support only global scaling because trajectories use different grids.")

    root, manifest, entries = load_chicago_manifest(config["basics"].get("data_root", "data/Chicago"))
    ntrajectories = len(entries)
    ntimes = int(manifest["time_steps"])
    dataset_output_size = get_chicago_output_size(root, entries[0])
    observable_indices = resolve_chicago_observables(config, dataset_output_size)
    target_indices = resolve_chicago_targets(config, dataset_output_size)
    output_size = len(target_indices)
    num_sensors = int(config["basics"].get("num_sensors", 0))
    include_precipitation = chicago_include_precipitation_input(config)
    terrain_conditioning = chicago_use_terrain_conditioning(config)
    lag = config["basics"]["lag"]
    jerk_regularization = float(config["training"].get("jerk_regularization", 0.0))
    second_order_regularization = float(config["training"].get("second_order_regularization", 0.0))
    if second_order_regularization > 0.0 and lag < 3:
        raise ValueError(f"Second-order regularization requires lag >= 3, got lag={lag}.")
    if jerk_regularization > 0.0 and lag < 4:
        raise ValueError(f"Jerk regularization requires lag >= 4, got lag={lag}.")
    if num_sensors == 0 and not include_precipitation:
        raise ValueError(
            "Chicago training requires at least one temporal input feature. "
            "Set num_sensors > 0 or include_precipitation_input: true."
        )
    nt_valid = ntimes - lag
    coord_size = 5 if terrain_conditioning else 2

    idx_train, idx_valid, _ = chicago_split_indices(entries, config["basics"].get("split_file"))
    normalizers = fit_chicago_normalizers(entries, idx_train, root)

    sensor_map = load_chicago_sensors("", config)
    if sensor_map is None:
        sensor_map = build_chicago_sensor_map(entries, idx_train, root, config)
    else:
        sensor_map = extend_chicago_sensor_map(entries, root, sensor_map)

    sensor_sequences = build_sensor_sequences(
        entries,
        root,
        sensor_map,
        normalizers,
        observable_indices,
        include_precipitation=include_precipitation,
    )
    sensors_data = torch.stack(sensor_sequences, dim=0)
    sensor_feature_count = num_sensors * len(observable_indices)
    if config["basics"]["noise"] > 0.0:
        sensors_data = add_chicago_input_noise(
            sensors_data,
            config["basics"]["noise"],
            sensor_feature_count=sensor_feature_count,
        )
    sensors_data, sensor_dropout_info = apply_chicago_sensor_dropout(
        sensors_data,
        config,
        num_sensors=num_sensors,
        observable_count=len(observable_indices),
        sensor_map=sensor_map,
        entries=entries,
        root=root,
    )

    if is_main:
        print(
            f"Loaded Chicago dataset with {ntrajectories} trajectories, {ntimes} time steps, "
            f"dataset_output_size={dataset_output_size}, target_indices={target_indices}, "
            f"output_size={output_size}, num_sensors={num_sensors}, observables={observable_indices}, "
            f"include_precipitation_input={include_precipitation}, "
            f"terrain_conditioning={terrain_conditioning}, "
            f"sensor_dropout={sensor_dropout_info}, "
            f"second_order_regularization={second_order_regularization}, "
            f"jerk_regularization={jerk_regularization}"
        )

    train_windows = make_windows(sensors_data[idx_train], lag)
    valid_windows = make_windows(sensors_data[idx_valid], lag)

    spatial_probs_train = {}
    spatial_probs_valid = {}
    if config["training"]["num_points"] is not None and args.importance_sampling:
        spatial_probs_train = compute_chicago_spatial_probs(entries, idx_train, root, lag)
        spatial_probs_valid = compute_chicago_spatial_probs(entries, idx_valid, root, lag)

    train_dataset = ChicagoTrajectoryDataset(
        entries=[entries[i] for i in idx_train],
        root=root,
        input_windows=train_windows,
        nt=nt_valid,
        lag=lag,
        normalizers=normalizers,
        target_indices=target_indices,
        terrain_conditioning=terrain_conditioning,
        num_points=config["training"]["num_points"],
        spatial_probs=spatial_probs_train,
    )
    valid_dataset = ChicagoTrajectoryDataset(
        entries=[entries[i] for i in idx_valid],
        root=root,
        input_windows=valid_windows,
        nt=nt_valid,
        lag=lag,
        normalizers=normalizers,
        target_indices=target_indices,
        terrain_conditioning=terrain_conditioning,
        num_points=config["training"]["num_points"],
        spatial_probs=spatial_probs_valid,
    )

    snapshot_weights = None
    if args.importance_sampling:
        snapshot_weights = compute_chicago_snapshot_weights(entries, idx_train, root, lag)

    train_sampler_generator = torch.Generator()
    train_sampler_generator.manual_seed(rank_seed)
    data_loader_generator = torch.Generator()
    data_loader_generator.manual_seed(rank_seed)

    if is_ddp:
        if snapshot_weights is not None:
            train_sampler = WeightedRandomSampler(
                weights=snapshot_weights,
                num_samples=len(train_dataset),
                replacement=True,
                generator=train_sampler_generator,
            )
        else:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=base_seed,
            )
        valid_sampler = DistributedSampler(
            valid_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            seed=base_seed,
        )
    else:
        if snapshot_weights is not None:
            train_sampler = WeightedRandomSampler(
                weights=snapshot_weights,
                num_samples=len(train_dataset),
                replacement=True,
                generator=train_sampler_generator,
            )
        else:
            train_sampler = None
        valid_sampler = None

    batch_size = config["training"]["batch_size"]
    if config["training"]["num_points"] is None and batch_size > 1:
        print("Chicago data uses variable grids; forcing batch_size=1 when num_points is None.")
        batch_size = 1

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=config["training"].get("num_workers", 8),
        pin_memory=True,
        persistent_workers=config["training"].get("num_workers", 8) > 0,
        worker_init_fn=worker_init_fn,
        generator=data_loader_generator,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        sampler=valid_sampler,
        shuffle=False,
        num_workers=config["training"].get("num_workers", 8),
        pin_memory=True,
        persistent_workers=config["training"].get("num_workers", 8) > 0,
        worker_init_fn=worker_init_fn,
        generator=data_loader_generator,
    )

    model = CSTRIDE(
        input_size=sensors_data.shape[-1],
        hidden_size=config["model"]["hidden_size"],
        hidden_layers=config["model"]["hidden_layers"],
        fmmnn_depth=config["model"]["fmmnn_depth"],
        fmmnn_width=config["model"]["fmmnn_width"],
        fmmnn_rank=config["model"]["fmmnn_rank"],
        coord_size=coord_size,
        output_size=output_size,
        num_fourier=config["model"]["num_fourier"],
        device=device,
    ).to(device)

    init_checkpoint = args.init_checkpoint or config["training"].get("init_checkpoint")
    if init_checkpoint is not None:
        _load_initial_checkpoint(model, init_checkpoint, device)

    save_path, resume_run_name, is_resuming = _resolve_run_location(
        config["training"]["save_path"], init_checkpoint
    )

    if is_ddp:
        local_rank = int(os.environ["LOCAL_RANK"])
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, process_group=valid_group)
    model = torch.compile(model) if config["training"].get("compile", False) else model

    criterion = torch.nn.MSELoss(reduction="mean")
    optimizer = SOAP(
        params=model.parameters(),
        lr=config["training"]["learning_rate"],
        betas=(0.95, 0.95),
        weight_decay=0.01,
        precondition_frequency=10,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config["training"]["gamma"],
        patience=4,
        threshold=0.01,
        threshold_mode="rel",
        cooldown=0,
        min_lr=1e-6,
        eps=1e-8,
    )

    if is_resuming:
        run_name = resume_run_name
        if is_main:
            print(f"Resuming into existing run directory: {os.path.join(save_path, run_name)}")
    elif is_ddp:
        ts = torch.tensor([int(time())], dtype=torch.int64, device=device)
        dist.broadcast(ts, src=valid_ranks[0], group=valid_group)
        run_name = f"training_{ts.item()}"
    else:
        run_name = f"training_{int(time())}"

    if is_main:
        os.makedirs(os.path.join(save_path, run_name), exist_ok=True)
        with open(os.path.join(save_path, run_name, "config.yaml"), "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        save_chicago_sensors(save_path, run_name, sensor_map)

        wandb.init(
            
            project=config.get("wandb", {}).get("project", "C-STRIDE"),
            config=config,
            name=run_name,
        )
        wandb.watch(model, log="gradients", log_freq=500)
        log_callback = wandb.log
    else:
        log_callback = None

    torch.manual_seed(rank_seed)
    np.random.seed(rank_seed)
    random.seed(rank_seed)

    results = train_model(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        epochs=config["training"]["epochs"],
        log_interval=config["training"]["log_interval"],
        eval_interval=10,
        patience=config["training"]["patience"],
        threshold=5e-3,
        save_dir=save_path if is_main else None,
        run_name=run_name if is_main else None,
        log_callback=log_callback,
        train_sampler=train_sampler if isinstance(train_sampler, DistributedSampler) else None,
        jerk_regularization=jerk_regularization,
        second_order_regularization=second_order_regularization,
    )

    if is_main:
        save_training_results(save_path, run_name, results["training_time"])

    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    torch.set_default_device("cpu")
    parser = argparse.ArgumentParser(description="Train C-STRIDE on the Chicago dataset")
    parser.add_argument("--config", default=str(ROOT / "configs/next_step/full.yaml"))
    parser.add_argument("--ddp", action="store_true", default=False)
    parser.add_argument("--importance-sampling", dest="importance_sampling", action="store_true", default=False)
    parser.add_argument(
        "--init-checkpoint",
        default=None,
        help="Optional path to model weights used to initialize training before starting a new run.",
    )
    args = parse_args(parser)
    with open(args.config, "r") as f:
        config = load_config(f)

    config.setdefault("training", {})
    if args.init_checkpoint is not None:
        config["training"]["init_checkpoint"] = args.init_checkpoint

    main(config, args)
