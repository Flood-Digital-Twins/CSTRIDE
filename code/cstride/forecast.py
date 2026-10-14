import torch

from .processdata import Padding


def split_single_trajectory_contiguous(ntimes, train_ratio=0.8):
    """
    Contiguous split for single-trajectory forecasting.

    Random time splits leak future information across autoregressive windows.
    """
    ntrain = round(ntimes * train_ratio)
    remaining = ntimes - ntrain
    nvalid = remaining // 2
    idx_train = torch.arange(0, ntrain, dtype=torch.long)
    idx_valid = torch.arange(ntrain, ntrain + nvalid, dtype=torch.long)
    idx_test = torch.arange(ntrain + nvalid, ntimes, dtype=torch.long)
    return idx_train.numpy(), idx_valid.numpy(), idx_test.numpy()


def build_forecast_window_tensors(
    sensors_data,
    coords_data,
    output_data,
    sensor_coords_data,
    lag,
    horizon,
    include_lookback_targets=False,
):
    """
    Construct aligned forecast windows from per-trajectory sequences.
    """
    if sensors_data.dim() != 3:
        raise ValueError("sensors_data must have shape (ntraj, nt, flattened_sensor_dim)")

    ntrajectories, ntimes, _ = sensors_data.shape
    valid_times = ntimes - horizon
    if valid_times <= 0:
        raise ValueError(
            f"Forecast horizon {horizon} is too large for sequence length {ntimes}."
        )

    histories = Padding(sensors_data, lag).reshape(ntrajectories, ntimes, lag, -1)
    histories = histories[:, :valid_times]

    coords_seq = torch.stack(
        [coords_data[:, step : step + valid_times] for step in range(1, horizon + 1)],
        dim=2,
    )
    output_seq = torch.stack(
        [output_data[:, step : step + valid_times] for step in range(1, horizon + 1)],
        dim=2,
    )
    sensor_coords_seq = torch.stack(
        [sensor_coords_data[:, step : step + valid_times] for step in range(1, horizon + 1)],
        dim=2,
    )

    tensors = (
        histories.reshape(-1, lag, histories.shape[-1]),
        coords_seq.reshape(-1, horizon, coords_data.shape[-2], coords_data.shape[-1]),
        output_seq.reshape(-1, horizon, output_data.shape[-2], output_data.shape[-1]),
        sensor_coords_seq.reshape(
            -1,
            horizon,
            sensor_coords_data.shape[-2],
            sensor_coords_data.shape[-1],
        ),
    )
    if not include_lookback_targets:
        return tensors

    max_lookback_steps = lag - 1
    if max_lookback_steps <= 0:
        raise ValueError("include_lookback_targets requires lag >= 2.")

    current_times = torch.arange(valid_times, device=coords_data.device).view(1, valid_times)
    current_times = current_times.expand(ntrajectories, -1)

    lookback_coords = []
    lookback_outputs = []
    lookback_valid_masks = []
    for step in range(1, max_lookback_steps + 1):
        target_times = current_times - (step - 1)
        valid_mask = current_times >= step
        target_times = target_times.clamp(min=0)

        traj_indices = torch.arange(ntrajectories, device=coords_data.device).view(-1, 1)
        step_coords = coords_data[traj_indices, target_times]
        step_outputs = output_data[traj_indices, target_times]

        lookback_coords.append(step_coords)
        lookback_outputs.append(step_outputs)
        lookback_valid_masks.append(valid_mask)

    lookback_coords = torch.stack(lookback_coords, dim=2)
    lookback_outputs = torch.stack(lookback_outputs, dim=2)
    lookback_valid_masks = torch.stack(lookback_valid_masks, dim=2)
    return tensors + (
        lookback_coords.reshape(-1, max_lookback_steps, coords_data.shape[-2], coords_data.shape[-1]),
        lookback_outputs.reshape(-1, max_lookback_steps, output_data.shape[-2], output_data.shape[-1]),
        lookback_valid_masks.reshape(-1, max_lookback_steps),
    )


def autoregressive_forecast(
    model,
    batch_x,
    batch_coords,
    future_sensor_coords,
    observables,
    future_exogenous=None,
):
    """
    Roll the one-step model forward over multiple future steps.
    """
    history = batch_x
    predictions = []

    horizon = batch_coords.shape[1]
    for step in range(horizon):
        prediction, latent = model(history, batch_coords[:, step])
        predictions.append(prediction)

        pred_sensors = model.decode(future_sensor_coords[:, step], latent)
        if observables is not None:
            pred_sensors = pred_sensors[..., observables]
        pred_sensors = pred_sensors.reshape(batch_x.shape[0], -1)
        if future_exogenous is not None:
            pred_sensors = torch.cat((pred_sensors, future_exogenous[:, step]), dim=-1)
        history = torch.cat((history[:, 1:], pred_sensors.unsqueeze(1)), dim=1)

    return torch.stack(predictions, dim=1)


def discounted_forecast_loss(predictions, targets, discount_factor, criterion):
    """
    Weighted multi-step loss with geometric discounting.
    """
    horizon = predictions.shape[1]
    weights = torch.pow(
        torch.full((horizon,), discount_factor, device=predictions.device),
        torch.arange(horizon, device=predictions.device, dtype=predictions.dtype),
    )
    weights = weights / weights.sum()

    total_loss = predictions.new_tensor(0.0)
    for step in range(horizon):
        total_loss = total_loss + weights[step] * criterion(
            predictions[:, step],
            targets[:, step],
        )
    return total_loss


def backward_history_matching_loss(
    model,
    batch_x,
    lookback_coords,
    lookback_target,
    lookback_valid_mask,
    lookback_steps,
    discount_factor,
    criterion,
):
    """
    Auxiliary loss that asks earlier history prefixes to decode the last observed state.
    """
    max_supported_steps = min(
        int(lookback_steps),
        batch_x.shape[1] - 1,
        lookback_target.shape[1],
    )
    if max_supported_steps <= 0:
        return batch_x.new_tensor(0.0)

    weights = torch.pow(
        torch.full((max_supported_steps,), discount_factor, device=batch_x.device),
        torch.arange(max_supported_steps, device=batch_x.device, dtype=batch_x.dtype),
    )

    total_loss = batch_x.new_tensor(0.0)
    has_valid_step = False
    for step in range(1, max_supported_steps + 1):
        valid_mask = lookback_valid_mask[:, step - 1]
        if not torch.any(valid_mask):
            continue

        has_valid_step = True
        prediction, _ = model(batch_x[valid_mask, :-step], lookback_coords[valid_mask, step - 1])
        step_loss = criterion(prediction, lookback_target[valid_mask, step - 1])
        weight = weights[step - 1]
        total_loss = total_loss + weight * step_loss

    if not has_valid_step:
        return batch_x.new_tensor(0.0)
    return total_loss
