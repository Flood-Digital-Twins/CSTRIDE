import torch
import time
import os


def _unwrap_model(model):
    """
    Strip torch.compile (_orig_mod) and DataParallel / DDP (module) wrappers
    so that saved state dicts always use the bare model's key names.
    """
    # torch.compile wraps under _orig_mod
    if hasattr(model, '_orig_mod'):
        model = model._orig_mod
    # DataParallel and DistributedDataParallel both expose .module
    if isinstance(model, (torch.nn.DataParallel,
                          torch.nn.parallel.DistributedDataParallel)):
        model = model.module
    return model

def latent_jerk_loss(hidden_sequence):
    """
    Mean squared third-order forward finite difference of LSTM hidden states.
    hidden_sequence has shape (batch, lag, hidden_size).
    """
    if hidden_sequence.shape[1] < 4:
        raise ValueError(
            f"Jerk regularization requires lag >= 4, got lag={hidden_sequence.shape[1]}."
        )
    third_diff = (
        hidden_sequence[:, 3:]
        - 3.0 * hidden_sequence[:, 2:-1]
        + 3.0 * hidden_sequence[:, 1:-2]
        - hidden_sequence[:, :-3]
    )
    return third_diff.pow(2).mean()


def latent_second_order_loss(hidden_sequence):
    """
    Mean squared second-order forward finite difference of LSTM hidden states.
    hidden_sequence has shape (batch, lag, hidden_size).
    """
    if hidden_sequence.shape[1] < 3:
        raise ValueError(
            f"Second-order regularization requires lag >= 3, got lag={hidden_sequence.shape[1]}."
        )
    second_diff = (
        hidden_sequence[:, 2:]
        - 2.0 * hidden_sequence[:, 1:-1]
        + hidden_sequence[:, :-2]
    )
    return second_diff.pow(2).mean()


def train_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    criterion,
    device,
    epoch,
    log_interval=50,
    jerk_regularization=0.0,
    second_order_regularization=0.0,
):
    '''
    Trains the model for one epoch.
    
    Args:
        model (nn.Module): The model to train.
        dataloader (DataLoader): DataLoader for the training dataset.
        optimizer: The optimizer for updating model parameters.
        scheduler: Learning rate scheduler.
        criterion: The loss function.
        device: The device (e.g., 'cuda' or 'cpu') to run training on.
        epoch (int): Current epoch number (for logging purposes).
        log_interval (int): How many batches to wait before logging training status.
    
    Returns:
        float: The average loss over the epoch.
    '''
    model.train() # Set model to training mode
    total_loss = 0
    total_prediction_loss = 0
    total_jerk_loss = 0
    total_second_order_loss = 0
    start_time = time.time()
    jerk_regularization = float(jerk_regularization)
    second_order_regularization = float(second_order_regularization)

    for batch_idx, (batch_x, batch_coords, batch_target) in enumerate(dataloader):
        # Move data to device
        batch_x = batch_x.to(device, non_blocking=True)
        batch_coords = batch_coords.to(device, non_blocking=True)
        batch_target = batch_target.to(device, non_blocking=True)

        # --- Train the whole model ---
        optimizer.zero_grad()
        if jerk_regularization > 0.0 or second_order_regularization > 0.0:
            predictions, _, hidden_sequence = model(batch_x, batch_coords, return_hidden_sequence=True)
            prediction_loss = criterion(predictions, batch_target)
            jerk_loss = (
                latent_jerk_loss(hidden_sequence)
                if jerk_regularization > 0.0
                else predictions.new_tensor(0.0)
            )
            second_order_loss = (
                latent_second_order_loss(hidden_sequence)
                if second_order_regularization > 0.0
                else predictions.new_tensor(0.0)
            )
            loss = (
                prediction_loss
                + jerk_regularization * jerk_loss
                + second_order_regularization * second_order_loss
            )
        else:
            predictions, _ = model(batch_x, batch_coords)
            prediction_loss = criterion(predictions, batch_target)
            jerk_loss = predictions.new_tensor(0.0)
            second_order_loss = predictions.new_tensor(0.0)
            loss = prediction_loss
        loss.backward() # Backpropagation
        optimizer.step()
        total_loss += loss.item() # Accumulate batch loss (use .item() to get Python number)
        total_prediction_loss += prediction_loss.item()
        total_jerk_loss += jerk_loss.item()
        total_second_order_loss += second_order_loss.item()

        # --- Logging ---
        if batch_idx % log_interval == 0:
            n_processed = batch_idx * dataloader.batch_size + len(batch_x)
            print(f"Train Epoch: {epoch} [{n_processed}/{len(dataloader.dataset)} "
                  f"({100. * batch_idx / len(dataloader):.0f}%)]\tLoss: {loss.item():.6f}")
            if jerk_regularization > 0.0 or second_order_regularization > 0.0:
                extra_terms = [f"Prediction Loss: {prediction_loss.item():.6f}"]
                if second_order_regularization > 0.0:
                    extra_terms.append(f"Second-Order Loss: {second_order_loss.item():.6f}")
                    extra_terms.append(f"Second-Order Coeff: {second_order_regularization:.6g}")
                if jerk_regularization > 0.0:
                    extra_terms.append(f"Jerk Loss: {jerk_loss.item():.6f}")
                    extra_terms.append(f"Jerk Coeff: {jerk_regularization:.6g}")
                print("\t" + " | ".join(extra_terms))

    scheduler.step(total_loss)
    avg_loss = total_loss / len(dataloader)
    avg_prediction_loss = total_prediction_loss / len(dataloader)
    avg_jerk_loss = total_jerk_loss / len(dataloader)
    avg_second_order_loss = total_second_order_loss / len(dataloader)
    epoch_time = time.time() - start_time
    print(f"====> Epoch: {epoch} Average loss: {avg_loss:.6f} Time: {epoch_time:.2f}s")
    if jerk_regularization > 0.0 or second_order_regularization > 0.0:
        extra_terms = [f"====> Prediction loss: {avg_prediction_loss:.6f}"]
        if second_order_regularization > 0.0:
            extra_terms.append(f"Second-order loss: {avg_second_order_loss:.6f}")
        if jerk_regularization > 0.0:
            extra_terms.append(f"Jerk loss: {avg_jerk_loss:.6f}")
        print(" | ".join(extra_terms))
    print(f"====> Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

    return avg_loss  # Return the average training loss

def evaluate(model, dataloader, criterion, device, jerk_regularization=0.0, second_order_regularization=0.0):
    """
    Evaluates the model on a given dataset.

    Args:
        model (nn.Module): The model to evaluate.
        dataloader (DataLoader): DataLoader for the validation or test dataset.
        criterion: The loss function.
        device: The device (e.g., 'cuda' or 'cpu') to run evaluation on.

    Returns:
        float: The average loss over the dataset.
    """
    model.eval()  # Set the model to evaluation mode (important for Dropout, BatchNorm)
    total_loss = 0.0
    total_prediction_loss = 0.0
    total_jerk_loss = 0.0
    total_second_order_loss = 0.0
    num_batches = len(dataloader)
    start_time = time.time()
    jerk_regularization = float(jerk_regularization)
    second_order_regularization = float(second_order_regularization)

    with torch.no_grad():  # Disable gradient calculations during evaluation
        for batch_idx, (batch_x, batch_coords, batch_target) in enumerate(dataloader):
            # Move data to the correct device
            batch_x = batch_x.to(device, non_blocking=True)
            batch_coords = batch_coords.to(device, non_blocking=True)
            batch_target = batch_target.to(device, non_blocking=True)

            # --- Forward Pass ---
            if jerk_regularization > 0.0 or second_order_regularization > 0.0:
                predictions, _, hidden_sequence = model(batch_x, batch_coords, return_hidden_sequence=True)
                prediction_loss = criterion(predictions, batch_target)
                jerk_loss = (
                    latent_jerk_loss(hidden_sequence)
                    if jerk_regularization > 0.0
                    else predictions.new_tensor(0.0)
                )
                second_order_loss = (
                    latent_second_order_loss(hidden_sequence)
                    if second_order_regularization > 0.0
                    else predictions.new_tensor(0.0)
                )
                loss = (
                    prediction_loss
                    + jerk_regularization * jerk_loss
                    + second_order_regularization * second_order_loss
                )
            else:
                predictions, _ = model(batch_x, batch_coords)
                prediction_loss = criterion(predictions, batch_target)
                jerk_loss = predictions.new_tensor(0.0)
                second_order_loss = predictions.new_tensor(0.0)
                loss = prediction_loss

            # --- Calculate Loss ---
            total_loss += loss.item() # Accumulate batch loss (use .item() to get Python number)
            total_prediction_loss += prediction_loss.item()
            total_jerk_loss += jerk_loss.item()
            total_second_order_loss += second_order_loss.item()

    avg_loss = total_loss / num_batches # Calculate the average loss
    eval_time = time.time() - start_time
    print(f"====> Validation Loss: {avg_loss:.6f} | Evaluation Time: {eval_time:.2f}s")
    if jerk_regularization > 0.0 or second_order_regularization > 0.0:
        print(
            f"====> Validation prediction loss: {total_prediction_loss / num_batches:.6f} | "
            f"Validation second-order loss: {total_second_order_loss / num_batches:.6f} | "
            f"Validation jerk loss: {total_jerk_loss / num_batches:.6f}"
        )

    return avg_loss # Return the average validation loss

def train_model(
    model,
    train_loader,
    valid_loader,
    optimizer,
    scheduler,
    criterion,
    device,
    epochs,
    log_interval=50,
    eval_interval=10,
    patience=None,
    threshold=5e-3,
    save_dir=None,
    run_name=None,
    log_callback=None,
    train_sampler=None,
    jerk_regularization=0.0,
    second_order_regularization=0.0,
):
    '''
    Trains the model with early stopping and model checkpointing.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for the training dataset.
        valid_loader (DataLoader): DataLoader for the validation dataset.
        optimizer: The optimizer for updating model parameters.
        scheduler: Learning rate scheduler.
        criterion: The loss function.
        device: The device (e.g., 'cuda' or 'cpu') to run training on.
        epochs (int): Number of epochs to train.
        log_interval (int): How many batches to wait before logging training status.
        eval_interval (int): How many epochs to wait before evaluating on validation set.
        patience (int or None): Number of evaluations with no improvement to wait before stopping.
        threshold (float): Minimum relative improvement to reset patience counter.
        save_dir (str or None): Directory to save model checkpoints.
        run_name (str or None): Name of the current run (used for saving checkpoints).
        log_callback (callable or None): Optional callback function for logging.
        train_sampler: Optional DistributedSampler; if provided, set_epoch() is called each epoch.
    Returns:
        dict: Information about the training process, including paths to saved models.
    '''
    best_val_loss = float("inf")
    patience_counter = 0
    patience_steps = None
    if patience is not None and eval_interval > 0:
        patience_steps = max(1, patience // eval_interval)

    best_model_path = None
    final_model_path = None
    if save_dir is not None and run_name is not None:
        best_model_path = os.path.join(save_dir, run_name, "best_model.ckpt")
        final_model_path = os.path.join(save_dir, run_name, "final_model.ckpt")

    print("Training...")
    start = time.time()
    for epoch in range(1, epochs + 1):
        print("-" * 30)
        print(f"Epoch {epoch}/{epochs}")

        # Required for proper shuffling with DistributedSampler
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_loss = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=device,
            epoch=epoch,
            log_interval=log_interval,
            jerk_regularization=jerk_regularization,
            second_order_regularization=second_order_regularization,
        )

        if eval_interval > 0 and epoch % eval_interval == 0:
            val_loss = evaluate(
                model=model,
                dataloader=valid_loader,
                criterion=criterion,
                device=device,
                jerk_regularization=jerk_regularization,
                second_order_regularization=second_order_regularization,
            )

            if log_callback is not None:
                log_callback({
                    "epoch_train_loss": train_loss,
                    "epoch_val_loss": val_loss,
                    "epoch": epoch,
                })

            if best_model_path is not None and val_loss < best_val_loss:
                try:
                    torch.save(_unwrap_model(model).state_dict(), best_model_path)
                except Exception as e:
                    print(f"Error saving best model: {e}")
                print(f"Best model saved at epoch {epoch} with validation loss: {val_loss:.6f}")

            if val_loss < best_val_loss * (1 - threshold):
                patience_counter = 0
                print("Patience counter reset.")
            else:
                patience_counter += 1
                if patience_steps is not None:
                    print(f"Patience counter: {patience_counter}/{patience_steps}")
                    if patience_counter >= patience_steps:
                        print("Early stopping triggered.")
                        break

            best_val_loss = min(best_val_loss, val_loss)
        else:
            if log_callback is not None:
                log_callback({
                    "epoch_train_loss": train_loss,
                    "epoch": epoch,
                })

    end = time.time()
    print(f"Training completed in {end - start:.2f} seconds")

    if final_model_path is not None:
        try:
            torch.save(_unwrap_model(model).state_dict(), final_model_path)
        except Exception as e:
            print(f"Error saving model: {e}")
        print(f"Model saved to {final_model_path}")

    return {
        "best_model_path": best_model_path,
        "final_model_path": final_model_path,
        "best_val_loss": best_val_loss,
        "training_time": end - start,
    }

def save_training_results(save_dir, run_name, training_time):
    '''
    Save training results to a text file.
    '''
    results_path = os.path.join(save_dir, run_name, "results.txt")
    with open(results_path, "w") as file:
        file.write("Training Results\n")
        file.write("================\n")
        file.write(f"Training Time: {training_time:.2f} seconds\n")
    print("Results saved to results.txt successfully.")
