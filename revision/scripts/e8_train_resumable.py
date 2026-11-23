"""E8 resumable training for wall-time-limited cluster jobs.

Runs the released C-STRIDE trainer (cstride.train_next_step) unchanged, except that its train_model
is replaced by a loop that (1) resumes from a state file holding model, optimizer (SOAP), scheduler,
epoch and RNG states, (2) saves that state atomically after every epoch, and (3) exits with code 3
before the wall limit so a dependent job continues. The epoch body is the released train_epoch /
evaluate; there is no early stopping (patience null) and no model selection: final_model.ckpt.

A restart re-creates the data-loader workers, whose random streams (point sampling) start again
from their seeds; the sampler's shuffling generator, torch/numpy/python and CUDA RNG states are
restored. This is logged at every resume.

Environment: E8_STATE (state file), E8_WALL_SECONDS (wall budget for this job, default unlimited).
Usage: python e8_train_resumable.py --config <cfg.yaml> --device cuda:0
"""
import paths as P  # noqa: E402  (input and output locations, see paths.py)
import os
import random
import sys
import time
from pathlib import Path

CS = P.CSTRIDE_ROOT
sys.path.insert(0, str(CS / 'code'))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import cstride.train as T  # noqa: E402
import cstride.train_next_step as TNS  # noqa: E402

STATE = Path(os.environ['E8_STATE'])
WALL = float(os.environ.get('E8_WALL_SECONDS', 'inf'))
T0 = time.time()


def _sampler_generator(loader):
    s = getattr(loader, 'sampler', None)
    return getattr(s, 'generator', None)


def resumable_train_model(model, train_loader, valid_loader, optimizer, scheduler, criterion, device, epochs,
                          log_interval=50, eval_interval=10, patience=None, threshold=5e-3, save_dir=None,
                          run_name=None, log_callback=None, train_sampler=None, jerk_regularization=0.0,
                          second_order_regularization=0.0):
    # patience None: fixed epoch budget, monitor loss logged only (E8).
    # patience set: the released cstride.train.train_model rule (E5): evaluate every eval_interval epochs,
    # save best_model on any improvement, reset the counter only on a relative improvement > threshold,
    # stop after patience // eval_interval evaluations without one. Its state is carried across resumes.
    patience_steps = max(1, patience // eval_interval) if (patience is not None and eval_interval > 0) else None
    eval_interval = int(os.environ.get('RESUMABLE_EVAL_INTERVAL', eval_interval))  # test override only
    if os.environ.get('RESUMABLE_PATIENCE_STEPS'):
        patience_steps = int(os.environ['RESUMABLE_PATIENCE_STEPS'])  # test override only
    net = T._unwrap_model(model)
    gen = _sampler_generator(train_loader)
    start_epoch, elapsed_before = 1, 0.0
    best_val_loss, patience_counter, best_epoch, stopped_early = float('inf'), 0, None, False
    name = STATE.stem.replace('_state', '')
    best_copy = STATE.parent / f'{name}_best_model.ckpt'
    if STATE.exists():
        s = torch.load(STATE, map_location='cpu', weights_only=False)  # RNG states must stay CPU ByteTensors
        net.load_state_dict(s['model'])
        optimizer.load_state_dict(s['optimizer'])
        scheduler.load_state_dict(s['scheduler'])
        torch.set_rng_state(s['torch_rng'])
        if torch.cuda.is_available() and s.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all(s['cuda_rng'])
        np.random.set_state(s['numpy_rng'])
        random.setstate(s['python_rng'])
        if gen is not None and s.get('sampler_generator') is not None:
            gen.set_state(s['sampler_generator'])
        start_epoch, elapsed_before = s['epoch'] + 1, s['elapsed']
        best_val_loss = s.get('best_val_loss', float('inf'))
        patience_counter = s.get('patience_counter', 0)
        best_epoch = s.get('best_epoch')
        stopped_early = s.get('stopped_early', False)
        print(f"RESUMED from {STATE} after epoch {s['epoch']} (lr {optimizer.param_groups[0]['lr']:.3g}, "
              f"best_val_loss {best_val_loss:.6g} at epoch {best_epoch}, patience counter {patience_counter}). "
              'NOTE: data-loader worker random streams restart from their seeds after a resume.', flush=True)
    print(f'Training epochs {start_epoch}..{epochs}; wall budget {WALL:.0f} s; '
          f'early stopping {"off" if patience_steps is None else f"after {patience_steps} evaluations"}', flush=True)
    for epoch in range(start_epoch, epochs + 1):
        if stopped_early:
            break
        print('-' * 30)
        print(f'Epoch {epoch}/{epochs}', flush=True)
        t_ep = time.time()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_loss = T.train_epoch(model=model, dataloader=train_loader, optimizer=optimizer, scheduler=scheduler,
                                   criterion=criterion, device=device, epoch=epoch, log_interval=log_interval,
                                   jerk_regularization=jerk_regularization,
                                   second_order_regularization=second_order_regularization)
        if eval_interval > 0 and epoch % eval_interval == 0:
            val_loss = T.evaluate(model=model, dataloader=valid_loader, criterion=criterion, device=device,
                                  jerk_regularization=jerk_regularization,
                                  second_order_regularization=second_order_regularization)
            if patience_steps is None:
                print(f'Monitor loss (training event, logged only, no selection) at epoch {epoch}: {val_loss:.6f}', flush=True)
            else:
                if val_loss < best_val_loss:
                    torch.save(net.state_dict(), best_copy)
                    if save_dir is not None and run_name is not None:
                        torch.save(net.state_dict(), os.path.join(save_dir, run_name, 'best_model.ckpt'))
                    best_epoch = epoch
                    print(f'Best model saved at epoch {epoch} with validation loss: {val_loss:.6f}', flush=True)
                if val_loss < best_val_loss * (1 - threshold):
                    patience_counter = 0
                    print('Patience counter reset.', flush=True)
                else:
                    patience_counter += 1
                    print(f'Patience counter: {patience_counter}/{patience_steps}', flush=True)
                    if patience_counter >= patience_steps:
                        print('Early stopping triggered.', flush=True)
                        stopped_early = True
                best_val_loss = min(best_val_loss, val_loss)
        state = dict(model=net.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                     best_val_loss=best_val_loss, patience_counter=patience_counter, best_epoch=best_epoch,
                     stopped_early=stopped_early,
                     epoch=epoch, elapsed=elapsed_before + (time.time() - T0), train_loss=train_loss,
                     torch_rng=torch.get_rng_state(),
                     cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                     numpy_rng=np.random.get_state(), python_rng=random.getstate(),
                     sampler_generator=gen.get_state() if gen is not None else None)
        tmp = STATE.with_suffix('.tmp')
        torch.save(state, tmp)
        os.replace(tmp, STATE)
        dt = time.time() - t_ep
        print(f'Epoch {epoch} wall {dt:.1f} s; state saved', flush=True)
        if stopped_early:
            break
        if epoch < epochs and (time.time() - T0) + 1.3 * dt > WALL:
            print(f'Stopping before the wall limit after epoch {epoch}; resubmit to continue.', flush=True)
            sys.exit(3)
    final = STATE.parent / f'{name}_final_model.ckpt'
    torch.save(net.state_dict(), final)
    if save_dir is not None and run_name is not None:
        torch.save(net.state_dict(), os.path.join(save_dir, run_name, 'final_model.ckpt'))
    done = torch.load(STATE, map_location='cpu', weights_only=False)['epoch'] if STATE.exists() else 0
    summary = dict(epochs_completed=int(done), configured_epochs=int(epochs), stopped_early=bool(stopped_early),
                   best_epoch=best_epoch, best_val_loss=None if best_epoch is None else float(best_val_loss),
                   training_seconds=float(elapsed_before + (time.time() - T0)))
    (STATE.parent / f'{name}_summary.json').write_text(__import__('json').dumps(summary, indent=1))
    print(f'Training complete: {summary}; final model {final}', flush=True)
    return dict(best_model_path=str(best_copy) if best_epoch else None, final_model_path=str(final),
                best_val_loss=summary['best_val_loss'], training_time=summary['training_seconds'])


TNS.train_model = resumable_train_model

if __name__ == '__main__':
    import argparse
    import multiprocessing
    # Python 3.14 (the cluster env) defaults to forkserver, which cannot pickle the trainer's local
    # worker_init_fn; fork is the Linux default of the Python 3.12 env the released runs used.
    if multiprocessing.get_start_method(allow_none=True) != 'fork':
        print(f'multiprocessing start method {multiprocessing.get_start_method(allow_none=True)} -> fork '
              '(as in the released Python 3.12 environment)', flush=True)
        multiprocessing.set_start_method('fork', force=True)
    torch.set_default_device('cpu')
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--ddp', action='store_true', default=False)
    parser.add_argument('--importance-sampling', dest='importance_sampling', action='store_true', default=False)
    parser.add_argument('--init-checkpoint', default=None)
    args = TNS.parse_args(parser)
    with open(args.config) as f:
        config = TNS.load_config(f)
    config.setdefault('training', {})
    TNS.main(config, args)
