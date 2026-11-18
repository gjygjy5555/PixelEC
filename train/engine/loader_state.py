"""Resume the yielded stream, not just a sampler's prefetched dispatch cursor.

Requires Accelerate's native StatefulDataLoader integration. Fail closed if the
installed versions cannot preserve that loader through accelerator.prepare().
No deterministic-per-index augmentation or change to worker count is introduced.
"""
import copy
import hashlib
import inspect
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler, RandomSampler


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state())


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())


class WorkerStateDataset(Dataset):
    """Expose the worker's augmentation RNGs to torchdata snapshots."""
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

    def state_dict(self):
        result = dict(rng=rng_state())
        if hasattr(self.dataset, 'state_dict'):
            result['dataset'] = self.dataset.state_dict()
        return result

    def load_state_dict(self, state):
        if 'dataset' in state:
            self.dataset.load_state_dict(state['dataset'])
        restore_rng(state['rng'])


class ResumableRandomSampler(Sampler):
    """Same fresh-epoch ordering and implicit seed draw as RandomSampler.

    State is consumed by torchdata along with its yielded-batch/worker snapshots;
    restoring only this position without torchdata's prefetch state is forbidden.
    """
    def __init__(self, data_source):
        self.n = len(data_source)
        self.order = None
        self.position = 0
        self.epoch = 0

    def __len__(self):
        return self.n

    def __iter__(self):
        if self.order is None:
            self.order = list(RandomSampler(range(self.n)))
            self.position = 0
        while self.position < self.n:
            item = self.order[self.position]
            self.position += 1
            yield item
        self.order = None
        self.position = 0
        self.epoch += 1

    def state_dict(self):
        return dict(n=self.n, order=copy.deepcopy(self.order), position=self.position, epoch=self.epoch)

    def load_state_dict(self, state):
        assert state['n'] == self.n
        self.order = copy.deepcopy(state['order'])
        self.position = state['position']
        self.epoch = state['epoch']
        assert 0 <= self.position <= self.n


class LoaderCheckpoint:
    def __init__(self, loader, fingerprint):
        self.loader, self.fingerprint = loader, fingerprint

    def state_dict(self):
        return dict(version=1, fingerprint=self.fingerprint, loader=self.loader.state_dict())

    def load_state_dict(self, state):
        assert state['version'] == 1 and state['fingerprint'] == self.fingerprint
        self.loader.load_state_dict(state['loader'])
        self.loader._restored_stream_pending = True
        self.loader._restored_at_epoch_end = bool(state['loader'].get('_iterator_finished',False))
        print('LOADER_STREAM_STATE_RESTORED', self.fingerprint, flush=True)


def install(official):
    from accelerate import DataLoaderConfiguration
    from torchdata.stateful_dataloader import StatefulDataLoader
    assert 'use_stateful_dataloader' in inspect.signature(DataLoaderConfiguration).parameters, 'Accelerate lacks stateful loader support; do not start'
    original_accelerator = official.Accelerator
    current = {}

    def loader_factory(dataset, *args, **kwargs):
        assert not args and kwargs.pop('shuffle', False) is True
        assert kwargs.get('sampler') is None and kwargs.get('batch_sampler') is None
        dataset = WorkerStateDataset(dataset)
        kwargs['sampler'] = ResumableRandomSampler(dataset)
        loader = StatefulDataLoader(dataset, shuffle=False, snapshot_every_n_steps=1, **kwargs)
        current['loader'] = loader
        # Paths/order are checked independently in the manifest audit.
        info = dict(n=len(dataset), batch_size=kwargs['batch_size'], workers=kwargs['num_workers'],
                    drop_last=kwargs['drop_last'], persistent=kwargs['persistent_workers'])
        current['fingerprint'] = hashlib.sha256(json.dumps(info, sort_keys=True).encode()).hexdigest()
        return loader

    class CheckedAccelerator(original_accelerator):
        def __init__(self, *a, **kw):
            assert 'dataloader_config' not in kw
            kw['dataloader_config'] = DataLoaderConfiguration(use_stateful_dataloader=True)
            super().__init__(*a, **kw)
            assert self.num_processes == 1, 'This authorized protocol is one process per route'

        def load_state(self, input_dir, *args, **kwargs):
            from pathlib import Path
            trusted=Path(input_dir).resolve()
            import os
            allowed = os.environ.get('PIXELEC_TRUSTED_CHECKPOINT')
            assert allowed and trusted == Path(allowed).resolve(), 'Only the explicitly selected full checkpoint is trusted'
            original_load=torch.load
            def own_state_load(path, *a, **kw):
                if not isinstance(path,(str,Path)) or not Path(path).resolve().is_relative_to(trusted):
                    raise RuntimeError('Only this experiment own checkpoint files may use trusted pickle')
                kw.setdefault('weights_only',False)
                return original_load(path,*a,**kw)
            torch.load=own_state_load
            try:return super().load_state(input_dir,*args,**kwargs)
            finally:torch.load=original_load

        def prepare(self, *objects, **kw):
            prepared = super().prepare(*objects, **kw)
            items = prepared if isinstance(prepared, tuple) else (prepared,)
            loaders = [x for x in items if hasattr(x, 'state_dict') and hasattr(x, 'base_dataloader')]
            assert len(loaders) == 1, 'Prepared stateful loader not found; do not train'
            loader = loaders[0]
            assert isinstance(loader.base_dataloader, StatefulDataLoader), 'Accelerate discarded stateful loader'
            shard_class=type(loader)
            if not getattr(shard_class,'_fresh_resume_rng_guard',False):
                original_iter=shard_class.__iter__
                def resume_iter(self):
                    if not getattr(self,'_restored_stream_pending',False):
                        yield from original_iter(self)
                        return
                    self._restored_stream_pending=False
                    saved=rng_state()
                    cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
                    epoch_end=getattr(self,'_restored_at_epoch_end',False)
                    base=self.base_dataloader
                    original_generator=base.generator
                    # Recreated workers need a base seed; uninterrupted
                    # persistent workers do not draw it again. Keep that draw
                    # off the training RNG, while letting a NEW epoch's sampler
                    # consume its legitimate original global seed.
                    if original_generator is None:
                        recreated_worker_generator=torch.Generator()
                        recreated_worker_generator.set_state(saved['torch'])
                        base.generator=recreated_worker_generator
                    iterator=original_iter(self)
                    try:first=next(iterator)
                    except StopIteration:
                        base.generator=original_generator
                        restore_rng(saved)
                        if cuda is not None:torch.cuda.set_rng_state_all(cuda)
                        return
                    base.generator=original_generator
                    if epoch_end:
                        random.setstate(saved['python'])
                        np.random.set_state(saved['numpy'])
                    else:restore_rng(saved)
                    if cuda is not None:torch.cuda.set_rng_state_all(cuda)
                    yield first
                    yield from iterator
                shard_class.__iter__=resume_iter
                shard_class._fresh_resume_rng_guard=True
            self.register_for_checkpointing(LoaderCheckpoint(loader, current['fingerprint']))
            print('LOADER_STREAM_CHECKPOINTING_ENABLED', current['fingerprint'], flush=True)
            return prepared

    official.DataLoader = loader_factory
    official.Accelerator = CheckedAccelerator


def smoke_test():
    """CPU-only worker-prefetch resume test; no test images or model updates."""
    from torchdata.stateful_dataloader import StatefulDataLoader

    class RandomDataset(Dataset):
        def __len__(self): return 73
        def __getitem__(self, i):
            return torch.tensor([i, random.random(), float(np.random.rand()), float(torch.rand(()))], dtype=torch.float64)

    def build():
        d = WorkerStateDataset(RandomDataset())
        return StatefulDataLoader(d, sampler=ResumableRandomSampler(d), batch_size=3,
                                  num_workers=2, persistent_workers=True, snapshot_every_n_steps=1)
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    first=build(); iterator=iter(first)
    for _ in range(7): next(iterator)
    state=copy.deepcopy(first.state_dict())
    expected=[next(iterator) for _ in range(10)]
    second=build(); second.load_state_dict(state); restored=iter(second)
    actual=[next(restored) for _ in range(10)]
    assert all(torch.equal(a,b) for a,b in zip(expected,actual)), 'Worker/augmentation/prefetch mismatch'
    print('WORKER_PREFETCH_RESUME_IDENTICAL_10_BATCHES', flush=True)

if __name__ == '__main__': smoke_test()
