import h5py
import numpy as np


def make_splits(n_tasks: int, RNG_SEED: int = 42):
    rng = np.random.RandomState(RNG_SEED)
    indices = np.arange(n_tasks)
    rng.shuffle(indices)

    n_test = 30
    n_val = 30
    n_train = n_tasks - n_test - n_val

    test = np.sort(indices[:n_test])
    val = np.sort(indices[n_test:n_test + n_val])
    train = np.sort(indices[n_test + n_val:])
    print(f"Split: {len(train)} train / {len(val)} val / {len(test)} test")
    return train, val, test


with h5py.File("../datasets/pc_mango/bbv_v1.hdf5", "a") as out:
    n_tasks = 281
    train, val, test = make_splits(n_tasks)
    splits = out.create_group("splits")
    splits.create_dataset("train_indices", data=train)
    splits.create_dataset("val_indices", data=val)
    splits.create_dataset("test_indices", data=test)