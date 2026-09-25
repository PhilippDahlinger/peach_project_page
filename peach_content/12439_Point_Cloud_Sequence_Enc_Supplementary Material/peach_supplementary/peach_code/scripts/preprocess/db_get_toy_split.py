import h5py
import numpy as np


def create_debug_dataset():
    """
    Creates a smaller debug dataset (db_debug.hdf5) from db_v3.hdf5
    - Copies the first 10 tasks
    - Copies all global_data (faces, metadata, etc.)
    - Creates new splits: 5 train, 3 val, 2 test
    """
    source_file = "../datasets/pc_mango/db_v3.hdf5"
    target_file = "../datasets/pc_mango/db_debug.hdf5"

    num_tasks_to_copy = 10
    train_indices = list(range(0, 5))      # First 5 tasks
    val_indices = list(range(5, 8))        # Next 3 tasks
    test_indices = list(range(8, 10))      # Last 2 tasks

    with h5py.File(source_file, "r") as src, h5py.File(target_file, "w") as dst:

        # ------------------------------
        # Copy global_data recursively
        # ------------------------------
        if "global_data" in src:
            src.copy("global_data", dst)
            print("Copied global_data")

        # ------------------------------
        # Get task names and sort them
        # ------------------------------
        task_names = sorted([k for k in src.keys() if k.startswith("task_")])
        tasks_to_copy = task_names[:num_tasks_to_copy]

        print(f"Found {len(task_names)} tasks in source file")
        print(f"Copying first {num_tasks_to_copy} tasks: {tasks_to_copy}")

        # ------------------------------
        # Copy selected tasks
        # ------------------------------
        for task_name in tasks_to_copy:
            src.copy(task_name, dst)
            print(f"  Copied {task_name}")

        # ------------------------------
        # Create new splits group
        # ------------------------------
        if "splits" in dst:
            del dst["splits"]

        splits_group = dst.create_group("splits")
        splits_group.create_dataset("train_indices", data=np.array(train_indices, dtype=np.int64))
        splits_group.create_dataset("val_indices", data=np.array(val_indices, dtype=np.int64))
        splits_group.create_dataset("test_indices", data=np.array(test_indices, dtype=np.int64))

        print(f"\nCreated splits:")
        print(f"  train_indices: {train_indices}")
        print(f"  val_indices: {val_indices}")
        print(f"  test_indices: {test_indices}")

    print(f"\nSuccessfully created {target_file}")


if __name__ == "__main__":
    create_debug_dataset()

