"""One-off data prep for the Phase 4 real end-to-end secure_mpc validation
run. Downloads MNIST, partitions it across N clients + 1 server validation
split, and writes the exact directory/file layout Flotilla's existing
dataset-loading code expects:

- Client side (client/client_file_manager.py:get_available_datasets):
  <datasets_dir>/<dataset_id>/train_dataset_config.yaml with a
  `dataset_details: {data_filename: <name>.pth}` key (data_filename gets
  deleted after resolving the path) and a sibling top-level `metadata:
  {num_items, label_distribution}` key -- this is exactly what
  aggregator_fedavg.py/aggregator_secure_mpc.py read as
  `current_dataset_detail["metadata"]["num_items"]` for FedAvg-style
  dataset-size weighting.
- Server validation side (server/server_file_manager.py:get_available_datasets):
  <validation_data_dir>/<dataset_id>/dataset_config.yaml (NOT
  "train_dataset_config.yaml" -- a different filename server-side), same
  `dataset_details.data_filename` convention, `data_filename` NOT deleted
  (server code just rewrites it to an absolute path).
- The .pth files themselves are torch.save()'d DataLoader objects wrapping
  a torch.utils.data.Subset, per src/utils/data_partitioner.py's existing
  (if unused-by-anything-else) reference pattern -- client_dataset_loader.py
  reads them back via `torch.load(path).dataset`.

LeNet5 (models/LeNet5/model.py) expects 32x32 input (MNIST is natively
28x28), hence the Resize(32) in the transform.
"""

import os
import pickle
from collections import Counter

import torch
from torchvision import datasets, transforms

NUM_CLIENTS = 3
DATASET_ID = "MNIST"
OUTPUT_ROOT = os.path.join(os.path.dirname(__file__), "mnist_data")

transform = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)


def _write_dataset_dir(root_dir, config_filename, subset, name):
    os.makedirs(root_dir, exist_ok=True)
    pth_path = os.path.join(root_dir, f"{name}.pth")
    loader = torch.utils.data.DataLoader(subset, batch_size=1, shuffle=True)
    torch.save(loader, pth_path)

    num_items = len(subset)
    label_distribution = dict(Counter(int(subset[i][1]) for i in range(num_items)))
    for k, v in label_distribution.items():
        label_distribution[k] = v / num_items

    config = {
        "dataset_details": {"data_filename": f"{name}.pth"},
        "metadata": {"num_items": num_items, "label_distribution": label_distribution},
    }
    with open(os.path.join(root_dir, config_filename), "w") as f:
        import yaml

        yaml.safe_dump(config, f)

    # Client-side counterpart to the above, in the FLAT shape
    # utils/get_data_summary.py writes and
    # client/client_file_manager.get_dataset_details reads back (num_items
    # at the top level, no "metadata" wrapper -- a different, unrelated
    # shape from the server-side dataset_config.yaml/current_dataset_detail
    # structure above; client_secure_agg_manager's dataset-size lookup goes
    # through THIS file, not the YAML one).
    summary = {
        "label_distribution": label_distribution,
        "num_items": num_items,
        "data_filename": pth_path,
    }
    summary_path = os.path.join(root_dir, f"{name}_summary.data")
    with open(summary_path, "wb") as f:
        pickle.dump(summary, f)

    print(
        f"wrote {root_dir}: {num_items} items -> {config_filename}, {name}.pth, {name}_summary.data"
    )


def main():
    print("Downloading MNIST (train split)...")
    full_train = datasets.MNIST(root=OUTPUT_ROOT, train=True, transform=transform, download=True)

    # Reserve a small server-side validation split, partition the rest
    # across NUM_CLIENTS clients (disjoint, roughly equal shares).
    total = len(full_train)
    val_size = 2000
    remaining = total - val_size
    per_client = remaining // NUM_CLIENTS

    val_indices = list(range(0, val_size))
    val_subset = torch.utils.data.Subset(full_train, val_indices)
    _write_dataset_dir(
        os.path.join(OUTPUT_ROOT, "val", DATASET_ID),
        "dataset_config.yaml",
        val_subset,
        "val_partition",
    )

    for i in range(NUM_CLIENTS):
        start = val_size + i * per_client
        end = total if i == NUM_CLIENTS - 1 else val_size + (i + 1) * per_client
        client_subset = torch.utils.data.Subset(full_train, list(range(start, end)))
        _write_dataset_dir(
            os.path.join(OUTPUT_ROOT, f"client{i}", DATASET_ID),
            "train_dataset_config.yaml",
            client_subset,
            "client_partition",
        )

    print("Done.")


if __name__ == "__main__":
    main()
