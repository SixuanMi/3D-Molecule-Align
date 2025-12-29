#!/usr/bin/env python3
import argparse
import os
import pickle
import sys
import types

from sklearn.model_selection import train_test_split

try:
    import torch
    from torch_geometric.data import Data
except ImportError as exc:
    print(
        "Error: torch and torch_geometric are required to match the training split filtering.",
        file=sys.stderr,
    )
    print(f"Details: {exc}", file=sys.stderr)
    sys.exit(1)

def ensure_rmsd_stub():
    try:
        import rmsd  # noqa: F401
        return
    except Exception:
        stub = types.SimpleNamespace(
            ELEMENT_WEIGHTS={},
            ELEMENT_NAMES={},
            NAMES_ELEMENT={},
            AXIS_REFLECTIONS=[],
        )
        sys.modules["rmsd"] = stub


def ensure_src_import_path(data_dir):
    candidates = [
        os.path.abspath(os.path.join(data_dir, "..", "..")),
        os.path.abspath(os.path.join(data_dir, "..")),
        os.path.abspath(os.path.join(data_dir, "..", "..", "..")),
    ]
    for candidate in candidates:
        if os.path.isfile(os.path.join(candidate, "src", "utils.py")):
            if candidate not in sys.path:
                sys.path.insert(0, candidate)
            return candidate
    return None


def resolve_data_dir(provided_dir):
    if provided_dir:
        return provided_dir
    candidates = ["data/ready", "rmsd_test/data/ready", "../data/ready"]
    for path in candidates:
        if os.path.isdir(path):
            return path
    return None


def dataset_label_from_filename(filename):
    base = os.path.splitext(os.path.basename(filename))[0]
    if base.endswith("_dataset"):
        base = base[: -len("_dataset")]
    return base


def mapping_is_valid(mapping):
    try:
        ref = mapping.structure_ref
        cand = mapping.structure_cand
        Data(
            x=torch.tensor(ref.atoms, dtype=torch.long),
            pos=torch.tensor(ref.coordinates, dtype=torch.float),
            h_t=torch.tensor(cand.atoms, dtype=torch.long),
            pos_t=torch.tensor(cand.coordinates, dtype=torch.float),
            mapping=torch.tensor(mapping.mapping_indices, dtype=torch.long),
        )
    except Exception:
        return False
    return True


def iter_dataset_files(data_dir):
    for filename in sorted(os.listdir(data_dir)):
        if filename.endswith("_dataset.pkl"):
            yield os.path.join(data_dir, filename)


def split_dataset_file(file_path, test_size=0.2, seed=42):
    try:
        with open(file_path, "rb") as f:
            mappings = pickle.load(f)
    except Exception as exc:
        print(f"Warning: failed to load {file_path}: {exc}", file=sys.stderr)
        return None, None, 0

    valid_mappings = []
    skipped_mappings = 0
    for m in mappings:
        if mapping_is_valid(m):
            valid_mappings.append(m)
        else:
            skipped_mappings += 1

    if not valid_mappings:
        print(f"Warning: no valid mappings in {file_path}", file=sys.stderr)
        return None, None, skipped_mappings

    try:
        train_mappings, val_mappings = train_test_split(
            valid_mappings,
            test_size=test_size,
            random_state=seed,
        )
    except ValueError as exc:
        print(f"Warning: split failed for {file_path}: {exc}", file=sys.stderr)
        return None, None, skipped_mappings

    return train_mappings, val_mappings, skipped_mappings


def main():
    parser = argparse.ArgumentParser(
        description="Split merged datasets into train/val, matching training split."
    )
    parser.add_argument("--data-dir", default=None, help="Path to ready data directory.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Validation split size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    data_dir = resolve_data_dir(args.data_dir)
    if not data_dir:
        print("Error: could not locate a ready data directory.", file=sys.stderr)
        sys.exit(1)

    ensure_rmsd_stub()
    src_root = ensure_src_import_path(data_dir)
    if not src_root:
        print(
            "Warning: could not locate src/utils.py; unpickling may fail.",
            file=sys.stderr,
        )

    total_files = 0
    total_skipped = 0
    for file_path in iter_dataset_files(data_dir):
        total_files += 1
        train_mappings, val_mappings, skipped = split_dataset_file(
            file_path, test_size=args.test_size, seed=args.seed
        )
        total_skipped += skipped
        if train_mappings is None or val_mappings is None:
            continue

        label = dataset_label_from_filename(file_path)
        train_path = os.path.join(data_dir, f"{label}_train.pkl")
        val_path = os.path.join(data_dir, f"{label}_val.pkl")

        with open(train_path, "wb") as f:
            pickle.dump(train_mappings, f)
        with open(val_path, "wb") as f:
            pickle.dump(val_mappings, f)

        print(
            f"{label}: total={len(train_mappings) + len(val_mappings)} "
            f"train={len(train_mappings)} val={len(val_mappings)}"
        )

    if total_files == 0:
        print("Error: no *_dataset.pkl files found to split.", file=sys.stderr)
        sys.exit(1)
    if total_skipped:
        print(f"Skipped mappings: {total_skipped}")


if __name__ == "__main__":
    main()
