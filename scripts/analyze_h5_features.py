"""Analyze Radient HDF5 features with pooled PCA and L2-norm plots.

All H5 files under the input directory are visited recursively and loaded one
at a time. The summary metrics and L2 distribution use every finite vector.
PCA uses a bounded, pooled sample so large feature directories fit in memory.

Usage:
    python scripts/analyze_h5_features.py /path/to/features
    python scripts/analyze_h5_features.py /path/to/features --output-dir /tmp/review
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def find_feature_dataset(h5_file: h5py.File) -> h5py.Dataset:
    """Find the feature dataset in a Radient H5 file or a nested variant."""
    if "features" in h5_file and isinstance(h5_file["features"], h5py.Dataset):
        return h5_file["features"]

    candidates = []

    def collect(name, obj):
        if isinstance(obj, h5py.Dataset) and name.endswith("features"):
            candidates.append(obj)

    h5_file.visititems(collect)
    if candidates:
        return candidates[0]

    numeric = []
    h5_file.visititems(
        lambda _name, obj: numeric.append(obj)
        if isinstance(obj, h5py.Dataset) and np.issubdtype(obj.dtype, np.number)
        else None
    )
    if len(numeric) == 1:
        return numeric[0]
    raise KeyError("Could not identify a unique numeric feature dataset")


def vectors_from_feature_array(features: np.ndarray) -> np.ndarray:
    """Convert axial vectors, token arrays, and channel-first maps to rows."""
    features = np.asarray(features)
    if not np.issubdtype(features.dtype, np.number):
        raise TypeError(f"Feature array must be numeric, got {features.dtype}")
    features = features.astype(np.float32, copy=False)

    # Remove a singleton batch dimension: [1,D,C] or [1,C,D,H,W].
    if features.ndim >= 3 and features.shape[0] == 1:
        features = features[0]
    if features.ndim == 1:
        return features[None, :]
    if features.ndim == 2:
        return features
    if features.ndim < 1:
        raise ValueError("Feature dataset is empty")

    # Volume maps are channel-first, e.g. [C,D,H,W].
    return np.moveaxis(features, 0, -1).reshape(-1, features.shape[0])


def text_attr(value) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8", errors="replace")
    return str(value)


def read_vectors(path: Path):
    with h5py.File(path, "r") as h5_file:
        dataset = find_feature_dataset(h5_file)
        raw = dataset[()]
        attrs = {key: text_attr(value) for key, value in dataset.attrs.items()}
        attrs.update({key: text_attr(value) for key, value in h5_file.attrs.items()})
    return vectors_from_feature_array(raw), attrs, tuple(raw.shape)


def summarize_file(path: Path, input_dir: Path):
    vectors, attrs, stored_shape = read_vectors(path)
    finite_mask = np.isfinite(vectors).all(axis=1)
    finite = vectors[finite_mask]
    norms = np.linalg.norm(finite, axis=1) if len(finite) else np.array([], dtype=np.float32)
    positive_norms = norms[norms > 1e-12]

    return {
        "file": str(path.relative_to(input_dir)),
        "encoder": attrs.get("encoder", ""),
        "pid": attrs.get("pid", ""),
        "series": attrs.get("series", ""),
        "stored_shape": "x".join(map(str, stored_shape)),
        "n_vectors": int(len(vectors)),
        "feature_dim": int(vectors.shape[1]),
        "finite_vectors": int(len(finite)),
        "finite_fraction": float(finite_mask.mean()) if len(finite_mask) else 0.0,
        "zero_vector_fraction": float(np.mean(norms <= 1e-12)) if len(norms) else float("nan"),
        "mean_vector_norm": float(norms.mean()) if len(norms) else float("nan"),
        "median_vector_norm": float(np.median(norms)) if len(norms) else float("nan"),
        "mean_feature_std": float(np.mean(np.std(finite, axis=0))) if len(finite) else float("nan"),
        "min_positive_norm": float(positive_norms.min()) if len(positive_norms) else float("nan"),
        "max_vector_norm": float(norms.max()) if len(norms) else float("nan"),
        "status": "ok",
        "error": "",
    }


def write_pooled_pca(files, rows, output_dir: Path, max_samples: int, rng):
    """Fit one PCA to a pooled, proportional sample of the largest dimension group."""
    by_dimension = defaultdict(list)
    for path in files:
        row = rows[str(path)]
        if row["status"] == "ok" and row["finite_vectors"] > 0:
            by_dimension[row["feature_dim"]].append((path, row["finite_vectors"]))

    if not by_dimension:
        dimensions, selected_files = None, []
    else:
        dimensions = max(
            by_dimension,
            key=lambda dim: sum(count for _, count in by_dimension[dim]),
        )
        selected_files = by_dimension[dimensions]

    pooled = []
    source_ids = []
    if selected_files:
        counts = np.asarray([count for _, count in selected_files], dtype=np.int64)
        total = int(counts.sum())
        quotas = np.floor(counts * min(max_samples, total) / total).astype(int)
        remainder = min(max_samples, total) - int(quotas.sum())
        fractions = counts * min(max_samples, total) / total - quotas
        for index in np.argsort(fractions)[::-1]:
            if remainder <= 0:
                break
            if quotas[index] < counts[index]:
                quotas[index] += 1
                remainder -= 1

        for file_index, ((path, _count), quota) in enumerate(zip(selected_files, quotas)):
            if quota <= 0:
                continue
            try:
                vectors, _, _ = read_vectors(path)
                vectors = vectors[np.isfinite(vectors).all(axis=1)]
                if len(vectors) > quota:
                    vectors = vectors[rng.choice(len(vectors), size=quota, replace=False)]
                pooled.append(vectors)
                source_ids.extend([file_index] * len(vectors))
            except Exception as exc:
                print(f"PCA sample skipped {path}: {type(exc).__name__}: {exc}")

    plot_path = output_dir / "pooled_feature_pca.png"
    fig, ax = plt.subplots(figsize=(8, 6))
    if pooled:
        sample = np.concatenate(pooled, axis=0)
        centered = sample - sample.mean(axis=0, keepdims=True)
        if len(sample) >= 2 and sample.shape[1] >= 2 and np.any(centered):
            _, singular_values, components = np.linalg.svd(centered, full_matrices=False)
            projection = centered @ components[:2].T
            variance = singular_values**2
            explained = variance / max(float(variance.sum()), 1e-12)
            colors = np.asarray(source_ids[: len(projection)])
            points = ax.scatter(
                projection[:, 0], projection[:, 1], c=colors, cmap="turbo",
                s=9, alpha=0.55, linewidths=0,
            )
            colorbar = fig.colorbar(points, ax=ax)
            colorbar.set_label("Source file index")
            ax.set_xlabel(f"PC1 ({explained[0]:.1%} variance)")
            ax.set_ylabel(f"PC2 ({explained[1]:.1%} variance)")
            ax.set_title(
                f"Pooled PCA · {len(projection):,} sampled vectors · "
                f"{len(selected_files)} files · {dimensions} features"
            )
        else:
            ax.text(0.5, 0.5, "PCA needs at least two varying feature dimensions", ha="center", va="center")
            ax.set_axis_off()
    else:
        ax.text(0.5, 0.5, "No compatible finite feature vectors found", ha="center", va="center")
        ax.set_axis_off()

    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    return plot_path, dimensions, {path for path, _ in selected_files}


def write_l2_distribution(files, rows, output_dir: Path):
    """Plot L2 norms for all finite vectors using a second one-file-at-a-time pass."""
    ok_rows = [row for row in rows.values() if row["status"] == "ok"]
    positive_minima = [row["min_positive_norm"] for row in ok_rows if np.isfinite(row["min_positive_norm"])]
    maxima = [row["max_vector_norm"] for row in ok_rows if np.isfinite(row["max_vector_norm"])]
    zero_count = sum(
        int(round(row["zero_vector_fraction"] * row["finite_vectors"]))
        for row in ok_rows
        if np.isfinite(row["zero_vector_fraction"])
    )

    plot_path = output_dir / "pooled_l2_distribution.png"
    if not positive_minima:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, "No nonzero finite feature vectors found", ha="center", va="center")
        ax.set_axis_off()
        fig.savefig(plot_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return plot_path

    low = np.log10(min(positive_minima))
    high = np.log10(max(maxima)) if maxima and max(maxima) > 0 else low
    if high <= low:
        high = low + 1e-6
    edges = np.linspace(low, high, 101)
    histogram = np.zeros(len(edges) - 1, dtype=np.int64)
    for path in files:
        if rows[str(path)]["status"] != "ok":
            continue
        try:
            vectors, _, _ = read_vectors(path)
            finite = vectors[np.isfinite(vectors).all(axis=1)]
            norms = np.linalg.norm(finite, axis=1)
            positive = norms[norms > 1e-12]
            if len(positive):
                histogram += np.histogram(np.log10(positive), bins=edges)[0]
        except Exception as exc:
            print(f"L2 histogram skipped {path}: {type(exc).__name__}: {exc}")

    centers = (edges[:-1] + edges[1:]) / 2
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.stairs(histogram, edges, fill=True, color="#2878B5", alpha=0.8)
    ax.set_xlabel("log10(L2 norm)")
    ax.set_ylabel("Feature-vector count")
    ax.set_title(f"Pooled feature L2 norms · zero vectors: {zero_count:,}")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    return plot_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, help="Directory containing feature H5 files")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=10000, help="Maximum pooled vectors used for PCA")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    input_dir = args.input_dir.expanduser().resolve()
    if not input_dir.is_dir():
        parser.error(f"input_dir is not a directory: {input_dir}")
    if args.max_samples < 2:
        parser.error("--max-samples must be at least 2")

    output_dir = (args.output_dir or input_dir / "feature_quality").expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(
        path for path in input_dir.rglob("*.h5")
        if output_dir == input_dir or output_dir not in path.parents
    )
    if not files:
        parser.error(f"No .h5 files found under {input_dir}")

    rows = {}
    for index, path in enumerate(files, start=1):
        try:
            row = summarize_file(path, input_dir)
            print(
                f"[{index}/{len(files)}] {path.relative_to(input_dir)} "
                f"vectors={row['n_vectors']} dim={row['feature_dim']} "
                f"finite={row['finite_fraction']:.3f}"
            )
        except Exception as exc:
            row = {
                "file": str(path.relative_to(input_dir)), "encoder": "", "pid": "", "series": "",
                "stored_shape": "", "n_vectors": 0, "feature_dim": 0, "finite_vectors": 0,
                "finite_fraction": 0.0, "zero_vector_fraction": "", "mean_vector_norm": "",
                "median_vector_norm": "", "mean_feature_std": "", "min_positive_norm": "",
                "max_vector_norm": "", "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"[{index}/{len(files)}] {path.relative_to(input_dir)} ERROR: {row['error']}")
        rows[str(path)] = row

    rng = np.random.default_rng(args.seed)
    pca_path, pca_dimension, included_files = write_pooled_pca(
        files, rows, output_dir, args.max_samples, rng
    )
    l2_path = write_l2_distribution(files, rows, output_dir)

    fields = [
        "file", "encoder", "pid", "series", "stored_shape", "n_vectors", "feature_dim",
        "finite_vectors", "finite_fraction", "zero_vector_fraction", "mean_vector_norm",
        "median_vector_norm", "mean_feature_std", "min_positive_norm", "max_vector_norm",
        "pooled_pca_dimension", "pooled_pca_included", "status", "error",
    ]
    summary_path = output_dir / "feature_quality_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=fields)
        writer.writeheader()
        for path in files:
            row = rows[str(path)]
            row["pooled_pca_dimension"] = pca_dimension or ""
            row["pooled_pca_included"] = str(path in included_files)
            writer.writerow(row)

    print(f"Wrote summary: {summary_path}")
    print(f"Wrote pooled PCA: {pca_path}")
    print(f"Wrote pooled L2 distribution: {l2_path}")
    if pca_dimension is not None:
        excluded = sum(
            1 for row in rows.values()
            if row["status"] == "ok" and row["feature_dim"] != pca_dimension
        )
        if excluded:
            print(
                f"PCA used the largest feature-dimension group ({pca_dimension}); "
                f"excluded {excluded} files with other dimensions."
            )


if __name__ == "__main__":
    main()
