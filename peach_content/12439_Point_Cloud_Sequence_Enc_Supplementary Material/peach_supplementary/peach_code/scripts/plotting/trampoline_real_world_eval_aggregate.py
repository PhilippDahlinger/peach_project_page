import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd
import scipy.stats
import seaborn as sns
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.plotting.configs import names, styles

matplotlib.use("Agg")
from matplotlib import pyplot as plt

MAX_AGGREGATION_TIMESTEP = 10


def tensor_to_float(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().reshape(-1)[0])
    return float(value)


def trimmed_mean(values) -> float:
    return float(
        scipy.stats.trim_mean(pd.Series(values).to_numpy(), proportiontocut=0.0)
    )


def trim_mean_estimator(values) -> float:
    return float(
        scipy.stats.trim_mean(pd.Series(values).to_numpy(), proportiontocut=0.0)
    )


def slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def infer_method_name(model_checkpoint: str | None) -> str:
    if not model_checkpoint:
        return "unknown"

    checkpoint_str = str(model_checkpoint).lower()
    method_aliases = {
        "oracle_mango_decoder": "oracle_encoder_mango_decoder",
        "pointpatch_encoder_mango_decoder": "pointpatch_encoder_mango_decoder",
        "peach_realworld": "pointpatch_encoder_mango_decoder",
        "dummy_mango_decoder": "dummy_encoder_mango_decoder",
        "pstnet_mango_decoder": "pstnet_encoder_mango_decoder",
        "gnn_encoder_mango_decoder": "gnn_encoder_mango_decoder",
        "gnn_mango_decoder": "gnn_encoder_mango_decoder",
        "oracle_mgn": "mgn_oracle",
        "mgn": "mgn",
    }
    for alias, method_name in sorted(
        method_aliases.items(), key=lambda item: len(item[0]), reverse=True
    ):
        if alias in checkpoint_str:
            return method_name

    checkpoint_path = Path(str(model_checkpoint))
    if checkpoint_path.parent.name:
        return checkpoint_path.parent.name
    return checkpoint_path.stem


def aggregate_loss_over_time(loss_over_time: torch.Tensor) -> tuple[float, int]:
    capped_timestep = min(MAX_AGGREGATION_TIMESTEP, max(loss_over_time.numel() - 1, 0))
    return float(loss_over_time[: capped_timestep + 1].mean()), capped_timestep


def format_material_combination(latex: int, sphere_id: int) -> str:
    return f"latex-{latex}_sphere-{sphere_id}"


def parse_seed(path: Path) -> str:
    for parent in path.parents:
        if parent.name.startswith("seed_"):
            return parent.name.replace("seed_", "")
    return "unknown"


def load_result_files(results_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    time_rows = []

    for result_path in tqdm(sorted(results_root.rglob("traj-*.pt")), desc="Loading result files"):
        result = torch.load(result_path, map_location="cpu")
        loss_over_time = result["loss_over_time"].detach().cpu().flatten()
        model_checkpoint = result.get("model_checkpoint")
        method = infer_method_name(model_checkpoint)
        mat_props = result.get("mat_props")
        if isinstance(mat_props, torch.Tensor):
            mat_props = mat_props.detach().cpu().flatten()
        else:
            mat_props = torch.tensor([])
        truncated_loss_mean, aggregation_max_timestep = aggregate_loss_over_time(
            loss_over_time
        )
        latex = int(result["latex"])
        sphere_id = int(result["sphere_id"])
        material_combination = format_material_combination(latex, sphere_id)

        row = {
            "path": str(result_path),
            "seed": str(result.get("seed", parse_seed(result_path))),
            "method": method,
            "model_checkpoint": str(model_checkpoint) if model_checkpoint else None,
            "material_id": result.get("material_id", result_path.parent.name),
            "condition_id": result.get("condition_id", result_path.parent.name),
            "material_combination": material_combination,
            "latex": latex,
            "sphere_id": sphere_id,
            "diameter": float(result["diameter"]),
            "target_traj_idx": int(result["target_traj_idx"]),
            "loss_mean": truncated_loss_mean,
            "loss_mean_all_time": float(loss_over_time.mean()),
            "loss_final": tensor_to_float(result["loss_final"]),
            "loss_min": float(loss_over_time.min()),
            "loss_max": float(loss_over_time.max()),
            "aggregation_max_timestep": aggregation_max_timestep,
        }
        for idx, value in enumerate(mat_props):
            row[f"mat_prop_{idx}"] = float(value)
        rows.append(row)

        for timestep, loss in enumerate(loss_over_time):
            time_rows.append(
                {
                    "seed": row["seed"],
                    "method": row["method"],
                    "material_id": row["material_id"],
                    "condition_id": row["condition_id"],
                    "material_combination": row["material_combination"],
                    "latex": row["latex"],
                    "sphere_id": row["sphere_id"],
                    "target_traj_idx": row["target_traj_idx"],
                    "timestep": timestep,
                    "loss": float(loss),
                }
            )

    if len(rows) == 0:
        raise FileNotFoundError(f"No traj-*.pt result files found in {results_root}")

    return pd.DataFrame(rows), pd.DataFrame(time_rows)


def summarize(data: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    summary = (
        data.groupby(group_cols, dropna=False)
        .agg(
            loss_mean=("loss_mean", "mean"),
            loss_mean_std=("loss_mean", "std"),
            loss_mean_all_time=("loss_mean_all_time", "mean"),
            loss_mean_all_time_std=("loss_mean_all_time", "std"),
            loss_final=("loss_final", "mean"),
            loss_final_std=("loss_final", "std"),
            n=("loss_mean", "count"),
        )
        .reset_index()
    )
    return summary


def save_summaries(data: pd.DataFrame, out_dir: Path) -> dict[str, pd.DataFrame]:
    summaries = {
        "per_seed_method_material_trajectory": summarize(
            data, ["method", "seed", "material_combination", "target_traj_idx"]
        ),
        "per_seed_method_material": summarize(
            data, ["method", "seed", "material_combination"]
        ),
        "per_seed_method": summarize(data, ["method", "seed"]),
        "per_method_material": summarize(data, ["method", "material_combination"]),
        "per_method": summarize(data, ["method"]),
        "per_seed_material_trajectory": summarize(
            data, ["seed", "material_id", "target_traj_idx"]
        ),
        "per_seed_material": summarize(data, ["seed", "material_id"]),
        "per_seed_condition": summarize(data, ["seed", "condition_id"]),
        "per_seed": summarize(data, ["seed"]),
        "per_material": summarize(data, ["material_id"]),
        "per_condition": summarize(data, ["condition_id"]),
        "per_trajectory": summarize(data, ["target_traj_idx"]),
        "overall": summarize(data.assign(all_results="all"), ["all_results"]),
    }

    write_json(data, out_dir / "trajectory_metrics.json")
    for name, summary in summaries.items():
        write_json(summary, out_dir / f"{name}.json")
    return summaries


def write_json(data: pd.DataFrame, out_path: Path):
    records = data.to_dict(orient="records")
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)


def configure_plot_style(figure_width: int = 8, figure_height: int = 4):
    sns.set(rc={"figure.figsize": (figure_width, figure_height)})
    sns.set_theme()
    sns.set_style("whitegrid")
    plt.rcParams.update({"ytick.left": True})


def build_method_style(methods_in_plot: list[str]):
    fallback_colors = sns.color_palette("tab10", n_colors=max(len(methods_in_plot), 1))
    fallback_markers = ["o", "s", "^", "D", "X", "P", "v", "*", "H", "<", ">"]
    fallback_dashes = ["", (4, 2), (2, 2), (1, 1)]

    palette: dict[str, Any] = {}
    markers: dict[str, Any] = {}
    dashes: dict[str, Any] = {}
    linewidths: dict[str, float] = {}
    for idx, method in enumerate(methods_in_plot):
        palette[method] = styles.method_colors.get(
            method, fallback_colors[idx % len(fallback_colors)]
        )
        markers[method] = styles.method_markers.get(
            method, fallback_markers[idx % len(fallback_markers)]
        )
        dashes[method] = styles.method_dashes.get(
            method, fallback_dashes[idx % len(fallback_dashes)]
        )
        linewidths[method] = styles.line_width.get(method, 3)
    return palette, markers, dashes, linewidths


def finalize_axes(
    ax,
    title: str,
    y_label: str = "Point-to-mesh loss",
    label_fontsize: int = 19,
    tick_fontsize: int = 16,
    boarder_linewidth: int = 3,
    border_color: str = "darkgray",
):
    plt.xlabel("Timestep", fontdict={"size": label_fontsize})
    plt.ylabel(y_label, fontdict={"size": label_fontsize})
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=tick_fontsize,
        colors=border_color,
        labelcolor="black",
    )
    ax.tick_params(
        axis="both",
        which="minor",
        labelsize=tick_fontsize,
        colors=border_color,
        labelcolor="black",
    )
    ax.yaxis.get_offset_text().set_fontsize(tick_fontsize)

    if not ax.lines:
        return

    plt.gca().spines["bottom"].set_linewidth(boarder_linewidth)
    plt.gca().spines["left"].set_linewidth(boarder_linewidth)
    plt.gca().spines["top"].set_linewidth(boarder_linewidth)
    plt.gca().spines["right"].set_linewidth(boarder_linewidth)
    plt.gca().spines["right"].set_color(border_color)
    plt.gca().spines["top"].set_color(border_color)
    plt.gca().spines["bottom"].set_color(border_color)
    plt.gca().spines["left"].set_color(border_color)
    plt.title(title, fontdict={"size": label_fontsize})


def translate_legend(ax, label_fontsize: int = 19, tick_fontsize: int = 16):
    legend = ax.get_legend()
    if legend is None:
        return

    for text in legend.get_texts():
        original_label = text.get_text()
        translated_label = names.method_names.get(original_label, original_label)
        text.set_text(translated_label)
        text.set_fontsize(tick_fontsize)
    legend.set_title("Methods")
    legend.get_title().set_fontsize(label_fontsize)


def aggregate_time_over_trajectories(
    time_data: pd.DataFrame, group_cols: list[str]
) -> pd.DataFrame:
    return (
        time_data.groupby(group_cols + ["method", "seed", "timestep"], dropna=False)
        .agg(loss=("loss", "mean"), n_trajectories=("loss", "count"))
        .reset_index()
    )


def plot_time_series_with_ci(
    time_data: pd.DataFrame,
    out_path: Path,
    title: str,
    figure_width: int = 8,
    figure_height: int = 4,
    max_timestep: int | None = None,
    use_log_scale: bool = True,
):
    if time_data.empty:
        return

    if max_timestep is not None:
        time_data = time_data[time_data["timestep"] <= max_timestep].copy()
        if time_data.empty:
            return

    methods_in_plot = list(time_data["method"].drop_duplicates())
    palette, markers, dashes, linewidths = build_method_style(methods_in_plot)

    plt.close()
    configure_plot_style(figure_width=figure_width, figure_height=figure_height)
    ax = sns.lineplot(
        data=time_data,
        x="timestep",
        y="loss",
        hue="method",
        style="method",
        hue_order=methods_in_plot,
        style_order=methods_in_plot,
        palette=palette,
        markers=markers,
        markersize=10,
        dashes=dashes,
        linewidth=2,
        errorbar="ci",
        estimator=trim_mean_estimator,
        err_style="band",
        err_kws={"alpha": 0.3},
    )

    for line, method in zip(ax.lines, methods_in_plot):
        line.set_linewidth(linewidths[method])

    if use_log_scale and (time_data["loss"] > 0).all():
        plt.yscale("log")

    if max_timestep is not None:
        plt.xlim(time_data["timestep"].min(), max_timestep)
    timestep_ticks = sorted(time_data["timestep"].unique())
    ax.set_xticks(timestep_ticks)

    finalize_axes(ax, title=title)
    translate_legend(ax)
    plt.tight_layout()
    if out_path.suffix == ".tex":
        import matplot2tikz
        matplot2tikz.save(
            str(out_path),
            axis_width=f"{figure_width}cm",
            axis_height=f"{figure_height}cm",
            strict=False,
        )
    else:
        plt.savefig(out_path, dpi=200)


def plot_loss_over_time(
    time_data: pd.DataFrame,
    out_dir: Path,
    method_plot_max_timestep: int | None = None,
    method_plot_use_log_scale: bool = True,
):
    overall_seed_aggregated = aggregate_time_over_trajectories(time_data, group_cols=[])
    write_json(
        overall_seed_aggregated,
        out_dir / "loss_over_time_per_seed_trajectory_aggregated.json",
    )
    overall_plot_title = "Mean loss over time across all trajectories"
    if method_plot_max_timestep is not None:
        overall_plot_title += f" (timesteps 0-{method_plot_max_timestep})"
    plot_time_series_with_ci(
        overall_seed_aggregated,
        out_dir / "loss_over_time_per_method.tex",
        title=overall_plot_title,
        max_timestep=method_plot_max_timestep,
        use_log_scale=method_plot_use_log_scale,
    )

    per_material_seed_aggregated = aggregate_time_over_trajectories(
        time_data, group_cols=["material_combination", "latex", "sphere_id"]
    )
    write_json(
        per_material_seed_aggregated,
        out_dir
        / "loss_over_time_per_material_combination_seed_trajectory_aggregated.json",
    )

    per_material_dir = out_dir / "loss_over_time_per_material_combination"
    per_material_dir.mkdir(parents=True, exist_ok=True)
    grouped = per_material_seed_aggregated.groupby(
        ["material_combination", "latex", "sphere_id"], dropna=False
    )
    for (material_combination, latex, sphere_id), material_data in grouped:
        plot_time_series_with_ci(
            material_data,
            per_material_dir / f"{slugify(str(material_combination))}.tex",
            title=f"Mean loss over time for Latex {latex}, Sphere {sphere_id}",
        )


def plot_bar(summary: pd.DataFrame, x_col: str, y_col: str, out_path: Path, title: str):
    plt.close()
    fig, ax = plt.subplots(figsize=(8, 4))
    summary = summary.sort_values(x_col)
    x = range(len(summary))
    yerr = summary[f"{y_col}_std"] if f"{y_col}_std" in summary else None
    ax.bar(x, summary[y_col], yerr=yerr, capsize=3)
    ax.set_xticks(list(x))
    ax.set_xticklabels(summary[x_col].astype(str), rotation=30, ha="right")
    ax.set_ylabel(y_col)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "results_root",
        type=Path,
        help="Directory containing saved real-world eval results.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Where to write CSV summaries and plots. Defaults to results_root/aggregated.",
    )
    parser.add_argument(
        "--method-plot-max-timestep",
        type=int,
        default=None,
        help="Cap the overall per-method time plot at this timestep.",
    )
    parser.add_argument(
        "--method-plot-linear-scale",
        action="store_true",
        help="Use a linear y-axis for the overall per-method time plot.",
    )
    args = parser.parse_args()

    out_dir = args.out_dir or args.results_root / "aggregated"
    out_dir.mkdir(parents=True, exist_ok=True)

    data, time_data = load_result_files(args.results_root)
    summaries = save_summaries(data, out_dir)
    write_json(time_data, out_dir / "loss_over_time.json")

    plot_bar(
        summaries["per_seed"],
        "seed",
        "loss_mean",
        out_dir / "loss_mean_per_seed.png",
        "Mean loss per seed",
    )
    plot_bar(
        summaries["per_material"],
        "material_id",
        "loss_mean",
        out_dir / "loss_mean_per_material.png",
        "Mean loss per material",
    )
    plot_bar(
        summaries["per_trajectory"],
        "target_traj_idx",
        "loss_mean",
        out_dir / "loss_mean_per_trajectory.png",
        "Mean loss per target trajectory",
    )
    plot_loss_over_time(
        time_data,
        out_dir,
        method_plot_max_timestep=args.method_plot_max_timestep,
        method_plot_use_log_scale=not args.method_plot_linear_scale,
    )

    print(
        f"Loaded per-trajectory .pt results and wrote derived summaries/plots to {out_dir}"
    )


if __name__ == "__main__":
    main()
