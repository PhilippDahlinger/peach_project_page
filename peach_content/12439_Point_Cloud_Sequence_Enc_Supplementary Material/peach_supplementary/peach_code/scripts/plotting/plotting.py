import os
from typing import List

import matplotlib
import numpy as np
import pandas as pd
import pylab
import scipy
import seaborn as sns
import matplot2tikz
import torch
from matplotlib import pyplot as plt
from matplotlib.ticker import LogLocator, AutoMinorLocator, ScalarFormatter
from omegaconf import OmegaConf

from scripts.plotting.configs import styles, names
from scripts.plotting.configs.names import env_names, metric_names


class MagnitudeFormatter(matplotlib.ticker.ScalarFormatter):
    def __init__(self, exponent=None):
        super().__init__()
        self._fixed_exponent = exponent

    def _set_order_of_magnitude(self):
        if self._fixed_exponent:
            self.orderOfMagnitude = self._fixed_exponent
        else:
            super()._set_order_of_magnitude()

    def _set_format(self):
        self.format = "%1.1f"


def bar_plot(data: dict,
          figure_width: int = 6,
          figure_height: int = 5,
          mode: str = "show",
          unique_name: str = None,
          formatter_magnitude: int = -4,
          use_major_formatter: bool = False,
          use_minor_formatter: bool = False,
          y_top_limit: float | None = None,
          y_bottom_limit: float | None = None,
          legend_order: List[int] | None = None,
          label_fontsize: int = 19,
          tick_fontsize: int = 16,
          linewidth: int = 3,
          boarder_linewidth: int = 3,
          border_color='darkgray',
          show_legend: bool = False,
          overwrite: bool = False,
          exclude_seeds=None,):
    """Bar plot: for each method, picks the context size with the lowest mean
    loss and plots it as a bar. CI shown as black error bars. Y-axis is log
    scale. Method labels are written below bars at 45 degrees."""

    # ------------------------------------------------------------------ #
    # 1. Load data  (identical to plot())                               #
    # ------------------------------------------------------------------ #
    df = {
        "env": [], "method": [], "context": [],
        "seed": [], "mat_prop_loss": [], "ml_loss": [],
    }
    for method_name, path in data.items():
        for seed in os.listdir(path):
            if exclude_seeds is not None and seed in exclude_seeds:
                continue
            output_path = os.path.join(path, seed)
            meta_data = OmegaConf.load(os.path.join(output_path, "meta_data.yaml"))
            mat_prop_losses = torch.load(os.path.join(output_path, "test_mat_prop_loss.pt"))
            try:
                ml_losses = torch.load(os.path.join(output_path, "test_ml_loss.pt"))
            except FileNotFoundError:
                ml_losses = [torch.tensor(0.0)] * len(mat_prop_losses)
            for context_idx, (mat_prop_loss, ml_loss) in enumerate(
                    zip(mat_prop_losses, ml_losses)):
                df["env"].append(meta_data.env_name)
                df["method"].append(method_name)
                df["context"].append(context_idx + 1)
                df["seed"].append(seed)
                df["mat_prop_loss"].append(mat_prop_loss.item())
                df["ml_loss"].append(ml_loss.item())

    assert len(set(df["env"])) == 1
    env_name = df["env"][0]
    methods  = list(data.keys())
    data_df  = pd.DataFrame(df)

    metrics = ["ml_loss", "mat_prop_loss"]
    for metric in metrics:

        plt.close()

        # filter oracle for mat_prop_loss (same as plot())
        if metric == "mat_prop_loss":
            metric_df = data_df[data_df["method"] != "oracle_encoder_mango_decoder"].copy()
        else:
            metric_df = data_df.copy()

        # ------------------------------------------------------------------ #
        # 2. For every method: IQM per context, pick best (lowest) context  #
        # ------------------------------------------------------------------ #
        method_order = (
            [m for m in methods if m != "oracle_encoder_mango_decoder"]
            if metric == "mat_prop_loss"
            else methods
        )

        bar_heights = []   # IQM of the best context
        ci_lo       = []   # lower CI bound (for error bar, distance from bar)
        ci_hi       = []   # upper CI bound
        bar_colors  = []

        for method in method_order:
            mdf = metric_df[metric_df["method"] == method]

            # IQM per context across seeds
            ctx_stats = (
                mdf.groupby("context")[metric]
                .agg(
                    iqm=lambda x: scipy.stats.trim_mean(x, proportiontocut=0.0),
                    # 95 % bootstrap CI endpoints
                    ci_low=lambda x: np.percentile(
                        [np.mean(np.random.choice(x, size=len(x), replace=True))
                         for _ in range(1000)], 2.5),
                    ci_high=lambda x: np.percentile(
                        [np.mean(np.random.choice(x, size=len(x), replace=True))
                         for _ in range(1000)], 97.5),
                )
                .reset_index()
            )

            best_row = ctx_stats.loc[ctx_stats["iqm"].idxmin()]
            bar_heights.append(best_row["iqm"])
            ci_lo.append(best_row["iqm"] - best_row["ci_low"])
            ci_hi.append(best_row["ci_high"] - best_row["iqm"])
            bar_colors.append(styles.method_colors.get(method, "steelblue"))

        # ------------------------------------------------------------------ #
        # 3. Draw bars                                                       #
        # ------------------------------------------------------------------ #
        sns.set(rc={'figure.figsize': (figure_width, figure_height)})
        sns.set_theme()
        sns.set_style("whitegrid")
        plt.rcParams.update({"ytick.left": True})

        fig, ax = plt.subplots()

        x_pos = np.arange(len(method_order))
        bars = ax.bar(
            x_pos,
            bar_heights,
            color=bar_colors,
            width=0.6,
            zorder=3,
        )

        # asymmetric error bars (black, cap visible)
        ax.errorbar(
            x_pos,
            bar_heights,
            yerr=[ci_lo, ci_hi],
            fmt="none",
            ecolor="black",
            elinewidth=1.5,
            capsize=5,
            capthick=1.5,
            zorder=4,
        )

        # ------------------------------------------------------------------ #
        # 4. Axes, labels, appearance                                        #
        # ------------------------------------------------------------------ #
        ax.set_yscale("log")

        if y_top_limit is not None:
            ax.set_ylim(top=y_top_limit)
        if y_bottom_limit is not None:
            ax.set_ylim(bottom=y_bottom_limit)

        fmt_obj = MagnitudeFormatter(formatter_magnitude)
        if use_major_formatter:
            ax.yaxis.set_major_formatter(fmt_obj)
        if use_minor_formatter:
            ax.yaxis.set_minor_formatter(fmt_obj)

        # x ticks: translated method names, 45 deg
        method_names = names.method_names
        x_labels = [method_names.get(m, m) for m in method_order]
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, rotation=45, ha="right",
                           fontsize=tick_fontsize)

        ax.set_ylabel(names.metric_names[metric], fontdict={'size': label_fontsize})
        ax.set_xlabel("")   # method names serve as x-label

        ax.tick_params(axis='y', which='major', labelsize=tick_fontsize,
                       colors=border_color, labelcolor="black")
        ax.tick_params(axis='y', which='minor', labelsize=tick_fontsize,
                       colors=border_color, labelcolor="black")
        ax.yaxis.get_offset_text().set_fontsize(tick_fontsize)

        # border
        for spine in ax.spines.values():
            spine.set_linewidth(boarder_linewidth)
            spine.set_color(border_color)

        ax.set_title(env_names[env_name], fontdict={'size': label_fontsize})
        # plt.title("Deformable Block (OOD)", fontdict={'size': label_fontsize})


        plt.tight_layout()

        # ------------------------------------------------------------------ #
        # 5. Save / show                                                     #
        # ------------------------------------------------------------------ #
        if mode in ("pdf", "tikz"):
            if unique_name is not None:
                file_name = unique_name
            else:
                method_filename = "__".join(sorted(methods))
                file_name = f"bar_{metric}___{method_filename}"
            out_path = (f"output/figures/quantitative/{env_name}/"
                        f"{metric}/bar_{file_name}.pdf")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            if os.path.isfile(out_path) and not overwrite:
                print(f"File {out_path} already exists. Skipping...")
            else:
                if mode == "pdf":
                    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
                elif mode == "tikz":
                    out_path = out_path.replace(".pdf", ".tex")
                    matplot2tikz.save(out_path)
                    fix_tikz_bar_plot(out_path)
                    print(f"Saved figure to {out_path}")
                print(f"Saved figure to {out_path}")
        elif mode == "show":
            plt.show()
        elif mode == "csv":
            import csv
            if unique_name is not None:
                file_name = unique_name
            else:
                method_filename = "__".join(sorted(methods))
                file_name = f"bar_{metric}___{method_filename}"
            out_path = (f"output/figures/quantitative/{env_name}/"
                        f"{metric}/bar_{file_name}.csv")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["method", "color", "bar_height", "ci_low", "ci_high"])
                writer.writeheader()
                for method, color, height, lo, hi in zip(method_order, bar_colors, bar_heights, ci_lo, ci_hi):
                    print(f"Method: {method}, Height: {height:.4e}, CI Low: {height - lo:.10e}, CI High: {height + hi:.10e}")
                    writer.writerow({
                        "method": method,
                        "color": color,
                        "bar_height": height,
                        "ci_low": height - lo,  # absolute lower bound
                        "ci_high": height + hi,  # absolute upper bound
                    })
            print(f"Saved CSV to {out_path}")
        else:
            raise ValueError(f"Unknown mode {mode}")

import re

def fix_tikz_bar_plot(tex_path: str) -> None:
    """Fix matplot2tikz bar plot output in-place:
    - Convert axis environment to semilogyaxis with log origin=infty.
    """
    with open(tex_path, "r") as f:
        content = f.read()

    # 1. Replace \begin{axis} / \end{axis} with semilogyaxis
    content = content.replace(r'\begin{axis}[', r'\begin{semilogyaxis}[')
    content = content.replace(r'\end{axis}', r'\end{semilogyaxis}')

    # 2. Inject log origin=infty
    content = content.replace(
        r'\begin{semilogyaxis}[',
        r'\begin{semilogyaxis}[' + '\n  log origin=infty,'
    )

    # 3. Remove ymode=log and log basis y={10} since semilogyaxis handles these
    # content = re.sub(r'ymode=log,\n', '', content)
    # content = re.sub(r'log basis y=\{10\},\n', '', content)

    with open(tex_path, "w") as f:
        f.write(content)

def plot(data: dict,
         figure_width: int = 4,
         figure_height: int = 5,
         mode: str = "show",
         unique_name: str = None,
         formatter_magnitude: int = -4,
         use_major_formatter: bool = False,
         use_minor_formatter: bool = False,
         y_top_limit: float | None = None,
         y_bottom_limit: float | None = None,
         legend_order: List[int] | None = None,
         label_fontsize: int = 19,
         tick_fontsize: int = 16,
         linewidth: int = 3,
         boarder_linewidth: int = 3,
         border_color='darkgray',
         show_legend: bool = False,
         overwrite: bool = False,
         exclude_seeds=None,):

    # load data
    df = {
        "env": [],
        "method": [],
        "context": [],
        "seed": [],
        "mat_prop_loss": [],
        "ml_loss": [],
    }
    for method_name, path in data.items():
        for seed in os.listdir(path):
            if exclude_seeds is not None and seed in exclude_seeds:
                continue
            output_path = os.path.join(path, seed)
            meta_data = os.path.join(output_path, "meta_data.yaml")
            # load
            meta_data = OmegaConf.load(meta_data)
            mat_prop_loss = os.path.join(output_path, "test_mat_prop_loss.pt")
            mat_prop_losses = torch.load(mat_prop_loss)
            try:
                ml_losses = os.path.join(output_path, "test_ml_loss.pt")
                ml_losses = torch.load(ml_losses)
            except FileNotFoundError:
                # in case there is no ml loss, set to zero
                ml_losses = [torch.tensor(0.0)] * len(mat_prop_losses)
            for context_idx, (mat_prop_loss, ml_loss) in enumerate(zip(mat_prop_losses, ml_losses)):
                context_size = context_idx + 1
                # if context_size > 4:
                #     continue
                df["env"].append(meta_data.env_name)
                df["method"].append(method_name)
                df["context"].append(context_size)
                df["seed"].append(seed)
                df["mat_prop_loss"].append(mat_prop_loss.item())
                df["ml_loss"].append(ml_loss.item())
    # assert all methods have the same env
    assert len(set(df["env"])) == 1
    env_name = df["env"][0]
    methods = data.keys()
    data = pd.DataFrame(df)

    # metrics = ["mat_prop_loss", "ml_loss"]
    metrics = ["ml_loss", "mat_prop_loss"]
    for metric in metrics:
        # close previous plot if present
        plt.close()
        # call seaborn
        sns.set(rc={'figure.figsize': (figure_width, figure_height)})
        sns.set_theme()
        sns.set_style("whitegrid")
        plt.rcParams.update({"ytick.left": True})
        # if metric is mat prop loss: filter out oracle method
        if metric == "mat_prop_loss":
            metric_data = data[data["method"] != "oracle_encoder_mango_decoder"]
        else:
            metric_data = data
        ax = sns.lineplot(data=metric_data, x="context", y=metric,
                     hue="method",
                     style="method",
                     palette=styles.method_colors,
                     markers=styles.method_markers,
                     markersize=10,
                     dashes=styles.method_dashes,
                     linewidth=2,
                     # errorbar=("pi", 75),
                     errorbar="ci",
                     # n_boot=1000,
                     estimator=lambda scores: scipy.stats.trim_mean(scores, proportiontocut=0.0, axis=None),  # IQM
                     )

        for line, method in zip(ax.lines, data["method"].unique()):
            line.set_linewidth(styles.line_width[method])

        ax = plt.gca()

        # # ticks and labels appearance
        # x_labels = data["context"].unique()
        # # only get numbers from context
        # x_labels = [int(x) for x in x_labels]
        # # remove trailing zeros
        # x_labels = [str(int(x)) for x in x_labels]

        # plt.xticks(ticks=range(len(x_labels)), labels=x_labels)
        plt.xlabel("Context Size", fontdict={'size': label_fontsize})
        plt.ylabel(names.metric_names[metric], fontdict={'size': label_fontsize})

        # Set the fontsize  and colors for the numbers on the ticks and the offset text.
        ax.tick_params(axis='both', which='major', labelsize=tick_fontsize, colors=border_color, labelcolor="black")
        ax.tick_params(axis='both', which='minor', labelsize=tick_fontsize, colors=border_color, labelcolor="black")
        ax.yaxis.get_offset_text().set_fontsize(tick_fontsize)

        # Ticks number formatting and y scale
        plt.yscale('log')
        fmt = MagnitudeFormatter(formatter_magnitude)
        if use_major_formatter:
            ax.yaxis.set_major_formatter(fmt)
        if use_minor_formatter:
            ax.yaxis.set_minor_formatter(fmt)

        # y limits
        if y_top_limit is not None:
            plt.ylim(top=y_top_limit)
        if y_bottom_limit is not None:
            plt.ylim(bottom=y_bottom_limit)

        # boarder
        plt.gca().spines['bottom'].set_linewidth(boarder_linewidth)
        plt.gca().spines['left'].set_linewidth(boarder_linewidth)
        plt.gca().spines['top'].set_linewidth(boarder_linewidth)
        plt.gca().spines['right'].set_linewidth(boarder_linewidth)
        plt.gca().spines['right'].set_color(border_color)
        plt.gca().spines['top'].set_color(border_color)
        plt.gca().spines['bottom'].set_color(border_color)
        plt.gca().spines['left'].set_color(border_color)

        # title
        plt.title(env_names[env_name], fontdict={'size': label_fontsize})
        # plt.title("Deformable Block (OOD)", fontdict={'size': label_fontsize})

        # legend
        if show_legend:
            # Get the handles and labels of the current axes
            if legend_order is not None:
                handles, labels = plt.gca().get_legend_handles_labels()
                legend = plt.legend([handles[idx] for idx in legend_order], [labels[idx] for idx in legend_order])
            else:
                legend = plt.legend()

            # Translate the labels using the dictionary
            method_names = names.method_names
            for text in legend.get_texts():
                original_label = text.get_text()
                translated_label = method_names.get(original_label, original_label)
                text.set_text(translated_label)
                text.set_fontsize(tick_fontsize)
            # Set the legend title
            legend.set_title('Methods')
            legend.get_title().set_fontsize(label_fontsize)
        else:
            ax.get_legend().remove()

        # save or show
        if mode == "pdf" or mode == "tikz":
            if unique_name is not None:
                file_name = unique_name
            else:
                method_filename = "__".join(sorted(methods))
                file_name = f"{metric}___{method_filename}"
            out_path = f"output/figures/quantitative/{env_name}/{metric}/{file_name}.pdf"
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            # check if file exists and only write if overwrite is true
            if os.path.isfile(out_path) and not overwrite:
                print(f"File {out_path} already exists. Skipping...")
            else:
                # set tight layout explicit in savefig
                if mode == "pdf":
                    plt.savefig(out_path, bbox_inches='tight', pad_inches=0, )
                elif mode == "tikz":
                    out_path = out_path.replace(".pdf", ".tex")
                    matplot2tikz.save(out_path)
                print(f"Saved figure to {out_path}")
        elif mode == "show":
            plt.show()
        else:
            raise ValueError(f"Unknown mode {mode}")


def db(show=False):
    data = {
        "pointpatch_encoder_mango_decoder": "evaluations/paper_exp/db/5405_db_pointpatch_encoder_mango_decoder/+ex_exp=54_pointpatch_encoder_mango_decoder,+pl=lichtenberg",
        "oracle_encoder_mango_decoder": "evaluations/paper_exp/db/5601_db_oracle_mango_decoder/+ex_exp=56_oracle_mango_decoder,+pl=horeka_h100",
        "mango_encoder_mango_decoder": "evaluations/paper_exp/db/5/exp5403_dp_easy_v5_ml_mgno/+ex_exp_dee_mgn=dp_easy_v5,+pl=horeka_gpu",
        "dummy_encoder_mango_decoder": "evaluations/paper_exp/db/5201_db_dummy_mango_decoder/+ex_exp=52_dummy_mango_decoder,+pl=horeka_h100",
        "mgn": "evaluations/paper_exp/db/5001_db_mgn/+ex_exp=50_mgn,+pl=horeka_h100",
        "mgn_oracle": "evaluations/paper_exp/db/5101_db_oracle_mgn/+ex_exp=51_oracle_mgn,+pl=horeka_h100",
        "pstnet_encoder_mango_decoder": "evaluations/paper_exp/db/5501_db_pstnet_mango_decoder/+ex_exp=55_pstnet_mango_decoder,+pl=horeka_h100",
        "gnn_encoder_mango_decoder": "evaluations/paper_exp/db/5704_db_gnn_mango_decoder/+ex_exp=57_gnn_mango_decoder,+pl=lichtenberg_all",


        # "test_opt": "evaluations/paper_exp/db/6201_db_test_opt_mango_decoder/+ex_exp=56_oracle_mango_decoder,+pl=horeka_h100"
    }
    if show:
        plot(data, mode="show", unique_name="db", show_legend=True)
        bar_plot(data, mode="show", unique_name="db", show_legend=True)
    plt.close()
    plot(data, mode="pdf", unique_name="db_context", overwrite=True)
    bar_plot(data, mode="csv", unique_name="db_bar", overwrite=True)
    plt.close()
    # plot(data, mode="tikz", unique_name="db_sim", overwrite=True)
    # plt.close()


def db_ablation(show=False):
    data = {
        "pointpatch_encoder_mango_decoder": "evaluations/paper_exp/db/5405_db_pointpatch_encoder_mango_decoder/+ex_exp=54_pointpatch_encoder_mango_decoder,+pl=lichtenberg",
        "peach_only_ml": "evaluations/paper_exp/db/5902_db_pointpatch_encoder_mango_decoder/+ex_exp=59_pointpatch_encoder_mango_decoder_only_ml,+pl=lichtenberg_all",
        "peach_only_ml_and_matprop": "evaluations/paper_exp/db/6002_db_pointpatch_encoder_mango_decoder/+ex_exp=60_pointpatch_encoder_mango_decoder_only_ml_and_matprop,+pl=lichtenberg",
        "peach_only_ml_and_sdf": "evaluations/paper_exp/db/6102_db_pointpatch_encoder_mango_decoder/+ex_exp=61_pointpatch_encoder_mango_decoder_only_ml_and_sdf,+pl=lichtenberg_a100",
        "oracle_encoder_mango_decoder": "evaluations/paper_exp/db/5601_db_oracle_mango_decoder/+ex_exp=56_oracle_mango_decoder,+pl=horeka_h100",
        # "dummy_encoder_mango_decoder": "evaluations/paper_exp/db/5201_db_dummy_mango_decoder/+ex_exp=52_dummy_mango_decoder,+pl=horeka_h100",


    }
    if show:
        plot(data, mode="show", unique_name="db_ablation", show_legend=True)
        bar_plot(data, mode="show", unique_name="db_ablation_bar", show_legend=True)
    plt.close()
    plot(data, mode="tikz", unique_name="db_ablation_context", overwrite=True)
    bar_plot(data, mode="csv", unique_name="db_ablation_bar", overwrite=True)
    plt.close()
    # plot(data, mode="tikz", unique_name="db_sim", overwrite=True)
    # plt.close()

def sd_ablation(show=False):
    data = {
        "pointpatch_encoder_mango_decoder": "evaluations/paper_exp/sd/5401_sd_pointpatch_encoder_mango_decoder/+ex_exp=54_pointpatch_encoder_mango_decoder,+pl=bwuni3",
        "peach_only_ml": "evaluations/paper_exp/sd/5902_sd_pointpatch_encoder_mango_decoder_only_ml/+ex_exp=59_pointpatch_encoder_mango_decoder_only_ml,+pl=bwuni3",
        "peach_only_ml_and_matprop": "evaluations/paper_exp/sd/6003d2_sd_pointpatch_encoder_mango_decoder_only_ml_and_matprop/+ex_exp=60_pointpatch_encoder_mango_decoder_only_ml_and_matprop,+pl=jsc_clusterduck",
        "peach_only_ml_and_sdf": "evaluations/paper_exp/sd/6103d2_sd_pointpatch_encoder_mango_decoder_only_ml_and_sdf/+ex_exp=61_pointpatch_encoder_mango_decoder_only_ml_and_sdf,+pl=jsc_clusterduck",
        "oracle_encoder_mango_decoder": "evaluations/paper_exp/sd/5603d2_sd_oracle_mango_decoder/+ex_exp=56_oracle_mango_decoder,+pl=jsc_clusterduck",
        # "dummy_encoder_mango_decoder": "evaluations/paper_exp/sd/5203d2_sd_dummy_mango_decoder/+ex_exp=52_dummy_mango_decoder,+pl=jsc_clusterduck",

    }
    if show:
        plot(data, mode="show", unique_name="sd_ablation", show_legend=True)
        bar_plot(data, mode="show", unique_name="sd_ablation_bar", show_legend=True)
    plt.close()
    plot(data, mode="tikz", unique_name="sd_ablation_context", overwrite=True)
    bar_plot(data, mode="csv", unique_name="sd_ablation_bar", overwrite=True)
    plt.close()
    # plot(data, mode="tikz", unique_name="db_sim", overwrite=True)
    # plt.close()


def db_ood(show=False):
    data = {
        "pointpatch_encoder_mango_decoder": "evaluations/paper_exp/db_ood/5406_db_ood_pointpatch_encoder_mango_decoder/+ex_exp_ood=54_pointpatch_encoder_mango_decoder,+pl=lichtenberg_all",
        "oracle_encoder_mango_decoder": "evaluations/paper_exp/db_ood/5604_db_ood_oracle_mango_decoder/+ex_exp_ood=56_oracle_mango_decoder,+pl=lichtenberg_all",
        "dummy_encoder_mango_decoder": "evaluations/paper_exp/db_ood/5204_db_ood_dummy_mango_decoder/+ex_exp_ood=52_dummy_mango_decoder,+pl=lichtenberg_all",

    }
    if show:
        plot(data, mode="show", unique_name="db_ood", show_legend=True)
        bar_plot(data, mode="show", unique_name="db_ood_bar", show_legend=True)
    plt.close()
    plot(data, mode="pdf", unique_name="db_ood_context", overwrite=True)
    bar_plot(data, mode="csv", unique_name="db_ood_bar", overwrite=True)
    plt.close()
    # plot(data, mode="tikz", unique_name="db_sim", overwrite=True)
    # plt.close()


def sd(show=False):
    data = {
        "pointpatch_encoder_mango_decoder": "evaluations/paper_exp/sd/5401_sd_pointpatch_encoder_mango_decoder/+ex_exp=54_pointpatch_encoder_mango_decoder,+pl=bwuni3",
        "oracle_encoder_mango_decoder": "evaluations/paper_exp/sd/5603d2_sd_oracle_mango_decoder/+ex_exp=56_oracle_mango_decoder,+pl=jsc_clusterduck",
        "mango_encoder_mango_decoder": "evaluations/paper_exp/sd/exp5401_planar_bending_v3_ml_mgno/+ex_exp_dee_mgn=planar_bending_v3,+pl=horeka_gpu",
        "dummy_encoder_mango_decoder": "evaluations/paper_exp/sd/5203d2_sd_dummy_mango_decoder/+ex_exp=52_dummy_mango_decoder,+pl=jsc_clusterduck",
        "mgn": "evaluations/paper_exp/sd/5001_sd_mgn/+ex_exp=50_mgn,+pl=bwuni3",
        "mgn_oracle": "evaluations/paper_exp/sd/5101_sd_oracle_mgn/+ex_exp=51_oracle_mgn,+pl=bwuni3",
        "pstnet_encoder_mango_decoder": "evaluations/paper_exp/sd/5501_sd_pstnet_mango_decoder/+ex_exp=55_pstnet_mango_decoder,+pl=bwuni3",
        "gnn_encoder_mango_decoder": "evaluations/paper_exp/sd/5701_sd_gnn_mango_decoder/+ex_exp=57_gnn_mango_decoder,+pl=bwuni3",
        # "test_opt": "evaluations/paper_exp/sd/6203d2_sd_test_opt_mango_decoder/+ex_exp=56_oracle_mango_decoder,+pl=jsc_clusterduck"
    }
    if show:
        plot(data, mode="show", unique_name="sd", show_legend=True)
        bar_plot(data, mode="show", unique_name="sd", show_legend=True)
    plt.close()
    plot(data, mode="pdf", unique_name="sd_context", overwrite=True)
    bar_plot(data, mode="csv", unique_name="sd_bar", overwrite=True)
    plt.close()
    # plot(data, mode="tikz", unique_name="db_sim", overwrite=True)
    # plt.close()


def trampoline_v4_sim(show=False):
    data = {
        "pointpatch_encoder_mango_decoder": "evaluations/paper_exp/trampoline_v4/5403d2_trampoline_v4_pointpatch_encoder_mango_decoder/+ex_exp_v4=54_pointpatch_encoder_mango_decoder,+pl=jsc_clusterduck",
        "oracle_encoder_mango_decoder": "evaluations/paper_exp/trampoline_v4/5603d2_trampoline_v4_oracle_mango_decoder/+ex_exp_v4=56_oracle_mango_decoder,+pl=jsc_clusterduck",
        "mango_encoder_mango_decoder": "evaluations/paper_exp/trampoline_v4/5303d2_trampoline_v4_cnn_deepset_mango_decoder/+ex_exp_v4=53_cnn_deepset_mango_decoder,+pl=jsc_clusterduck",
        "dummy_encoder_mango_decoder": "evaluations/paper_exp/trampoline_v4/5203d2_trampoline_v4_dummy_mango_decoder/+ex_exp_v4=52_dummy_mango_decoder,+pl=jsc_clusterduck",
        "mgn": "evaluations/paper_exp/trampoline_v4/5004d2_trampoline_v4_mgn/+ex_exp_v4=50_mgn,+pl=jsc_clusterduck",
        "mgn_oracle": "evaluations/paper_exp/trampoline_v4/5103d2_trampoline_v4_oracle_mgn/+ex_exp_v4=51_oracle_mgn,+pl=jsc_clusterduck",

        "pstnet_encoder_mango_decoder": "evaluations/paper_exp/trampoline_v4/5503_trampoline_v4_pstnet_mango_decoder/+ex_exp_v4=55_pstnet_mango_decoder,+pl=lichtenberg",
        "gnn_encoder_mango_decoder": "evaluations/paper_exp/trampoline_v4/5703d2_trampoline_v4_gnn_mango_decoder/+ex_exp_v4=57_gnn_mango_decoder,+pl=jsc_clusterduck",
        # "test_opt": "evaluations/paper_exp/trampoline_v4/6201_trampoline_v4_test_opt_mango_decoder/+ex_exp_v4=56_oracle_mango_decoder,+pl=jsc_clusterduck"

    }
    if show:
        bar_plot(data, mode="show", unique_name="trampoline_v4_sim_bar", show_legend=True)
        plot(data, mode="show", unique_name="trampoline_v4_sim_context", show_legend=True)
    plt.close()
    plot(data, mode="pdf", unique_name="trampoline_v4_sim_context", overwrite=True)
    bar_plot(data, mode="csv", unique_name="trampoline_v4_sim_bar", overwrite=True)

    plt.close()
    # plot(data, mode="tikz", unique_name="db_sim", overwrite=True)
    # plt.close()


def bbv(show=False):
    data = {
        "pointpatch_encoder_mango_decoder": "evaluations/paper_exp/bbv/6401_bbv_pointpatch_encoder_mango_decoder_low_mem/+ex_exp=64_pointpatch_encoder_mango_decoder_low_mem,+pl=jsc_clusterduck",
        "oracle_encoder_mango_decoder": "evaluations/paper_exp/bbv/5604_bbv_oracle_mango_decoder/+ex_exp=56_oracle_mango_decoder,+pl=jsc_clusterduck",
        "mango_encoder_mango_decoder": "evaluations/paper_exp/bbv/5302_bbv_cnn_deepset_mango_decoder/+ex_exp=53_cnn_deepset_mango_decoder,+pl=jsc_clusterduck",
        "dummy_encoder_mango_decoder": "evaluations/paper_exp/bbv/5202_bbv_dummy_mango_decoder/+ex_exp=52_dummy_mango_decoder,+pl=jsc_clusterduck",


        "mgn": "evaluations/paper_exp/bbv/5002_bbv_mgn/+ex_exp=50_mgn,+pl=jsc_clusterduck",
        "mgn_oracle": "evaluations/paper_exp/bbv/5102_bbv_oracle_mgn/+ex_exp=51_oracle_mgn,+pl=jsc_clusterduck",
        "pstnet_encoder_mango_decoder": "evaluations/paper_exp/bbv/5501_bbv_pstnet_mango_decoder/+ex_exp=55_pstnet_mango_decoder,+pl=jsc_clusterduck",

        "gnn_encoder_mango_decoder": "evaluations/paper_exp/bbv/5702_bbv_gnn_mango_decoder/+ex_exp=57_gnn_mango_decoder,+pl=jsc_clusterduck",
        # "test_opt": "evaluations/paper_exp/bbv/6204_bbv_test_opt/+ex_exp=56_oracle_mango_decoder,+pl=jsc_clusterduck",

    }
    if show:
        bar_plot(data, mode="show", unique_name="bbv", show_legend=True)
        plot(data, mode="show", unique_name="bbv", show_legend=True)
    plt.close()
    plot(data, mode="pdf", unique_name="bbv_context", overwrite=True)
    bar_plot(data, mode="csv", unique_name="bbv_bar", overwrite=True,
             )
    plt.close()


if __name__ == "__main__":

    show = True
    # bbv(show)
    # db(show)
    # sd(show)
    # trampoline_v4_sim(show)
    # db_ood(show)
    db_ablation(show)
    sd_ablation(show)




