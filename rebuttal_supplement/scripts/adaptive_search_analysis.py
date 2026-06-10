from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SAMPLE_KEYS = ["speaker", "split", "utt_id", "target_word"]
DEFAULT_BUDGETS = [1, 3, 5, 10, 14]


@dataclass(frozen=True)
class Experiment:
    name: str
    result_csv: Path
    population_csv: Path


EXPERIMENTS = [
    Experiment(
        "10-word",
        Path("datasets/torgo_10word_wrong_subset_multi_results.csv"),
        Path("datasets/torgo_10word_split_clean.csv"),
    ),
    Experiment(
        "30-word",
        Path("datasets/torgo_30word_wrong_subset_multi_None_results.csv"),
        Path("datasets/torgo_30word_split_clean.csv"),
    ),
    Experiment(
        "50-word",
        Path("datasets/torgo_50word_wrong_subset_multi_None_results.csv"),
        Path("datasets/torgo_50word_split_clean.csv"),
    ),
    Experiment(
        "100-word",
        Path("datasets/torgo_100word_wrong_subset_multi_None_results.csv"),
        Path("datasets/torgo_100word_split_clean.csv"),
    ),
    Experiment(
        "Full lexicon",
        Path("datasets/torgo_fulllex_wrong_subset_multi_None_results.csv"),
        Path("datasets/torgo_single_word_headmic_split.csv"),
    ),
]


def config_label(epsilon: float, steps: int) -> str:
    return f"eps={epsilon:.5f},K={steps}"


def load_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = set(SAMPLE_KEYS + ["epsilon", "steps", "mapped_correct"]) - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    df = df.copy()
    df["epsilon"] = pd.to_numeric(df["epsilon"])
    df["steps"] = pd.to_numeric(df["steps"]).astype(int)
    df["mapped_correct"] = pd.to_numeric(df["mapped_correct"]).astype(int)
    df = df[(df["epsilon"] > 0) & (df["steps"] > 0)].copy()
    df["config"] = [
        config_label(epsilon, steps)
        for epsilon, steps in zip(df["epsilon"], df["steps"])
    ]
    df["sample_id"] = df[SAMPLE_KEYS].astype(str).agg("||".join, axis=1)

    duplicate_mask = df.duplicated(["sample_id", "config"], keep=False)
    if duplicate_mask.any():
        duplicates = df.loc[duplicate_mask, ["sample_id", "config"]].head()
        raise ValueError(f"{path} has duplicate sample/config rows:\n{duplicates}")
    return df


def load_population_counts(path: Path) -> dict[str, int]:
    df = pd.read_csv(path)
    if "split" not in df.columns:
        raise ValueError(f"{path} has no split column")
    return df.groupby("split").size().astype(int).to_dict()


def make_success_matrix(df: pd.DataFrame, split: str) -> pd.DataFrame:
    subset = df[df["split"] == split]
    matrix = subset.pivot(index="sample_id", columns="config", values="mapped_correct")
    if matrix.isna().any().any():
        missing = int(matrix.isna().sum().sum())
        raise ValueError(f"{split} success matrix has {missing} missing evaluations")
    return matrix.astype(bool)


def config_metadata(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df[["config", "epsilon", "steps"]]
        .drop_duplicates()
        .set_index("config")
        .sort_values(["epsilon", "steps"])
    )


def greedy_coverage_order(
    success: pd.DataFrame, metadata: pd.DataFrame
) -> tuple[list[str], pd.DataFrame]:
    remaining_configs = list(success.columns)
    uncovered = pd.Series(True, index=success.index)
    order: list[str] = []
    rows: list[dict[str, float | int | str]] = []

    while remaining_configs:
        candidates = []
        for config in remaining_configs:
            newly_covered = int((success[config] & uncovered).sum())
            total_success = int(success[config].sum())
            candidates.append(
                (
                    -newly_covered,
                    -total_success,
                    float(metadata.loc[config, "epsilon"]),
                    int(metadata.loc[config, "steps"]),
                    config,
                )
            )
        _, _, _, _, selected = min(candidates)
        marginal = int((success[selected] & uncovered).sum())
        uncovered &= ~success[selected]
        order.append(selected)
        remaining_configs.remove(selected)
        rows.append(
            {
                "position": len(order),
                "config": selected,
                "epsilon": float(metadata.loc[selected, "epsilon"]),
                "steps": int(metadata.loc[selected, "steps"]),
                "marginal_dev_repairs": marginal,
                "cumulative_dev_repairs": int((~uncovered).sum()),
                "cumulative_dev_repair_rate": float((~uncovered).mean()),
            }
        )
    return order, pd.DataFrame(rows)


def first_success_positions(success: pd.DataFrame, order: list[str]) -> np.ndarray:
    ordered = success[order].to_numpy(dtype=bool)
    any_success = ordered.any(axis=1)
    first = np.argmax(ordered, axis=1) + 1
    return np.where(any_success, first, len(order) + 1)


def bootstrap_proportion_ci(
    values: np.ndarray, rng: np.random.Generator, n_bootstrap: int
) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    samples = rng.choice(values, size=(n_bootstrap, values.size), replace=True)
    estimates = samples.mean(axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def evaluate_policy(
    experiment: str,
    split: str,
    success: pd.DataFrame,
    order: list[str],
    total_population: int,
    budgets: list[int],
    rng: np.random.Generator,
    n_bootstrap: int,
) -> pd.DataFrame:
    first_success = first_success_positions(success, order)
    wrong_count = len(success)
    baseline_correct = total_population - wrong_count
    rows = []

    for budget in budgets:
        budget = min(budget, len(order))
        repaired = first_success <= budget
        repaired_count = int(repaired.sum())
        repair_rate = float(repaired.mean())
        ci_low, ci_high = bootstrap_proportion_ci(
            repaired.astype(float), rng, n_bootstrap
        )
        queries = np.minimum(first_success, budget)
        successful_queries = queries[repaired]
        rows.append(
            {
                "experiment": experiment,
                "split": split,
                "budget": budget,
                "wrong_inputs": wrong_count,
                "repaired_inputs": repaired_count,
                "repair_rate": repair_rate,
                "repair_ci_low": ci_low,
                "repair_ci_high": ci_high,
                "baseline_correct": baseline_correct,
                "total_inputs": total_population,
                "baseline_accuracy": baseline_correct / total_population,
                "assisted_accuracy": (baseline_correct + repaired_count)
                / total_population,
                "mean_queries_all_failures": float(queries.mean()),
                "median_queries_all_failures": float(np.median(queries)),
                "mean_queries_successes": (
                    float(successful_queries.mean())
                    if successful_queries.size
                    else float("nan")
                ),
                "median_queries_successes": (
                    float(np.median(successful_queries))
                    if successful_queries.size
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def parameter_performance(
    experiment: str,
    dev: pd.DataFrame,
    test: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for config in metadata.index:
        rows.append(
            {
                "experiment": experiment,
                "config": config,
                "epsilon": float(metadata.loc[config, "epsilon"]),
                "steps": int(metadata.loc[config, "steps"]),
                "dev_repairs": int(dev[config].sum()),
                "dev_repair_rate": float(dev[config].mean()),
                "test_repairs": int(test[config].sum()),
                "test_repair_rate": float(test[config].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["experiment", "dev_repair_rate", "epsilon", "steps"],
        ascending=[True, False, True, True],
    )


def overlap_table(experiment: str, success: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for left in success.columns:
        left_set = success.index[success[left]]
        for right in success.columns:
            right_set = success.index[success[right]]
            intersection = len(left_set.intersection(right_set))
            union = len(left_set.union(right_set))
            rows.append(
                {
                    "experiment": experiment,
                    "left_config": left,
                    "right_config": right,
                    "intersection": intersection,
                    "union": union,
                    "jaccard": intersection / union if union else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def plot_budget_curves(policy: pd.DataFrame, output_path: Path) -> None:
    test = policy[policy["split"] == "test"]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for experiment, group in test.groupby("experiment", sort=False):
        group = group.sort_values("budget")
        ax.plot(
            group["budget"],
            group["repair_rate"] * 100,
            marker="o",
            linewidth=2,
            label=experiment,
        )
    ax.set_xlabel("Maximum downstream recognition queries")
    ax.set_ylabel("Held-out failed-input repair rate (%)")
    ax.set_xticks(DEFAULT_BUDGETS)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def write_markdown_summary(
    output_path: Path,
    policy: pd.DataFrame,
    parameters: pd.DataFrame,
    orders: dict[str, list[str]],
) -> None:
    lines = [
        "# Dev-Learned, Test-Frozen Adaptive Search",
        "",
        "The parameter order is learned using only dev-speaker failures and is "
        "then frozen before held-out test evaluation.",
        "",
        "## Held-out test results",
        "",
        "| Vocabulary | Budget | Repair rate (95% bootstrap CI) | "
        "Assisted accuracy | Mean queries |",
        "|---|---:|---:|---:|---:|",
    ]
    test = policy[policy["split"] == "test"]
    for row in test.itertuples(index=False):
        lines.append(
            f"| {row.experiment} | {row.budget} | "
            f"{100 * row.repair_rate:.1f}% "
            f"[{100 * row.repair_ci_low:.1f}, {100 * row.repair_ci_high:.1f}] | "
            f"{100 * row.assisted_accuracy:.1f}% | "
            f"{row.mean_queries_all_failures:.2f} |"
        )

    lines.extend(["", "## Dev-selected single configurations", ""])
    lines.append(
        "| Vocabulary | Configuration selected on dev | Dev repair | Test repair |"
    )
    lines.append("|---|---|---:|---:|")
    for experiment, group in parameters.groupby("experiment", sort=False):
        best = group.sort_values(
            ["dev_repair_rate", "epsilon", "steps"],
            ascending=[False, True, True],
        ).iloc[0]
        lines.append(
            f"| {experiment} | eps={best.epsilon:.5f}, K={int(best.steps)} | "
            f"{100 * best.dev_repair_rate:.1f}% | "
            f"{100 * best.test_repair_rate:.1f}% |"
        )

    lines.extend(["", "## Frozen greedy orders", ""])
    for experiment, order in orders.items():
        lines.append(f"**{experiment}:** " + " -> ".join(order))
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/results/adaptive_search"),
    )
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    policy_frames = []
    parameter_frames = []
    order_frames = []
    overlap_frames = []
    orders: dict[str, list[str]] = {}

    for experiment in EXPERIMENTS:
        results = load_results(experiment.result_csv)
        population = load_population_counts(experiment.population_csv)
        metadata = config_metadata(results)
        dev = make_success_matrix(results, "dev")
        test = make_success_matrix(results, "test")

        if set(dev.columns) != set(test.columns):
            raise ValueError(f"{experiment.name}: dev/test configurations differ")
        dev = dev[metadata.index]
        test = test[metadata.index]

        order, order_df = greedy_coverage_order(dev, metadata)
        orders[experiment.name] = order
        order_df.insert(0, "experiment", experiment.name)
        order_frames.append(order_df)

        budgets = sorted(set(min(b, len(order)) for b in DEFAULT_BUDGETS))
        policy_frames.append(
            evaluate_policy(
                experiment.name,
                "dev",
                dev,
                order,
                population["dev"],
                budgets,
                rng,
                args.bootstrap,
            )
        )
        policy_frames.append(
            evaluate_policy(
                experiment.name,
                "test",
                test,
                order,
                population["test"],
                budgets,
                rng,
                args.bootstrap,
            )
        )
        parameter_frames.append(
            parameter_performance(experiment.name, dev, test, metadata)
        )
        overlap_frames.append(overlap_table(experiment.name, test))

    policy = pd.concat(policy_frames, ignore_index=True)
    parameters = pd.concat(parameter_frames, ignore_index=True)
    order_details = pd.concat(order_frames, ignore_index=True)
    overlaps = pd.concat(overlap_frames, ignore_index=True)

    policy.to_csv(args.output_dir / "adaptive_search_policy_results.csv", index=False)
    parameters.to_csv(
        args.output_dir / "single_parameter_results.csv", index=False
    )
    order_details.to_csv(
        args.output_dir / "dev_greedy_parameter_orders.csv", index=False
    )
    overlaps.to_csv(args.output_dir / "test_parameter_overlap.csv", index=False)
    plot_budget_curves(policy, args.output_dir / "success_by_query_budget.png")
    write_markdown_summary(
        args.output_dir / "summary.md", policy, parameters, orders
    )

    print(policy[policy["split"] == "test"].to_string(index=False))
    print(f"\nSaved analysis to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
