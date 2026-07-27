"""Validate and summarize local occupancy-characterization artifacts."""

import argparse
import ast
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import statistics

try:
    from .occupancy_characterization_matrix import (
        infer_agent_online_updates,
        infer_behavior_policy,
    )
except ImportError:
    from occupancy_characterization_matrix import (
        infer_agent_online_updates,
        infer_behavior_policy,
    )


PERFORMANCE_KEY_PRIORITY = (
    "success",
    "success_rate",
    "episode.return",
    "return",
    "reward",
)
OCCUPANCY_METRICS = (
    "eval/balanced_accuracy",
    "eval/logit_gap",
    "eval/offline_probability_mean",
    "eval/online_probability_mean",
)
SUMMARY_METRICS = (
    "final_eval_balanced_accuracy",
    "final_eval_logit_gap",
    "final_eval_offline_probability_mean",
    "final_eval_online_probability_mean",
    "mean_last_5_eval_balanced_accuracy",
    "mean_last_5_eval_logit_gap",
    "mean_last_5_eval_offline_probability_mean",
    "mean_last_5_eval_online_probability_mean",
    "final_online_performance",
    "mean_online_performance",
    "minimum_online_performance",
)
PAIR_DELTA_METRICS = (
    "final_eval_balanced_accuracy",
    "mean_last_5_eval_balanced_accuracy",
    "final_eval_logit_gap",
    "mean_last_5_eval_logit_gap",
    "final_online_performance",
)
CORRELATION_METRICS = (
    "final_eval_balanced_accuracy",
    "mean_last_5_eval_balanced_accuracy",
    "final_eval_logit_gap",
    "mean_last_5_eval_logit_gap",
    "final_online_performance",
)
NON_TARGET_EXCLUSIONS = {
    "tags",
    "train_action_gain",
    "train_action_delay",
    "eval_action_gain",
    "eval_action_delay",
}


class CharacterizationError(ValueError):
    """Raised when experiment artifacts cannot be safely characterized."""


def _load_json(path):
    try:
        return json.loads(Path(path).read_text())
    except Exception as exc:
        raise CharacterizationError(
            f"failed to read JSON {path}: {exc}"
        ) from exc


def _read_finite_numeric_csv(path, *, allow_empty=False):
    path = Path(path)
    try:
        with path.open(newline="") as file:
            reader = csv.DictReader(file)
            fieldnames = reader.fieldnames
            raw_rows = list(reader)
    except Exception as exc:
        raise CharacterizationError(
            f"failed to read CSV {path}: {exc}"
        ) from exc
    if not fieldnames:
        raise CharacterizationError(f"CSV has no header: {path}")
    if not raw_rows and not allow_empty:
        raise CharacterizationError(f"CSV has no data rows: {path}")
    rows = []
    for row_index, raw_row in enumerate(raw_rows, start=2):
        row = {}
        for name in fieldnames:
            raw_value = raw_row.get(name)
            if raw_value is None or raw_value == "":
                raise CharacterizationError(
                    f"{path}:{row_index}: missing numeric value for {name!r}."
                )
            try:
                value = float(raw_value)
            except ValueError as exc:
                raise CharacterizationError(
                    f"{path}:{row_index}: non-numeric value "
                    f"{raw_value!r} for {name!r}."
                ) from exc
            if not math.isfinite(value):
                raise CharacterizationError(
                    f"{path}:{row_index}: non-finite value "
                    f"{raw_value!r} for {name!r}."
                )
            row[name] = value
        rows.append(row)
    return rows, tuple(fieldnames)


def _normalise_value(value):
    if isinstance(value, tuple):
        return [_normalise_value(item) for item in value]
    if isinstance(value, list):
        return [_normalise_value(item) for item in value]
    if isinstance(value, str) and value.startswith("("):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return value
        return _normalise_value(parsed)
    return value


def _actual_flag(flags, key):
    if key.startswith("agent."):
        agent = flags.get("agent")
        if not isinstance(agent, dict):
            raise CharacterizationError(
                "flags.json has no nested agent configuration."
            )
        nested_key = key.split(".", 1)[1]
        if nested_key not in agent:
            raise CharacterizationError(
                f"flags.json is missing agent flag {nested_key!r}."
            )
        return agent[nested_key]
    if key not in flags:
        raise CharacterizationError(
            f"flags.json is missing expected flag {key!r}."
        )
    return flags[key]


def _validate_flags(run, flags, run_directory):
    for key, expected in run["kwargs"].items():
        if key == "agent":
            agent = flags.get("agent", {})
            if agent.get("agent_name") != "qam":
                raise CharacterizationError(
                    f"{run_directory}: expected QAM agent configuration."
                )
            continue
        if key == "save_dir":
            actual_path = Path(flags.get("save_dir", "")).resolve()
            if actual_path != run_directory.resolve():
                raise CharacterizationError(
                    f"{run_directory}: save_dir does not match run directory."
                )
            continue
        actual = _actual_flag(flags, key)
        if _normalise_value(actual) != _normalise_value(expected):
            raise CharacterizationError(
                f"{run_directory}: flag {key!r} mismatch: "
                f"expected {expected!r}, got {actual!r}."
            )
    if flags.get("start_training", 0) <= flags.get("online_steps", 0):
        raise CharacterizationError(
            f"{run_directory}: start_training must exceed online_steps."
        )
    behavior_policy = infer_behavior_policy(flags)
    agent_online_updates = infer_agent_online_updates(flags)
    if (
        behavior_policy != "fixed_offline_agent"
        or agent_online_updates != "disabled"
    ):
        raise CharacterizationError(
            f"{run_directory}: fixed-policy characterization requires "
            "behavior_policy=fixed_offline_agent and "
            "agent_online_updates=disabled."
        )
    if run.get("behavior_policy") != behavior_policy:
        raise CharacterizationError(
            f"{run_directory}: behavior_policy metadata does not match "
            "flags.json."
        )
    if run.get("agent_online_updates") != agent_online_updates:
        raise CharacterizationError(
            f"{run_directory}: agent_online_updates metadata does not "
            "match flags.json."
        )


def _identity_matches(run, flags):
    return (
        flags.get("run_group") == run["run_group"]
        and flags.get("seed") == run["seed"]
        and flags.get("env_name") == run["env_name"]
        and flags.get("tags") == run["tags"]
        and flags.get("train_action_gain")
        == run["kwargs"]["train_action_gain"]
        and flags.get("train_action_delay")
        == run["kwargs"]["train_action_delay"]
    )


def _validate_manifest(manifest, run_group):
    required = {
        "schema_version",
        "mode",
        "run_group",
        "base_commit",
        "generated_at",
        "run_count",
        "runs",
    }
    missing = required - set(manifest)
    if missing:
        raise CharacterizationError(
            f"manifest is missing fields: {sorted(missing)}."
        )
    if manifest["schema_version"] != 1:
        raise CharacterizationError(
            f"unsupported manifest schema: {manifest['schema_version']}."
        )
    if manifest["run_group"] != run_group:
        raise CharacterizationError(
            "manifest run_group does not match requested run_group."
        )
    if manifest["mode"] not in ("local", "pilot"):
        raise CharacterizationError(
            f"unsupported manifest mode: {manifest['mode']!r}."
        )
    if manifest["run_count"] != len(manifest["runs"]):
        raise CharacterizationError("manifest run_count is inconsistent.")
    expected = 2 if manifest["mode"] == "local" else 21
    if manifest["run_count"] != expected:
        raise CharacterizationError(
            f"{manifest['mode']} manifest must contain {expected} runs."
        )
    identities = [
        (run["condition"], run["seed"]) for run in manifest["runs"]
    ]
    if len(set(identities)) != len(identities):
        raise CharacterizationError(
            "manifest contains duplicate (condition, seed) runs."
        )
    for run in manifest["runs"]:
        kwargs = run.get("kwargs")
        if not isinstance(kwargs, dict):
            raise CharacterizationError(
                "manifest run is missing kwargs."
            )
        behavior_policy = infer_behavior_policy(kwargs)
        agent_online_updates = infer_agent_online_updates(kwargs)
        if behavior_policy == "random_only":
            raise CharacterizationError(
                "invalid_for_fixed_policy_characterization: "
                "balanced_sampling=True caused random actions for every "
                "online step."
            )
        if (
            behavior_policy != "fixed_offline_agent"
            or agent_online_updates != "disabled"
        ):
            raise CharacterizationError(
                "manifest run is not fixed-policy characterization: "
                f"behavior_policy={behavior_policy}, "
                f"agent_online_updates={agent_online_updates}."
            )
        if run.get("behavior_policy") != behavior_policy:
            raise CharacterizationError(
                "manifest behavior_policy metadata is missing or does not "
                "match run flags."
            )
        if run.get("agent_online_updates") != agent_online_updates:
            raise CharacterizationError(
                "manifest agent_online_updates metadata is missing or does "
                "not match run flags."
            )
    by_seed = defaultdict(list)
    for run in manifest["runs"]:
        by_seed[run["seed"]].append(run)
    for seed_runs in by_seed.values():
        nominal = next(
            (
                run
                for run in seed_runs
                if run["condition"] == "nominal"
            ),
            None,
        )
        if nominal is None:
            raise CharacterizationError(
                "every seed must include a nominal run."
            )
        nominal_common = {
            key: value
            for key, value in nominal["kwargs"].items()
            if key not in NON_TARGET_EXCLUSIONS
        }
        for run in seed_runs:
            common = {
                key: value
                for key, value in run["kwargs"].items()
                if key not in NON_TARGET_EXCLUSIONS
            }
            if common != nominal_common:
                raise CharacterizationError(
                    "nominal and shifted runs have non-target flag mismatch."
                )


def _validate_progress(run_directory, flags):
    progress_path = run_directory / "progress.tk"
    if not progress_path.exists():
        return
    content = progress_path.read_text().strip()
    expected = f"online,{flags['online_steps']}"
    if content != expected:
        raise CharacterizationError(
            f"{run_directory}: unfinished progress {content!r}; "
            f"expected {expected!r}."
        )


def _validate_no_online_updates(run_directory):
    path = run_directory / "online_agent.csv"
    if not path.exists():
        return
    with path.open(newline="") as file:
        reader = csv.reader(file)
        rows = list(reader)
    if len(rows) > 1:
        raise CharacterizationError(
            f"{run_directory}: online_agent.csv contains update data."
        )


def _online_steps(rows, offline_steps, path):
    steps = []
    for row in rows:
        if "step" not in row:
            raise CharacterizationError(f"{path}: missing step column.")
        step = row["step"]
        if not step.is_integer():
            raise CharacterizationError(
                f"{path}: step must be an integer, got {step}."
            )
        online_step = int(step) - int(offline_steps)
        if online_step > 0:
            steps.append(online_step)
    return steps


def _expected_occupancy_steps(flags):
    interval = int(flags["occupancy_update_interval"])
    start_size = int(flags["occupancy_start_size"])
    online_steps = int(flags["online_steps"])
    return [
        step
        for step in range(interval, online_steps + 1, interval)
        if step >= start_size
    ]


def _is_index_column(name):
    normalised = name.strip().lower().replace("-", "_").replace(" ", "_")
    return (
        normalised in {
            "step",
            "index",
            "idx",
            "row",
            "row_index",
            "global_step",
            "online_step",
        }
        or normalised.startswith("unnamed:")
    )


def _candidate_statistics(values):
    numeric_values = [float(value) for value in values]
    finite_values = [
        value for value in numeric_values if math.isfinite(value)
    ]
    unique_count = len(set(finite_values))
    is_constant = bool(finite_values) and unique_count == 1
    if not numeric_values:
        reason = "missing"
    elif len(finite_values) != len(numeric_values):
        reason = "non_finite"
    elif len(finite_values) < 2:
        reason = "insufficient_samples"
    elif is_constant:
        reason = "constant"
    else:
        reason = "informative"
    return {
        "count": len(numeric_values),
        "finite_count": len(finite_values),
        "minimum": min(finite_values) if finite_values else None,
        "maximum": max(finite_values) if finite_values else None,
        "mean": (
            statistics.fmean(finite_values) if finite_values else None
        ),
        "std": (
            statistics.pstdev(finite_values) if finite_values else None
        ),
        "unique_count": unique_count,
        "is_constant": is_constant,
        "informative": reason == "informative",
        "reason": reason,
    }


def performance_candidate_statistics(rows, fieldnames=None, requested=None):
    """Summarize semantically valid numeric policy-performance fields."""
    if fieldnames is None:
        fieldnames = tuple(rows[0]) if rows else ()
    numeric_columns = tuple(
        name for name in fieldnames if not _is_index_column(name)
    )
    candidate_names = [
        name for name in PERFORMANCE_KEY_PRIORITY if name in numeric_columns
    ]
    if (
        requested is not None
        and requested in numeric_columns
        and requested not in candidate_names
    ):
        candidate_names.append(requested)
    return {
        name: _candidate_statistics(
            [row[name] for row in rows if name in row]
        )
        for name in candidate_names
    }


def _select_performance_key(candidate_statistics, numeric_columns, requested):
    if requested is not None:
        if requested not in numeric_columns:
            raise CharacterizationError(
                f"performance key {requested!r} not found; numeric columns: "
                + ", ".join(numeric_columns)
            )
        return requested, "explicit"
    for candidate in PERFORMANCE_KEY_PRIORITY:
        statistics_for_candidate = candidate_statistics.get(candidate)
        if (
            statistics_for_candidate is not None
            and statistics_for_candidate["informative"]
        ):
            return candidate, "automatic"
    for candidate in PERFORMANCE_KEY_PRIORITY:
        if candidate in candidate_statistics:
            return candidate, "automatic"
    raise CharacterizationError(
        "could not auto-select performance key; numeric columns: "
        + ", ".join(numeric_columns)
    )


def _detector_summary(rows):
    for name in OCCUPANCY_METRICS:
        if name not in rows[0]:
            raise CharacterizationError(
                f"occupancy CSV is missing metric {name!r}."
            )
    window_size = min(5, len(rows))
    window = rows[-window_size:]
    result = {"occupancy_row_count": len(rows), "last_5_window_size": window_size}
    suffixes = {
        "eval/balanced_accuracy": "eval_balanced_accuracy",
        "eval/logit_gap": "eval_logit_gap",
        "eval/offline_probability_mean": (
            "eval_offline_probability_mean"
        ),
        "eval/online_probability_mean": (
            "eval_online_probability_mean"
        ),
    }
    for source, suffix in suffixes.items():
        result[f"final_{suffix}"] = rows[-1][source]
        result[f"mean_last_5_{suffix}"] = statistics.fmean(
            row[source] for row in window
        )
    return result


def _performance_summary(rows, key):
    values = [row[key] for row in rows]
    return {
        "final_online_performance": values[-1],
        "mean_online_performance": statistics.fmean(values),
        "minimum_online_performance": min(values),
        "online_performance_row_count": len(values),
    }


def _validate_offline_training_trajectories(validated_runs):
    by_seed = defaultdict(list)
    for validated in validated_runs:
        by_seed[validated["run"]["seed"]].append(validated)
    maximum_absolute_difference = 0.0
    for seed, seed_runs in by_seed.items():
        nominal = next(
            (
                validated
                for validated in seed_runs
                if validated["run"]["condition"] == "nominal"
            ),
            None,
        )
        if nominal is None:
            raise CharacterizationError(
                f"seed {seed}: missing nominal offline trajectory."
            )
        nominal_fields = nominal["offline_fieldnames"]
        nominal_rows = nominal["offline_rows"]
        for shifted in seed_runs:
            if shifted is nominal:
                continue
            condition = shifted["run"]["condition"]
            if shifted["offline_fieldnames"] != nominal_fields:
                raise CharacterizationError(
                    f"seed {seed} {condition}: offline training trajectory "
                    "header does not match nominal."
                )
            shifted_rows = shifted["offline_rows"]
            if len(shifted_rows) != len(nominal_rows):
                raise CharacterizationError(
                    f"seed {seed} {condition}: offline training trajectory "
                    "row count does not match nominal."
                )
            for nominal_row, shifted_row in zip(
                nominal_rows,
                shifted_rows,
            ):
                nominal_step = nominal_row.get("step")
                shifted_step = shifted_row.get("step")
                if nominal_step != shifted_step:
                    raise CharacterizationError(
                        f"seed {seed} {condition}: offline training "
                        f"trajectory step {shifted_step} does not match "
                        f"nominal step {nominal_step}."
                    )
                for field in nominal_fields:
                    nominal_value = nominal_row[field]
                    shifted_value = shifted_row[field]
                    difference = abs(nominal_value - shifted_value)
                    maximum_absolute_difference = max(
                        maximum_absolute_difference,
                        difference,
                    )
                    tolerance = 1e-12 + 1e-12 * abs(nominal_value)
                    if difference > tolerance:
                        raise CharacterizationError(
                            f"seed {seed} {condition}: offline training "
                            f"trajectory mismatch at step "
                            f"{int(nominal_step)}, field {field}: "
                            f"{shifted_value} != {nominal_value}."
                        )
    return maximum_absolute_difference


def _write_csv(path, rows):
    path = Path(path)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0])
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def average_ranks(values):
    """Return one-based average ranks with deterministic tie handling."""
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position + 1
        while (
            end < len(order)
            and values[order[end]] == values[order[position]]
        ):
            end += 1
        average = (position + 1 + end) / 2.0
        for ordered_index in order[position:end]:
            ranks[ordered_index] = average
        position = end
    return ranks


def pearson_correlation(xs, ys):
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    centered_x = [value - mean_x for value in xs]
    centered_y = [value - mean_y for value in ys]
    denominator = math.sqrt(
        sum(value * value for value in centered_x)
        * sum(value * value for value in centered_y)
    )
    if denominator == 0:
        return None
    return sum(
        x_value * y_value
        for x_value, y_value in zip(centered_x, centered_y)
    ) / denominator


def spearman_rank_correlation(xs, ys):
    """Return Spearman correlation or None for fewer than three points."""
    if len(xs) != len(ys) or len(xs) < 3:
        return None
    return pearson_correlation(average_ranks(xs), average_ranks(ys))


def _paired_deltas(run_rows):
    nominal_by_seed = {
        row["seed"]: row
        for row in run_rows
        if row["condition"] == "nominal"
    }
    paired = []
    for row in run_rows:
        if row["condition"] == "nominal":
            continue
        nominal = nominal_by_seed.get(row["seed"])
        if nominal is None:
            raise CharacterizationError(
                f"missing same-seed nominal for {row['condition']}."
            )
        delta = {
            "condition": row["condition"],
            "condition_family": row["condition_family"],
            "severity": row["severity"],
            "seed": row["seed"],
            "reference_condition": "nominal",
        }
        for metric in PAIR_DELTA_METRICS:
            delta[f"delta_{metric}"] = row[metric] - nominal[metric]
        paired.append(delta)
    return paired


def _aggregate(run_rows):
    grouped = defaultdict(list)
    for row in run_rows:
        grouped[row["condition"]].append(row)
    aggregates = []
    for condition, rows in grouped.items():
        aggregate = {
            "condition": condition,
            "condition_family": rows[0]["condition_family"],
            "severity": rows[0]["severity"],
            "n": len(rows),
        }
        for metric in SUMMARY_METRICS:
            values = [row[metric] for row in rows]
            aggregate[f"{metric}_mean"] = statistics.fmean(values)
            aggregate[f"{metric}_std"] = statistics.pstdev(values)
        aggregates.append(aggregate)
    return sorted(
        aggregates,
        key=lambda row: (
            row["condition_family"],
            row["severity"],
            row["condition"],
        ),
    )


def aggregate_run_summaries(run_rows):
    """Aggregate run-level summaries by condition using population std."""
    return _aggregate(run_rows)


def _correlations(aggregate_rows):
    result = {}
    for family in ("gain", "delay"):
        family_rows = [
            row
            for row in aggregate_rows
            if row["condition_family"] == family
        ]
        result[family] = {}
        for metric in CORRELATION_METRICS:
            value = spearman_rank_correlation(
                [row["severity"] for row in family_rows],
                [row[f"{metric}_mean"] for row in family_rows],
            )
            result[family][metric] = (
                "unavailable" if value is None else value
            )
    return result


def _report_text(
    manifest,
    run_rows,
    paired_rows,
    performance_key,
    validation,
):
    def format_statistic(value):
        return "unavailable" if value is None else f"{value:.6g}"

    lines = [
        "# Occupancy characterization report",
        "",
        "## 实验范围",
        "",
        (
            f"Mode: `{manifest['mode']}`; run group: "
            f"`{manifest['run_group']}`; completed: "
            f"{validation['completed_run_count']}/{validation['run_count']}."
        ),
        "",
        (
            "Online transitions 由固定离线 Agent 采集；"
            "QAM online update 被禁用。"
        ),
        "",
        "## 完整性检查",
        "",
        (
            "flags、token/progress、occupancy/eval CSV、step 唯一性、"
            "有限值和 online-agent absence 全部通过。"
        ),
        (
            "same-seed offline training trajectory consistency check: "
            "PASS；max absolute difference="
            f"{validation['offline_training_trajectory_max_abs_difference']:.6g}。"
        ),
        "",
        "## 每个 condition 的指标",
        "",
        "| condition | seed | final balanced acc | final logit gap | final performance |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in run_rows:
        lines.append(
            f"| {row['condition']} | {row['seed']} | "
            f"{row['final_eval_balanced_accuracy']:.6g} | "
            f"{row['final_eval_logit_gap']:.6g} | "
            f"{row['final_online_performance']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Paired comparison",
            "",
            (
                "以下 delta 均为 shift - same-seed nominal；只报告数值，"
                "不将 local 两点解释为趋势或统计显著性。"
            ),
            "",
        ]
    )
    for row in paired_rows:
        lines.append(
            f"- {row['condition']} seed {row['seed']}: "
            f"Δ final balanced accuracy="
            f"{row['delta_final_eval_balanced_accuracy']:.6g}, "
            f"Δ final logit gap="
            f"{row['delta_final_eval_logit_gap']:.6g}, "
            f"Δ final performance="
            f"{row['delta_final_online_performance']:.6g}"
        )
    lines.extend(
        [
            "",
            "## 性能字段选择",
            "",
            (
                f"Selected performance key: `{performance_key}` "
                f"({validation['performance_key_source']}); informative="
                f"`{str(validation['performance_key_informative']).lower()}`; "
                f"reason=`{validation['performance_key_reason']}`."
            ),
            "",
            (
                "| candidate | count | finite | min | max | mean | std | "
                "unique | constant | status |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for name, candidate in validation["performance_candidates"].items():
        lines.append(
            f"| `{name}` | {candidate['count']} | "
            f"{candidate['finite_count']} | "
            f"{format_statistic(candidate['minimum'])} | "
            f"{format_statistic(candidate['maximum'])} | "
            f"{format_statistic(candidate['mean'])} | "
            f"{format_statistic(candidate['std'])} | "
            f"{candidate['unique_count']} | "
            f"{str(candidate['is_constant']).lower()} | "
            f"{candidate['reason']} |"
        )
    lines.extend(
        [
            "",
            "## 风险和解释边界",
            "",
            "- Detector logit 是未校准 occupancy-ratio proxy，不是纯 dynamics ratio。",
            "- Nominal rollout 仍可能与 offline dataset 存在 visitation shift。",
            "- Local 两个 run、单 seed 不支持显著性或单调性结论。",
            "- Fresh online detector eval sampling 不保证与训练样本严格 disjoint。",
            (
                "- 所选性能指标在这些短 run 中没有提供可辨识变化；"
                "不能用于 detector score 与策略性能关系分析。"
                if not validation["performance_key_informative"]
                else "- 所选性能指标在这些 run 中存在可辨识变化。"
            ),
            "",
            "## Pilot gate",
            "",
            (
                "Detector pilot readiness: "
                f"`{str(validation['detector_pilot_ready']).lower()}`."
            ),
            (
                "Performance correlation readiness: "
                f"`{str(validation['performance_correlation_ready']).lower()}`."
            ),
            "",
            (
                "实验工具和 detector characterization 路径已通过 local gate，"
                "可以进入多 seed detector pilot。"
                if validation["detector_pilot_ready"]
                else "Detector characterization local gate 尚未通过。"
            ),
            (
                "当前固定策略的可用性能候选字段均无可辨识变化，因此本地结果"
                "不能用于 detector score 与策略性能关系分析。在进行 "
                "performance-correlation 实验前，需要先完成策略能力校准。"
                if not validation["performance_correlation_ready"]
                else "当前性能字段可用于后续多 seed performance-correlation 分析。"
            ),
            "",
        ]
    )
    return "\n".join(lines)


def summarize_characterization(
    exp_root,
    run_group,
    output_dir,
    performance_key=None,
    manifest_path=None,
):
    """Validate all manifest runs, write summaries, and return validation."""
    exp_root = Path(exp_root)
    output_dir = Path(output_dir)
    if manifest_path is None:
        candidate = output_dir.parent / "manifest.json"
        if not candidate.exists():
            candidate = Path("logs") / run_group / "manifest.json"
        manifest_path = candidate
    manifest = _load_json(manifest_path)
    _validate_manifest(manifest, run_group)

    candidates = []
    for flags_path in exp_root.rglob("flags.json"):
        flags = _load_json(flags_path)
        if flags.get("run_group") == run_group:
            candidates.append((flags_path.parent, flags))

    validated_runs = []
    for run in manifest["runs"]:
        matches = [
            (directory, flags)
            for directory, flags in candidates
            if _identity_matches(run, flags)
        ]
        if len(matches) != 1:
            raise CharacterizationError(
                f"{run['condition']} seed {run['seed']} matched "
                f"{len(matches)} experiment directories; expected 1."
            )
        run_directory, flags = matches[0]
        _validate_flags(run, flags, run_directory)
        if not (run_directory / "token.tk").exists():
            raise CharacterizationError(
                f"{run_directory}: token.tk is missing."
            )
        _validate_progress(run_directory, flags)
        _validate_no_online_updates(run_directory)

        occupancy_path = run_directory / "occupancy_detector.csv"
        eval_path = run_directory / "eval.csv"
        offline_agent_path = run_directory / "offline_agent.csv"
        if not occupancy_path.exists():
            raise CharacterizationError(
                f"{run_directory}: occupancy_detector.csv is missing."
            )
        if not eval_path.exists():
            raise CharacterizationError(
                f"{run_directory}: eval.csv is missing."
            )
        if not offline_agent_path.exists():
            raise CharacterizationError(
                f"{run_directory}: offline_agent.csv is missing."
            )
        occupancy_rows, _ = _read_finite_numeric_csv(occupancy_path)
        eval_rows, eval_fieldnames = _read_finite_numeric_csv(eval_path)
        offline_rows, offline_fieldnames = _read_finite_numeric_csv(
            offline_agent_path
        )

        occupancy_steps = _online_steps(
            occupancy_rows,
            flags["offline_steps"],
            occupancy_path,
        )
        if len(set(occupancy_steps)) != len(occupancy_steps):
            raise CharacterizationError(
                f"{occupancy_path}: duplicate occupancy step."
            )
        if occupancy_steps != sorted(occupancy_steps):
            raise CharacterizationError(
                f"{occupancy_path}: occupancy steps are not increasing."
            )
        expected_steps = _expected_occupancy_steps(flags)
        if occupancy_steps != expected_steps:
            raise CharacterizationError(
                f"{occupancy_path}: occupancy online steps "
                f"{occupancy_steps} do not match {expected_steps}."
            )
        if manifest["mode"] == "local" and len(occupancy_rows) != 10:
            raise CharacterizationError(
                f"{occupancy_path}: local run must have 10 occupancy rows."
            )

        online_eval_rows = [
            row
            for row in eval_rows
            if int(row["step"]) - int(flags["offline_steps"]) > 0
        ]
        if not online_eval_rows:
            raise CharacterizationError(
                f"{eval_path}: no online evaluation rows."
            )
        validated_runs.append(
            {
                "run": run,
                "run_directory": run_directory,
                "online_eval_rows": online_eval_rows,
                "eval_fieldnames": eval_fieldnames,
                "offline_rows": offline_rows,
                "offline_fieldnames": offline_fieldnames,
                "occupancy_steps": occupancy_steps,
                "detector_summary": _detector_summary(occupancy_rows),
            }
        )

    offline_trajectory_max_abs_difference = (
        _validate_offline_training_trajectories(validated_runs)
    )
    numeric_columns = tuple(
        name
        for name in validated_runs[0]["eval_fieldnames"]
        if not _is_index_column(name)
        and all(
            name in validated["eval_fieldnames"]
            for validated in validated_runs
        )
    )
    all_online_eval_rows = [
        row
        for validated in validated_runs
        for row in validated["online_eval_rows"]
    ]
    candidate_statistics = performance_candidate_statistics(
        all_online_eval_rows,
        fieldnames=numeric_columns,
        requested=performance_key,
    )
    selected_performance_key, performance_key_source = (
        _select_performance_key(
            candidate_statistics,
            numeric_columns,
            performance_key,
        )
    )
    selected_performance_statistics = candidate_statistics[
        selected_performance_key
    ]

    run_rows = []
    for validated in validated_runs:
        run = validated["run"]
        row = {
            "condition": run["condition"],
            "condition_family": run["condition_family"],
            "severity": run["severity"],
            "seed": run["seed"],
            "method": run["method"],
            "domain": run["domain"],
            "task": run["task"],
            "env_name": run["env_name"],
            "run_group": run_group,
            "experiment_directory": str(validated["run_directory"]),
            "performance_key": selected_performance_key,
            "performance_key_source": performance_key_source,
            "performance_key_informative": (
                selected_performance_statistics["informative"]
            ),
            "performance_key_reason": (
                selected_performance_statistics["reason"]
            ),
            "occupancy_online_steps": ";".join(
                str(step) for step in validated["occupancy_steps"]
            ),
            **validated["detector_summary"],
            **_performance_summary(
                validated["online_eval_rows"],
                selected_performance_key,
            ),
        }
        run_rows.append(row)

    paired_rows = _paired_deltas(run_rows)
    aggregate_rows = aggregate_run_summaries(run_rows)
    correlations = _correlations(aggregate_rows)
    observable_difference = any(
        abs(row["delta_final_eval_balanced_accuracy"]) > 1e-6
        or abs(row["delta_mean_last_5_eval_balanced_accuracy"]) > 1e-6
        or abs(row["delta_final_eval_logit_gap"]) > 1e-6
        or abs(row["delta_mean_last_5_eval_logit_gap"]) > 1e-6
        for row in paired_rows
    )
    detector_pilot_ready = len(run_rows) == len(manifest["runs"])
    performance_correlation_ready = (
        detector_pilot_ready
        and any(
            candidate["informative"]
            for candidate in candidate_statistics.values()
        )
    )
    validation = {
        "schema_version": 1,
        "mode": manifest["mode"],
        "run_group": run_group,
        "run_count": len(manifest["runs"]),
        "completed_run_count": len(run_rows),
        "flags_validated_count": len(run_rows),
        "behavior_policy": "fixed_offline_agent",
        "agent_online_updates": "disabled",
        "offline_training_trajectory_consistent": True,
        "offline_training_trajectory_max_abs_difference": (
            offline_trajectory_max_abs_difference
        ),
        "performance_key": selected_performance_key,
        "performance_key_source": performance_key_source,
        "performance_key_informative": (
            selected_performance_statistics["informative"]
        ),
        "performance_key_reason": (
            selected_performance_statistics["reason"]
        ),
        "performance_candidates": candidate_statistics,
        "all_metrics_finite": True,
        "occupancy_steps_complete": True,
        "duplicate_occupancy_steps": False,
        "online_agent_updates": False,
        "summary_outputs_complete": True,
        "hard_gate_passed": detector_pilot_ready,
        "detector_pilot_ready": detector_pilot_ready,
        "performance_correlation_ready": performance_correlation_ready,
        "observable_detector_difference": observable_difference,
        "correlations": correlations,
        "recommend_pilot": (
            detector_pilot_ready and observable_difference
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "run_summary.csv", run_rows)
    _write_csv(output_dir / "aggregate_summary.csv", aggregate_rows)
    _write_csv(output_dir / "paired_deltas.csv", paired_rows)
    (output_dir / "validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "report.md").write_text(
        _report_text(
            manifest,
            run_rows,
            paired_rows,
            selected_performance_key,
            validation,
        )
    )
    return validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-root", type=Path, required=True)
    parser.add_argument("--run-group", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--performance-key")
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()

    validation = summarize_characterization(
        exp_root=args.exp_root,
        run_group=args.run_group,
        output_dir=args.output_dir,
        performance_key=args.performance_key,
        manifest_path=args.manifest,
    )
    print(json.dumps(validation, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
