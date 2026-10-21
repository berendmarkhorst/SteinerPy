"""Export audited PACE JSON results to a CSV and a public Markdown report."""

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics


def load_rows(directory):
    """Include pending cases; never report a finished subset as the whole suite."""
    expected = {}
    for path in sorted(directory.glob("manifest-*.json")):
        manifest = json.loads(path.read_text())
        for case in manifest["cases"]:
            key = (case["track"], case["instance"], manifest["config"])
            if key in expected:
                raise ValueError("overlapping manifests: {}".format(key))
            expected[key] = dict(
                case,
                config=manifest["config"],
                status="pending",
                run=dict(
                    manifest["run"],
                    time_limit=manifest["time_limit"],
                    memory_gb=manifest["memory_gb"],
                    threads=1,
                ),
            )
    seen = set()
    for path in sorted(directory.glob("*/result.json")):
        result = json.loads(path.read_text())
        key = (result["track"], result["instance"], result["run"]["config"])
        if key not in expected:
            raise ValueError("result outside manifests: {}".format(key))
        if key in seen:
            raise ValueError("duplicate result: {}".format(key))
        seen.add(key)
        for field in ("commit", "dataset_commit", "time_limit", "memory_gb"):
            if result["run"][field] != expected[key]["run"][field]:
                raise ValueError(
                    "result differs from manifest: {} {}".format(key, field)
                )
        if result["instance_sha256"] != expected[key]["instance_sha256"]:
            raise ValueError("instance checksum differs from manifest")
        expected[key] = dict(result, config=key[2])
    return [expected[key] for key in sorted(expected)]


def metrics(rows):
    completed = [r for r in rows if r["status"] != "pending"]
    valid = [
        r
        for r in completed
        if r.get("feasible") and r["status"] in {"optimal", "feasible"}
    ]
    optimal = [
        r for r in valid if r.get("reference_optimal") or r.get("solver_proven_optimal")
    ]
    return dict(
        expected=len(rows),
        completed=len(completed),
        valid=len(valid),
        optimal=len(optimal),
        timeouts=sum(r["status"] == "timeout" for r in completed),
        errors=sum(
            r["status"] not in {"optimal", "feasible", "timeout"} for r in completed
        ),
        median_wall=(
            statistics.median(r["wall_seconds"] for r in valid) if valid else None
        ),
        quality=(
            100 * sum(r["reference_ratio"] for r in valid) / len(rows)
            if rows and len(completed) == len(rows)
            else None
        ),
    )


def write_csv(rows, path):
    fields = [
        "track",
        "instance",
        "split",
        "config",
        "status",
        "phase",
        "nodes",
        "edges",
        "terminals",
        "reduced_nodes",
        "reduced_edges",
        "reference_lower",
        "reference_upper",
        "objective",
        "gap",
        "feasible",
        "reference_optimal",
        "solver_proven_optimal",
        "reference_ratio",
        "wall_seconds",
        "parse_seconds",
        "preprocess_seconds",
        "model_build_seconds",
        "solve_seconds",
        "solver_seconds",
        "peak_rss_mb",
        "error",
        "instance_sha256",
        "commit",
        "dataset_commit",
        "python",
        "cpu",
        "threads",
        "seed",
        "time_limit",
        "memory_gb",
        "packages",
        "competitor_commit",
        "compiler",
        "binary_sha256",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            flattened = dict(row, **row["run"])
            flattened["packages"] = json.dumps(
                row["run"].get("packages", {}), sort_keys=True
            )
            flattened["binary_sha256"] = json.dumps(
                row["run"].get("binary_sha256", {}), sort_keys=True
            )
            writer.writerow(flattened)


def comparison_section(rows, output):
    expected = {
        (track, "instance{:03}.gr".format(number), config)
        for track in (1, 2, 3)
        for number in (1, 2, 3)
        for config in ("highs", "composition-highs", "wata")
    }
    if {(r["track"], r["instance"], r["config"]) for r in rows} != expected:
        raise ValueError("comparison requires all nine cases for all three methods")
    if any(r["run"]["time_limit"] != 90 or r["run"]["memory_gb"] != 8 for r in rows):
        raise ValueError("comparison requires equal 90-second, 8 GiB limits")
    for field in ("cpu", "dataset_commit"):
        if len({r["run"][field] for r in rows}) != 1:
            raise ValueError("comparison differs in " + field)
    checksums = defaultdict(set)
    for row in rows:
        checksums[(row["track"], row["instance"])].add(row["instance_sha256"])
    if any(len(values) != 1 for values in checksums.values()):
        raise ValueError("comparison instance checksums differ")
    write_csv(rows, output / "comparison.csv")
    lines = [
        "## Same-hardware comparison sample",
        "",
        "This separate experiment uses instances 001–003 from each track, one run per method, "
        "with a 90-second full wall deadline, one thread and an 8 GiB per-process address-space limit. "
        "It compares the classic specialized HiGHS model, the generic composition model "
        "(`Problem(..., node_costs={})`), and the original MIT-licensed "
        "[wata&sigma submission](https://github.com/wata-orz/steiner_tree). "
        "The external binary and checker run inside the same deadline; the binary's reported "
        "runtime is separately available as `solve_seconds`. Its peak RSS is the maximum "
        "individual-process RSS, not summed parent/child memory. All output edges are checked.",
        "",
        "| Track | Method | Finished / 3 | Valid trees | Optima | Timeouts | Errors | Median wall, valid (s) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for track in (1, 2, 3):
        for config in ("highs", "composition-highs", "wata"):
            m = metrics(
                [r for r in rows if r["track"] == track and r["config"] == config]
            )
            lines.append(
                "| {} | {} | {}/3 | {} | {} | {} | {} | {} |".format(
                    track,
                    config,
                    m["completed"],
                    m["valid"],
                    m["optimal"],
                    m["timeouts"],
                    m["errors"],
                    (
                        "{:.3f}".format(m["median_wall"])
                        if m["median_wall"] is not None
                        else "—"
                    ),
                )
            )
    lines += [
        "",
        "These nine cases are a fixed engineering sample, not a representative ranking of the full "
        "competition. [Per-instance comparison data](comparison.csv) records source revisions, "
        "compiler, binary checksums and matching dataset checksums. Keep this sample's 90-second "
        "results separate from the full sweep's 1,800-second results.",
        "",
    ]
    return lines


def export(rows, output, comparison=None):
    if not rows:
        raise ValueError("No manifests found")
    counts = Counter((r["config"], r["track"]) for r in rows)
    required = {
        (config, track): size
        for config in ("highs", "gurobi", "heuristic")
        for track, size in ((1, 200), (2, 200), (3, 199))
    }
    if counts != required:
        raise ValueError(
            "The full report requires all five manifests covering 599 cases per configuration"
        )
    output.mkdir(parents=True, exist_ok=True)
    write_csv(rows, output / "results.csv")
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["track"], row["config"])].append(row)
    complete = all(r["status"] != "pending" for r in rows)
    lines = [
        "# SteinerPy on PACE 2018",
        "",
        "{} snapshot generated {}.".format(
            "Completed" if complete else "**Provisional**",
            datetime.now(timezone.utc).isoformat(),
        ),
        "",
        "This benchmark attempts the official few-terminal, low-treewidth and heuristic tracks. "
        "Track 2 is solved as a graph problem; SteinerPy does not consume its supplied tree decompositions. "
        "The [organizers' dataset](https://github.com/PACE-challenge/SteinerTree-PACE-2018-instances) "
        "contains 200, 200 and 199 instances respectively; Track 3 instance 058 was withdrawn.",
        "",
        "## Measurements",
        "",
        "Each configuration runs once per instance in a fresh process with one solver/BLAS thread, "
        "an 8 GiB address-space limit, and a 1,800-second wall deadline covering imports, parsing, "
        "preprocessing, model construction, solving and validation. Default classic reductions and "
        "the few-terminal dynamic program remain enabled. Dual-ascent acceleration, decomposition, "
        "and experimental local-search/implied-profit options are disabled. Heuristic mode is the "
        "existing single-pass `exact=False` algorithm; it can finish before the deadline.",
        "",
        "The Gurobi runs are licensed; HiGHS and heuristic mode are the open-source alternatives. "
        "All costs and connectivity are checked on original edges. A feasible incumbent is not counted "
        "as an optimum unless its value matches a published optimum or the solver proves it. "
        "Timeouts produce no scored solution when the public solve call has not returned by the deadline.",
        "",
        "| Track | Configuration | Finished / expected | Valid trees | Proven/reference optima | Timeouts | Errors / memory exits | Median wall, valid (s) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for (track, config), group in sorted(grouped.items()):
        m = metrics(group)
        lines.append(
            "| {} | {} | {}/{} | {} | {} | {} | {} | {} |".format(
                track,
                config,
                m["completed"],
                m["expected"],
                m["valid"],
                m["optimal"],
                m["timeouts"],
                m["errors"],
                (
                    "{:.3f}".format(m["median_wall"])
                    if m["median_wall"] is not None
                    else "—"
                ),
            )
        )
    lines += [
        "",
        "Median times cover valid returned solutions only; configurations can solve different subsets. "
        "Use the per-instance CSV for paired timing comparisons. `solve_seconds` measures the whole "
        "`get_solution` call (including model construction); `model_build_seconds` times its classic "
        "model builders, and `solver_seconds` is the library-reported solver runtime.",
        "",
        "## Original private instances",
        "",
        "| Track | Configuration | Finished / expected | Optima | Quality index (%) |",
        "|---|---|---:|---:|---:|",
    ]
    for (track, config), group in sorted(grouped.items()):
        m = metrics([r for r in group if r["split"] == "private"])
        lines.append(
            "| {} | {} | {}/{} | {} | {} |".format(
                track,
                config,
                m["completed"],
                m["expected"],
                m["optimal"],
                (
                    "{:.3f}".format(m["quality"])
                    if m["quality"] is not None
                    else "pending"
                ),
            )
        )
    lines += [
        "",
        "Quality is `100 × mean(reference_upper / objective)` over the full requested private set, "
        "with zero for missing/invalid solutions. A completed sweep is required before publishing this score. "
        "The upper reference is not necessarily an optimum for Track 3.",
        "",
        "## Historical PACE participants",
        "",
        "The following are published 2018 results, not reruns on Snellius. "
        "[Source: official PACE report, sections 4.1–4.3](https://pacechallenge.org/files/PACE18-report.pdf).",
        "",
        "| Track | Place | Team | Published private-set result |",
        "|---|---:|---|---:|",
    ]
    with Path(__file__).with_name("pace2018_reference.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            score = row["value"] + (
                "/100 solved" if row["track"] != "3" else "% quality"
            )
            lines.append(
                "| {} | {} | {} | {} |".format(
                    row["track"], row["rank"], row["team"], score
                )
            )
    lines += [
        "",
        "The exact-track solved counts provide historical context on the same private instances. "
        "They do not establish a competition ranking or speedup: hardware, software and resource limits differ. "
        "Track 3's current archive omits one formerly private instance and its reference bounds need not be "
        "the same as the 2018 scoring bounds, so its quality index is not directly interchangeable with "
        "the published competition score. The table above does not contain rerun measurements.",
        "",
    ]
    if comparison:
        lines += comparison_section(comparison, output)
    lines += [
        "## Reproducibility",
        "",
    ]
    contexts = {
        json.dumps(
            {
                k: r["run"].get(k)
                for k in (
                    "commit",
                    "dataset_commit",
                    "python",
                    "packages",
                    "cpu",
                    "time_limit",
                    "memory_gb",
                )
            },
            sort_keys=True,
        )
        for r in rows
    }
    for context in sorted(contexts):
        lines += ["```json", json.dumps(json.loads(context), indent=2), "```", ""]
    failures = Counter(
        (r["config"], r.get("phase", "unknown"))
        for r in rows
        if r["status"] == "timeout"
    )
    if failures:
        lines += [
            "## Where wall deadlines expired",
            "",
            "| Configuration | Phase | Instances |",
            "|---|---|---:|",
        ]
        for (config, phase), count in sorted(failures.items()):
            lines.append("| {} | {} | {} |".format(config, phase, count))
        lines.append("")
    errors = Counter(
        (r["config"], r.get("phase", "unknown"), r.get("error", r["status"]))
        for r in rows
        if r["status"] not in {"optimal", "feasible", "timeout", "pending"}
    )
    if errors:
        lines += [
            "## Rejected results and resource failures",
            "",
            "| Configuration | Phase | Reason | Instances |",
            "|---|---|---|---:|",
        ]
        for (config, phase, reason), count in sorted(errors.items()):
            lines.append(
                "| {} | {} | {} | {} |".format(
                    config,
                    phase,
                    reason.replace("|", "\\|").replace("\n", " "),
                    count,
                )
            )
        lines.append("")
    lines += [
        "The [per-instance results](results.csv) include checksums, versions, objective/proof distinctions, "
        "memory and phase timings. Raw JSON and original-edge solution certificates remain in the HPC "
        "run directory. The runner and Slurm script live in `benchmarks/pace2018.py` and "
        "`benchmarks/slurm_pace2018.sh`.",
        "",
    ]
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--comparison-results", type=Path)
    args = parser.parse_args()
    export(
        load_rows(args.results),
        args.output,
        load_rows(args.comparison_results) if args.comparison_results else None,
    )


if __name__ == "__main__":
    main()
