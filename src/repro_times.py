"""
Wall-clock accounting for the reproducibility pass (reads repro_jobs.tsv written by scripts/repro/submit_all.sh and sacct).
Reports per job: queue wait, run time, state, GPU model (from the job log), plus: total wall-clock from the first submission to the last job's
end, the critical path, and total compute time. Step A and Step C times come from their own reports (repro_step_a.json / repro_report.json).

Usage:  python src/repro_times.py [--jobs repro_jobs.tsv]
"""
import argparse
import glob
import json
import os
import re
import subprocess
from datetime import datetime

import pandas as pd


def ts(x):
    return datetime.fromisoformat(x) if x and x not in ("Unknown", "None") else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", default="repro_jobs.tsv")
    args = ap.parse_args()
    jobs = pd.read_csv(args.jobs, sep="\t")
    cp = subprocess.run(["sacct", "-j", ",".join(map(str, jobs.jobid)), "-X", "-P", "--format=JobID,JobName,State,Submit,Start,End,Elapsed,ElapsedRaw,NodeList"],
                        capture_output=True, text=True)
    s = pd.read_csv(pd.io.common.StringIO(cp.stdout), sep="|")
    rows = []
    for r in s.itertuples():
        sub, st, en = ts(r.Submit), ts(r.Start), ts(r.End)
        gpu = ""
        for f in glob.glob(f"{r.JobName}_{r.JobID}.out"):
            m = re.search(r"GPU: (.+)", open(f, errors="ignore").read())
            gpu = m.group(1).strip() if m else ""
        rows.append({"job": r.JobName, "id": r.JobID, "state": r.State, "queue_wait_min": (st - sub).total_seconds() / 60 if st and sub else None,
                     "run_min": r.ElapsedRaw / 60, "node": r.NodeList, "gpu": gpu, "submit": sub, "start": st, "end": en})
    d = pd.DataFrame(rows)
    print(d[["job", "id", "state", "queue_wait_min", "run_min", "gpu"]].round(1).to_string(index=False))
    ok = d.state.eq("COMPLETED").all()
    first_sub, last_end = d.submit.min(), d.end.max()
    first_start = d.start.min()
    print(f"\nall jobs COMPLETED: {ok}")
    print(f"total wall-clock, first submission -> last job end: {(last_end - first_sub).total_seconds() / 3600:.2f} h "
          f"(includes queue waits; first job started {(first_start - first_sub).total_seconds() / 60:.0f} min after submission)")
    print(f"wall-clock from the first job START to the last job end (compute-bound): {(last_end - first_start).total_seconds() / 3600:.2f} h")
    print(f"sum of run times (total compute): {d.run_min.sum() / 60:.2f} h; longest single job: {d.loc[d.run_min.idxmax(), 'job']} {d.run_min.max() / 60:.2f} h")
    if os.path.exists("repro_step_a.json"):
        print(f"Step A (data verification) elapsed: {json.load(open('repro_step_a.json'))['elapsed_sec'] / 60:.1f} min")
    if os.path.exists("repro_report.json"):
        print(f"Step C (comparison) elapsed: {json.load(open('repro_report.json'))['elapsed_sec'] / 60:.1f} min")


if __name__ == "__main__":
    main()
