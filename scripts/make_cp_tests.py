"""Generate hidden tests for the cp seed problems.

    python scripts/make_cp_tests.py [--backend docker] [--seed N] [slug ...]

For each problem this makes test inputs, runs the reference solution on them through
the code runner, and writes data/private/cp/<slug>.json as a list of {input, output}.
Then it judges the reference solution against the new file and prints the verdict.

The reference solutions are answers, so they live in data/private/cp/ref/<slug>.py,
which is gitignored like the tests. Only the input generators are public. The seed is
random unless given, so the hidden inputs cannot be rebuilt from this file.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import secrets
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import challenges, runner  # noqa: E402
from app.config import Settings  # noqa: E402

PRIVATE = ROOT / "data" / "private" / "cp"
GEN_TIME_LIMIT_S = 10      # generous, the reference only has to finish
BIG = 10 ** 9


def fmt(*lines) -> str:
    return "".join(" ".join(map(str, ln)) + "\n" for ln in lines)


def gen_sum_a_list(rng: random.Random) -> list[str]:
    cases = [fmt([1], [5]), fmt([1], [-BIG]), fmt([3], [0, 0, 0]),
             fmt([4], [BIG] * 4), fmt([5], [-1, -2, -3, -4, -5])]
    for n in (10, 100, 1000, 1000):
        cases.append(fmt([n], [rng.randint(-BIG, BIG) for _ in range(n)]))
    return cases


def gen_range_sums(rng: random.Random) -> list[str]:
    def case(n, q, lo=-BIG, hi=BIG):
        a = [rng.randint(lo, hi) for _ in range(n)]
        qs = []
        for _ in range(q):
            l, r = sorted((rng.randint(1, n), rng.randint(1, n)))
            qs.append((l, r))
        return fmt((n, q), a, *qs)

    cases = [fmt((1, 1), [7], (1, 1)), fmt((3, 3), [-1, -2, -3], (1, 3), (2, 2), (3, 3)),
             case(10, 10), case(100, 100, -5, 5), case(2000, 2000)]
    n, q = 200_000, 50_000
    cases.append(case(n, q))
    cases.append(fmt((n, q), [BIG] * n, *([(1, n)] * q)))   # longest ranges, biggest sums
    return cases


def gen_pair_count(rng: random.Random) -> list[str]:
    def case(n, lo, hi, k=None):
        a = [rng.randint(lo, hi) for _ in range(n)]
        if k is None:
            k = rng.choice(a) + rng.choice(a)
        return fmt((n, k), a)

    n = 200_000
    return [fmt((1, 2), [1]), fmt((2, 2), [1, 1]), fmt((4, 0), [0, 0, 0, 0]),
            fmt((5, 100), [1, 2, 3, 4, 5]), case(50, -10, 10), case(5000, 0, 100),
            case(n, -BIG, BIG), case(n, 0, 1000), fmt((n, 6), [3] * n)]


GENERATORS = {
    "sum-a-list": gen_sum_a_list,
    "range-sums": gen_range_sums,
    "pair-count": gen_pair_count,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("slugs", nargs="*", default=list(GENERATORS))
    ap.add_argument("--backend", default="docker")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    os.chdir(ROOT)   # seeds point at tests by repo-relative path

    settings = Settings(_env_file=None, runner_backend=args.backend)
    seeds = {row["slug"]: row for row in json.loads((ROOT / "seeds" / "cp.json").read_text())}
    seed = args.seed if args.seed is not None else secrets.randbits(64)
    rng = random.Random(seed)
    for slug in args.slugs:
        ref = (PRIVATE / "ref" / f"{slug}.py").read_text(encoding="utf-8")
        tests = []
        for i, stdin in enumerate(GENERATORS[slug](rng), 1):
            res = runner.run_python(ref, stdin, time_limit_s=GEN_TIME_LIMIT_S,
                                    memory_mb=512, settings=settings)
            if res.status != "ok":
                print(f"{slug} test {i}: reference failed with {res.status}\n{res.stderr}")
                return 1
            tests.append({"input": stdin, "output": res.stdout})
        out = PRIVATE / f"{slug}.json"
        out.write_text(json.dumps(tests), encoding="utf-8")

        row = {**seeds[slug], "id": 0}
        ch = challenges.load(row)
        ch.prepare(settings, lambda: None)
        d = ch.evaluate(ref).detail
        print(f"{slug}: wrote {len(tests)} tests to {out.relative_to(ROOT)}, "
              f"reference {d['verdict']} {d['passed']}/{d['total']} in {d['time_ms']} ms max")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
