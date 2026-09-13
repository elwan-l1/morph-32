"""Audit the completed quality/timing cohort independently of pending task benchmarks."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from morph32.paths import write_json
from morph32.statistics import timing_summary

PROFILES = [
    "native",
    "morph32-3s",
    "morph32-c",
    "independent-3s",
    "affine-3bit",
    "morph32-3s-uncompressed",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    signatures, rows, hashes = {}, {}, {}
    for name in PROFILES:
        rows[name] = {}
        for domain in ["text", "code"]:
            path = args.folder / f"{name}-{domain}.json"
            raw = path.read_bytes()
            hashes[path.name] = hashlib.sha256(raw).hexdigest()
            r = json.loads(raw)
            assert r["status"] == "complete"
            assert (
                len(r["token_nll"])
                == r["corpus"]["tokens"]
                == r["quality"]["scored_tokens"]
                == 4096
            )
            nll = sum(r["token_nll"]) / 4096
            assert math.isclose(nll, r["quality"]["nll"], rel_tol=1e-6)
            assert math.isclose(math.exp(nll), r["quality"]["perplexity"], rel_tol=1e-6)
            prov = r["provenance"]
            signature = dict(
                corpus=r["corpus"],
                tokenizer=r["tokenizer_sha256"],
                device=prov["device"],
                platform=prov["platform"],
                packages=prov["packages"],
                source=prov["source_sha256"],
                settings={
                    k: prov[k]
                    for k in [
                        "decode_variant",
                        "prefill_variant",
                        "decode_values",
                        "fuse_gate_up",
                        "metadata_variant",
                        "prefill_block_m",
                    ]
                },
            )
            if name == "native":
                signatures[domain] = signature
            assert signature == signatures[domain], (name, domain, "protocol mismatch")
            if name != "native":
                assert len(r["token_kl"]) == len(r["token_agreement"]) == 4096
                assert math.isclose(
                    sum(r["token_kl"]) / 4096,
                    r["quality"]["mean_kl_reference_to_candidate"],
                    abs_tol=1e-6,
                )
                assert math.isclose(
                    sum(r["token_agreement"]) / 4096,
                    r["quality"]["reference_token_agreement"],
                    abs_tol=1e-6,
                )
                assert math.isclose(
                    r["quality"]["nll"] - rows["native"][domain]["nll"],
                    r["quality"]["delta_nll_nats_per_token"],
                    abs_tol=1e-6,
                )
            rows[name][domain] = r["quality"]
            if domain == "text":
                runs = r["generation_runs"]
                assert len(runs) == 10 and all(len(v["token_ids"]) == 64 for v in runs)
                rows[name].update(
                    file_bytes=r["inventory"]["file_bytes"],
                    resident_parameter_bytes=r["resident_parameter_bytes"],
                    generation=timing_summary([v["generation_tps"] for v in runs]),
                    prefill=timing_summary([v["prompt_tps"] for v in runs]),
                )
    write_json(
        args.folder / "quality-audit.json",
        dict(
            status="complete",
            scope="Six-format quality and timing cohort only; downstream capability results are separate and may still be running.",
            profiles=rows,
            source_sha256=hashes,
            auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            timing_limitation="Sequential fresh-process measurements. Small differences can reflect machine drift; ten repetitions and IQR do not remove between-session confounding. Affine control uses an intentionally simple prefill kernel.",
        ),
    )
    print(
        "Verified 12 quality reports, 49,152 scored token positions, and 60 timed generation runs."
    )


if __name__ == "__main__":
    main()
