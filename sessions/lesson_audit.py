#!/usr/bin/env python3
"""
Lesson integrity audit for the Frontier Lab curriculum.
Classifies every EXPECTED lesson file as:
  OK        — exists, correct quest-id, no leftover vmap-template markers, 7 sections / 4 quiz / 3 demos
  MISSING   — file does not exist
  LEFTOVER  — exists but still contains day-03-vmap template content (verify never ran / failed)
  DEGRADED  — exists but structurally off (wrong section/quiz/demo counts, orphaned demo keys, Chinese chars)
Prints a machine-readable summary and per-file reasons.
"""
import os, re, glob, json, sys

BASE = os.path.join(os.path.dirname(__file__))  # sessions/

# (week, [ (day-file-slug, quest-id) ... ])  — the full expected manifest.
MANIFEST = {
 1:[("day-01-jax-immutability","w01-d01-jax"),("day-02-prng-keys","w01-d02-prng"),("day-03-vmap","w01-d03-vmap"),("day-04-jit","w01-d04-jit"),("day-05-flax-optax","w01-d05-flax"),("day-06-vit-capstone","w01-d06-vit")],
 2:[("day-01-rooflines","w02-d01-roofline"),("day-02-tpu-architecture","w02-d02-tpu"),("day-03-sharding","w02-d03-sharding"),("day-04-multi-device","w02-d04-multidevice"),("day-05-memory-footprint","w02-d05-memory"),("day-06-distributed-checkpointing","w02-d06-distckpt")],
 3:[("day-01-transformer-arithmetic","w03-d01-arith"),("day-02-qkv-matrices","w03-d02-qkv"),("day-03-kv-cache","w03-d03-kv"),("day-04-production-code","w03-d04-prodcode"),("day-05-pretraining-logic","w03-d05-pretrain"),("day-06-scaling-exercises-rehearsal","w03-d06-examrehearsal")],
 4:[("day-01-data-parallel-fsdp","w04-d01-fsdp"),("day-02-tensor-parallel","w04-d02-tp"),("day-03-pipeline-parallel","w04-d03-pp"),("day-04-llama3-tpu","w04-d04-llama3"),("day-05-distillation","w04-d05-distill"),("day-06-pytorch-sync-consolidation","w04-d06-consolidation")],
 5:[("day-01-kaplan-paradigm","w05-d01-kaplan"),("day-02-chinchilla-correction","w05-d02-chinchilla"),("day-03-isoflops-methodology","w05-d03-isoflops"),("day-04-power-law-derivation","w05-d04-powerlaw"),("day-05-data-wall","w05-d05-datawall"),("day-06-scaling-simulator-visualization","w05-d06-simviz")],
 6:[("day-01-inference-mechanics","w06-d01-infermech"),("day-02-memory-wall","w06-d02-memwall"),("day-03-batching-economics","w06-d03-batching"),("day-04-serving-llama3-tpus","w06-d04-serving"),("day-05-quantization-intro","w06-d05-llmint8"),("day-06-inference-calculator-roofline","w06-d06-infercalc")],
 7:[("day-01-moe-fundamentals","w07-d01-moefund"),("day-02-moe-scaling-deviations","w07-d02-moescale"),("day-03-load-balancing-problem","w07-d03-loadbal"),("day-04-distributed-moe-routing","w07-d04-moeroute"),("day-05-moe-vs-dense-benchmarks","w07-d05-moevsdense"),("day-06-routing-design-deep-synthesis","w07-d06-routesynth")],
 8:[("day-01-comprehensive-profiling","w08-d01-profiling"),("day-02-systems-review","w08-d02-sysreview")],
 9:[("day-01-infrastructure-setup","w09-d01-infra"),("day-02-custom-tokenizer-design","w09-d02-tokenizer"),("day-03-synthetic-data-generation","w09-d03-syndata"),("day-04-static-shape-engineering","w09-d04-staticshape"),("day-05-architecture-sizing","w09-d05-archsize"),("day-06-training-loop-baseline-training","w09-d06-trainbaseline")],
 10:[("day-01-evaluation-infrastructure","w10-d01-evalinfra"),("day-02-profiling-step-time","w10-d02-stepprofile"),("day-03-hardware-optimization","w10-d03-hwopt"),("day-04-isoflops-experiment-design","w10-d04-isoflopsdesign"),("day-05-automation-scripting","w10-d05-automation"),("day-06-execution-data-aggregation","w10-d06-execagg")],
 11:[("day-01-plotting-parabolas","w11-d01-parabolas"),("day-02-deriving-scaling-law","w11-d02-derivelaw"),("day-03-scientific-documentation","w11-d03-scidocs"),("day-04-moe-architecture-upgrade","w11-d04-moeupgrade"),("day-05-moe-baseline-training","w11-d05-moebaseline"),("day-06-moe-isoflops-matrix-law-derivation","w11-d06-moeisolaw")],
 12:[("day-01-report-finalization","w12-d01-reportfinal"),("day-02-repository-cleansing","w12-d02-repocleanse")],
 13:[("day-01-pallas-abstraction","w13-d01-pallas"),("day-02-basic-kernel-construction","w13-d02-basickernel"),("day-03-hardware-conflicts","w13-d03-hwconflict"),("day-04-flashattention-paradigm","w13-d04-flashattn"),("day-05-hardware-evolution","w13-d05-hwevo"),("day-06-moe-kernel-challenge-performance-proof","w13-d06-moekernel")],
 14:[("day-01-modern-kernel-dsls","w14-d01-tkintro"),("day-02-tensor-memory-acceleration","w14-d02-tma"),("day-03-worker-overlapping","w14-d03-overlap"),("day-04-thunderkittens-architecture","w14-d04-tkarch"),("day-05-comparative-benchmarking","w14-d05-compbench"),("day-06-cuda-tk-setup-implementation","w14-d06-tkimpl")],
 15:[("day-01-quantization-theory-int8","w15-d01-int8theory"),("day-02-incoherent-processing-quip","w15-d02-quip"),("day-03-extreme-compression-quip-sharp","w15-d03-quipsharp"),("day-04-additive-quantization-qtip","w15-d04-qtip"),("day-05-memory-economics-synthesis","w15-d05-memecon"),("day-06-ptq-implementation-perplexity-benchmarking","w15-d06-ptqperplex")],
 16:[("day-01-decoding-innovations","w16-d01-decoding"),("day-02-cache-eviction-snapkv","w16-d02-snapkv"),("day-03-gqa-mqa","w16-d03-gqamqa"),("day-04-speculative-decoding","w16-d04-specdecode"),("day-05-ring-attention","w16-d05-ringattn"),("day-06-systems-review-essay","w16-d06-hwlottery")],
 17:[("day-01-consolidation-publishing","w17-d01-consolidation")],
 19:[("day-01-adrs-paradigm","w19-d01-adrsparadigm"),("day-02-adrs-architecture","w19-d02-adrsarch"),("day-03-case-studies-automation","w19-d03-casestudies"),("day-04-less-is-more-principle","w19-d04-lessismore"),("day-05-reward-hacking-threat","w19-d05-rewardhack"),("day-06-evaluation-engineering-holdout","w19-d06-evalholdout")],
 20:[("day-01-mathematical-discovery-funsearch","w20-d01-funsearch"),("day-02-formal-theorem-proving-alphageometry","w20-d02-alphageom"),("day-03-evolutionary-prompts-evoprompt","w20-d03-evoprompt"),("day-04-data-engineering-automation-ruleflow","w20-d04-ruleflow"),("day-05-agentic-scaffolding","w20-d05-scaffold"),("day-06-adrs-implementation-simulation-execution","w20-d06-adrsimpl")],
 21:[("day-01-formal-verification-refinement","w21-d01-tlaplus")],
 23:[("day-01-screencast-production-code","w23-d01-screencode"),("day-02-screencast-production-math","w23-d02-screenmath"),("day-03-kernel-systems-report","w23-d03-kernelreport"),("day-04-portfolio-assembly","w23-d04-portfolio")],
 24:[("day-01-cold-email-engineering","w24-d01-coldemail"),("day-02-targeted-leadership-outreach","w24-d02-leadershipoutreach"),("day-03-researcher-network-outreach","w24-d03-researchoutreach"),("day-04-interview-preparation","w24-d04-interviewprep")],
}

# vmap-template fingerprints. A file is contaminated if it has the vmap quest id, the vmap
# quiz text, OR all three vmap playground demo keys together (a lone 'loop' key is legit).
STRONG_MARKERS = ['w01-d03-vmap', 'jax.vmap(f)(X)']
VMAP_DEMO_TRIO = ['data-demo="loop"', 'data-demo="vmap"', 'data-demo="inaxes"']

# week-1 days 01,02,04,05 + day-03-vmap are pristine hand-made originals — never flag them.
PRISTINE = {(1,'day-01-jax-immutability'),(1,'day-02-prng-keys'),(1,'day-03-vmap'),
            (1,'day-04-jit'),(1,'day-05-flax-optax')}

def classify(week, slug, qid, path):
    reasons = []
    if not os.path.exists(path):
        return "MISSING", ["file does not exist"]
    t = open(path, encoding="utf-8").read()
    if (week, slug) not in PRISTINE:
        for m in STRONG_MARKERS:
            if m in t:
                reasons.append(f"leftover-marker: {m}")
        if all(m in t for m in VMAP_DEMO_TRIO):
            reasons.append("leftover-marker: vmap demo trio (loop/vmap/inaxes)")
    # correct quest id present (informational — a valid-but-different id is not a defect)
    if f'data-quest-id="{qid}"' not in t:
        reasons.append(f"note: quest-id != manifest guess ({qid})")
    # structural counts
    secs = t.count('class="sec"')
    gotit = t.count('class="gotit"')
    demos = t.count('data-demo=')
    quiz = len(re.findall(r'\{q:', t))
    if secs != 7: reasons.append(f"sections={secs} (want 7)")
    if gotit != 7: reasons.append(f"gotit={gotit} (want 7)")
    if demos != 3: reasons.append(f"data-demo={demos} (want 3)")
    if quiz != 4: reasons.append(f"quiz={quiz} (want 4)")
    if re.search(r'[一-鿿]', t):
        reasons.append("contains Chinese characters")
    # orphaned demo keys: every data-demo value must appear as a DEMOS key (quoted or bare)
    demo_vals = set(re.findall(r'data-demo="([^"]+)"', t))
    for dv in demo_vals:
        if not re.search(r'''['"]?''' + re.escape(dv) + r'''['"]?\s*:\s*\{''', t):
            reasons.append(f"orphaned demo key '{dv}' (button has no DEMOS entry)")
    # scroll-reveal format present, no click-stepper leftovers
    if 'var BUILD=' not in t: reasons.append("missing BUILD array (not new format)")
    if 'renderStep' in t or 'data-sec="code"' in t: reasons.append("old click-stepper leftover")
    leftover = any(r.startswith("leftover-marker") for r in reasons)
    hard = [r for r in reasons if not r.startswith("note:")]
    if leftover:
        return "LEFTOVER", reasons
    if hard:
        return "DEGRADED", hard
    return "OK", []

def main():
    buckets = {"OK":[], "MISSING":[], "LEFTOVER":[], "DEGRADED":[]}
    for week, days in MANIFEST.items():
        for slug, qid in days:
            path = os.path.join(BASE, f"week-{week:02d}", f"{slug}.html")
            status, reasons = classify(week, slug, qid, path)
            rel = f"week-{week:02d}/{slug}.html"
            buckets[status].append((rel, qid, reasons))
    total = sum(len(v) for v in buckets.values())
    print(f"TOTAL expected lessons: {total}")
    for k in ("OK","MISSING","LEFTOVER","DEGRADED"):
        print(f"  {k}: {len(buckets[k])}")
    for k in ("MISSING","LEFTOVER","DEGRADED"):
        if buckets[k]:
            print(f"\n=== {k} ===")
            for rel, qid, reasons in buckets[k]:
                print(f"  {rel}  [{qid}]")
                for r in reasons:
                    print(f"       - {r}")
    # emit machine-readable recovery set
    recover = [rel for k in ("MISSING","LEFTOVER","DEGRADED") for rel,_,_ in buckets[k]]
    open(os.path.join(BASE, "_recover_set.json"), "w").write(json.dumps(recover, indent=2))
    print(f"\nRecovery set ({len(recover)}) written to sessions/_recover_set.json")
    return 0 if not recover else 1

if __name__ == "__main__":
    sys.exit(main())
