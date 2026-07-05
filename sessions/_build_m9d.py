#!/usr/bin/env python3
"""Builder for M9d — First Real Multi-Device Run (the cost/quota gate).
Two lessons + review, rendered via _lesson_gen. Run: python3 sessions/_build_m9d.py

M9d is a small COST/QUOTA discipline gate between M9c (the sharding calculator)
and M9e (orchestration): go from the simulator to one real multi-device job,
sanity-check measured vs predicted step time, then pass a ~$50 cost gate before
scaling up.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _lesson_gen import write_lesson, write_review

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "week-m9d")
os.makedirs(OUT, exist_ok=True)

def L(name, spec): write_lesson(os.path.join(OUT, name), spec)

# ---------------- Day 1 — Your First Real Sharded Run ----------------
L("day-01-real-run.html", dict(
  qid="m9d-d01-run", title="Module 9d · Day 1 — Your First Real Sharded Run", nav_title="Spiral · M9d Day 1",
  eyebrow="Module 9d · Scale · Day 1", h1="Your First Real Sharded Run",
  lead="In M9c you built a sharding calculator that <em>predicts</em> how long a step should take across many devices. Today you press the button for real: one small job on real hardware, one step measured, and one comparison — did the machine agree with your math? This is the moment your plan meets reality, and the discipline that keeps a wrong plan from burning money.",
  goal="<b>🎯 By the end, you'll be able to:</b> run one small sharded job, measure real step time, compare it to the calculator's prediction with a simple ratio, and follow the dry-run-then-scale rule that catches a broken run before it gets expensive.",
  prev_href="../index.html", prev_label="Curriculum Map", next_href="day-02-cost-gate.html", next_label="The Cost & Quota Gate",
  sections=[
    {"title":"From simulator to a real button-press","body":
     '''<div class="callout c-info"><span class="ic">🧭</span><div><b>What this assumes:</b> M9c — you have a sharding calculator that predicts step time from FLOPs, memory, and network cost. Nothing new about hardware is needed; we just run one real job.</div></div>
     <h4>Why a real run at all</h4>
     <p>A calculator is a model of reality, not reality. It assumes clean numbers: perfect overlap, no stalls, the compiler does exactly what you expect. Real hardware adds surprises — a slow data pipeline, a bad memory layout, a compile that quietly does the wrong thing. The only way to know your plan holds is to run it once, small, and look.</p>
     <h4>The one number that matters first</h4>
     <p>The <span class="term" data-tip="The wall-clock time to do one full training update: forward pass, backward pass, and the weight update, for one batch.">step time</span> — how many seconds one training step takes — is your headline number. Your M9c calculator gave you a <span class="term" data-tip="The step time your calculator says the hardware should take, computed from FLOPs, memory bandwidth, and network cost.">predicted step time</span>. A real run gives you a <span class="term" data-tip="The step time you actually observe on real hardware, read from a timer around one training step.">measured step time</span>. The whole lesson is: get both, and compare.</p>
     <h4>The discipline: dry-run, then scale</h4>
     <p>You never jump straight to the full-size job. First you do a <span class="term" data-tip="A tiny, short, cheap version of your real job — few steps, small batch — run only to check that everything works and the numbers look sane before spending real money.">dry-run</span>: a few steps, small, cheap. You check it works and the step time is sane. Only then do you scale up. A dry-run costs cents; a broken full run costs your whole budget.</p>''',
     "gotit":"Got the setup"},
    {"title":"A dress rehearsal before opening night","body":
     '''<div class="relate">
       <div class="card"><span class="big">🎭</span><h5>The dress rehearsal</h5><p>Before a big show sells 2000 tickets, the cast does one full run-through to an empty room. It costs almost nothing and catches the broken prop, the missed cue, the light that never turns on. You would never open the paid show without it.</p></div>
       <div class="card"><span class="big">⏱️</span><h5>The stopwatch check</h5><p>A runner times one practice lap before racing. If the practice lap is wildly slower than their usual pace, something is wrong — a pulled muscle, the wrong shoes — and they find out for free, not mid-race.</p></div>
     </div>
     <p><strong>In one line:</strong> a dry-run is a cheap dress rehearsal for your training job, and the step-time comparison is the stopwatch that tells you whether the rehearsal went as planned.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div>Where the analogy breaks: a dress rehearsal mostly finds obvious, visible mistakes. A bad training run can look perfectly fine — no error, loss going down — while secretly running 10× too slow. You have to <i>read the timer</i>, because the failure is silent.</div></div>''',
     "gotit":"Got the picture"},
    {"title":"Predicted vs measured, in three steps","body":
     '''<p>Watch a real dry-run: read the calculator's prediction, time one real step, then compute the ratio that tells you if reality agrees. <strong>Click all three.</strong></p>'''},
    {"title":"The prediction-vs-measured check","body":
     '''<h4>Step 1 — the idea in words</h4>
     <p>You have two step times: what the calculator predicted, and what the hardware measured. Divide the measured by the predicted. If the answer is close to 1, reality agrees with your plan. If it is much bigger than 1, something is slowing you down and you must find it <em>before</em> scaling.</p>
     <h4>Step 2 — the formula</h4>
     <div class="callout c-info"><span class="ic">🧮</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.9">
       ratio = measured_step_time / predicted_step_time<br>
       <span class="dim"># ratio ≈ 1.0–1.5  → healthy, small real-world overhead</span><br>
       <span class="dim"># ratio ≫ 1  (say 3× or 10×) → something is wrong, stop and debug</span>
     </div></div>
     <p>Symbols: <code>measured_step_time</code> = seconds you timed on real hardware. <code>predicted_step_time</code> = seconds the M9c calculator gave you. <code>ratio</code> = how many times slower reality is than the plan. A little overhead is normal; a large ratio is a red flag.</p>
     <h4>Step 3 — worked example</h4>
     <div class="callout c-info"><span class="ic">🔢</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.8">
       predicted_step_time = 0.40 s   <span class="dim"># from your M9c calculator</span><br>
       measured_step_time  = 0.52 s   <span class="dim"># timed on the real dry-run</span><br><br>
       ratio = 0.52 / 0.40 = <span class="hl">1.3</span><br>
       <span class="dim"># 1.3 → 30% overhead → healthy, safe to scale up</span>
     </div></div>
     <p>A ratio of 1.3 means real life added 30% on top of the ideal — expected, from data loading and small stalls. You would happily scale that up. A ratio of 10 would mean each step is ten times too slow, and scaling it would waste ten times the money.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div><b>Silent failure — a bad compile silently 10×'s step time.</b> Modern accelerators <span class="term" data-tip="Turning your model code into an optimized program the accelerator runs. A good compile fuses operations and lays out memory well; a bad one does not.">compile</span> your model into an optimized program. If a shape is wrong or a setting is off, the compiler can fall back to a slow path — no error, no crash, loss still goes down. Only the step time gives it away: instead of 0.4 s it is 4 s. If you never compare measured to predicted, you scale this 10× waste to the full cluster and pay 10× for the same work. The timer is the only alarm.</div></div>
     <div class="callout c-warn"><span class="ic">⚖️</span><div><b>Trade-off — a small verification run vs jumping to full scale.</b> A verification dry-run costs a little time and a few cents, and it delays your "real" start by minutes. Jumping straight to full scale saves that setup time — <i>if</i> everything is perfect. But if anything is wrong, the full run burns your whole budget before you notice. The dry-run trades a tiny, certain cost for protection against a large, likely one. At frontier scale, always pay the small cost first.</div></div>''',
     "gotit":"Got the check"},
    {"title":"Building the verification run","body":
     '''<p>Here is the verification run assembled piece by piece: the prediction, the small dry-run, the timer, the ratio, and the go/no-go decision. <strong>Scroll slowly.</strong></p>'''},
    {"title":"Check yourself","body":'''<p>4 questions, instant feedback. <strong>Answer all four.</strong></p>'''},
    {"title":"Prove it with a dry-run","body":
     '''<p>Understanding is not doing. Run one real (or simulated) dry-run and compare. Pick one:</p>
     <h4>Option A · write it yourself</h4>
     <p>Create <code>experiments/foundations/m9d_dry_run.py</code>. Take your M9c predicted step time as a number. Run a tiny sharded job (even 8 steps on 2 devices, or a CPU stand-in), time one step with a wall-clock timer, and print the measured step time, the predicted step time, and the ratio. Add a rule that prints <code>"STOP — investigate"</code> if the ratio is above 3.</p>
     <h4>Option B · let Claude build it</h4>
     <div class="prompt">
       <div class="prompt-h"><span class="prompt-l">triggers <b>frontier-experiment-lab</b></span><button class="copy" type="button" data-copy="#pp">📋 copy</button></div>
       <pre class="prompt-t" id="pp">Use /frontier-experiment-lab to build my Module 9d Day 1 artifact.
Create experiments/foundations/m9d_dry_run.py: a tiny sharded (or multi-process CPU stand-in) training job. Given a predicted_step_time constant from the M9c calculator, run ~8 warmup+timed steps, measure the median step time with a wall-clock timer, and print measured, predicted, and ratio = measured/predicted. Print "STOP - investigate (ratio {ratio})" if ratio > 3, else "OK to scale". Note that a bad compile shows up as a large ratio.</pre>
     </div>
     <div class="callout c-info"><span class="ic">📝</span><div>Write one line in <code>sessions/week-m9d/day-01-log.md</code>: what ratio did you get, and was it safe to scale?</div></div>''',
     "gotit":"Done — on to Day 2"},
  ],
  demos={
    "predict":{"btn":"① read the prediction","html":'''<span class="prompt">&gt;&gt;&gt;</span> predicted = m9c_calculator.step_time()
<span class="hl">predicted = 0.40 s</span>   <span class="dim"># from your M9c sharding calculator</span>
<span class="dim"># this is the ideal: perfect overlap, no stalls</span>''',
      "take":"<b>①  Start from the plan.</b> Your M9c calculator says one step should take 0.40 s. That is the number reality has to beat or match."},
    "measure":{"btn":"② time one real step","html":'''<span class="prompt">&gt;&gt;&gt;</span> t0 = time(); train_one_step(); t1 = time()
<span class="prompt">&gt;&gt;&gt;</span> measured = t1 - t0
<span class="hl">measured = 0.52 s</span>   <span class="dim"># timed on the real dry-run</span>''',
      "take":"<b>②  Measure reality.</b> Put a stopwatch around one real training step. Here the hardware took 0.52 s — a bit more than the plan, which is normal."},
    "ratio":{"btn":"③ compute the ratio","html":'''<span class="prompt">&gt;&gt;&gt;</span> ratio = measured / predicted
<span class="prompt">&gt;&gt;&gt;</span> ratio = 0.52 / 0.40
<span class="hl">ratio = 1.3</span>   <span class="ok"># healthy → OK to scale</span>
<span class="dim"># if this were 10, you'd stop and hunt the slowdown</span>''',
      "take":"<b>③  Compare.</b> Measured ÷ predicted = 1.3. Only 30% overhead — safe to scale. A ratio of 10 would mean stop and debug before spending more."},
  },
  build=[
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='140' y='16' width='240' height='30' rx='6' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='260' y='35' fill='#1F6280'>predicted step time = 0.40 s</text></g></svg>",
     "note":"<b>Start with the prediction.</b> Your M9c calculator gives the ideal step time — perfect overlap, no stalls. This is the target reality must roughly match."},
    {"viz":"<svg viewBox='0 0 520 70'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='120' y='20' width='280' height='34' rx='6' fill='#EDE9F8' stroke='#7C6DAA' stroke-width='2'/><text x='260' y='35' fill='#5E5191'>dry-run: 8 steps, small batch,</text><text x='260' y='49' fill='#5E5191'>2 devices — cheap</text></g></svg>",
     "note":"<b>Run a tiny dry-run.</b> A few steps on a few devices — costs cents. Its only job is to check the machinery works and the timing looks sane."},
    {"viz":"<svg viewBox='0 0 520 70'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='120' y='20' width='280' height='34' rx='6' fill='#FCF3DC' stroke='#C99A12' stroke-width='2'/><text x='260' y='34' fill='#9A7208'>t0 = time(); step(); t1 = time()</text><text x='260' y='48' fill='#9A7208'>measured = 0.52 s</text></g></svg>",
     "note":"<b>Time one real step.</b> A stopwatch around the training step gives the measured step time — what the hardware actually does, surprises included."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='12' text-anchor='middle'><rect x='120' y='16' width='280' height='30' rx='6' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='260' y='36' fill='#1F6280'>ratio = 0.52 / 0.40 = 1.3</text></g></svg>",
     "note":"<b>Compute the ratio.</b> Measured ÷ predicted. Near 1 means reality agrees with the plan; far above 1 means a hidden slowdown you must find."},
    {"viz":"<svg viewBox='0 0 520 80'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='30' y='28' width='210' height='30' rx='6' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2'/><text x='135' y='47' fill='#1a5c38'>ratio ≤ 1.5 → scale up ✓</text><rect x='280' y='28' width='210' height='30' rx='6' fill='#FDE8E8' stroke='#C93B3B' stroke-width='2'/><text x='385' y='47' fill='#C93B3B'>ratio ≫ 1 → stop, debug</text></g></svg>",
     "note":"<b>The go/no-go decision.</b> A healthy ratio → scale up with confidence. A big ratio (a bad compile, a slow pipeline) → stop now, before the waste multiplies across the full cluster."},
  ],
  quiz=[
    {"q":"1. What is the first single number you compare after a dry-run?","opts":["the final loss","the measured step time vs the predicted step time","the number of devices","the file size of the checkpoint"],"ans":1,"fb":"Step time is the headline. Comparing measured to the calculator's predicted step time tells you whether reality matches your plan before you scale."},
    {"q":"2. Your calculator predicts 0.50 s/step and the dry-run measures 0.60 s/step. The ratio is:","opts":["0.83","1.2","1.1","6.0"],"ans":1,"fb":"ratio = measured / predicted = 0.60 / 0.50 = 1.2 — about 20% overhead, healthy and safe to scale."},
    {"q":"3. A run finishes with no error and the loss is going down, but each step takes 4 s instead of the predicted 0.4 s. Most likely cause?","opts":["the loss function is wrong","a bad compile fell back to a slow path — a silent 10× slowdown","the reward is negative","the batch is too small to matter"],"ans":1,"fb":"A ratio of 10 with no crash is the classic silent bad-compile: the accelerator quietly used a slow path. Only the step time reveals it — never scale it up."},
    {"q":"4. Why do a small dry-run instead of jumping straight to the full-scale job?","opts":["dry-runs give a better final model","a dry-run costs cents and catches a broken run before the full job burns the whole budget","the full job cannot be timed","dry-runs skip the compile step"],"ans":1,"fb":"The dry-run trades a tiny, certain cost for protection against a large, likely one. If something is wrong, you find out for cents instead of your whole spend."},
  ],
  fin={"em":"⏱️","h3":"Day 1 complete — your plan met reality!",
       "p":"You ran one small job, timed one real step, and compared it to the calculator: <code>ratio = measured / predicted</code>. Near 1 means scale up; far above 1 means stop and hunt the slowdown — often a silently slow compile. And you learned the dry-run-then-scale rule that catches disasters for cents. Next: <b>Day 2</b> — the cost and quota gate that decides whether you're allowed to press the big button at all."},
))
# ---------------- Day 2 — The Cost & Quota Gate ----------------
L("day-02-cost-gate.html", dict(
  qid="m9d-d02-cost", title="Module 9d · Day 2 — The Cost & Quota Gate", nav_title="Spiral · M9d Day 2",
  eyebrow="Module 9d · Scale · Day 2", h1="The Cost & Quota Gate",
  lead="Yesterday you proved one step is fast enough. Today you make sure you can actually afford — and are allowed — to run the whole thing. Real hardware is rented by the hour, budgets are finite, and clusters run out of space. This lesson is the gate: estimate the cost, request the quota ahead of time, set billing alarms, and checkpoint so a killed run is never a total loss.",
  goal="<b>🎯 By the end, you'll be able to:</b> estimate a run's dollar cost from step time, request quota in advance, set billing alerts at 50% and 90% of a cap, and checkpoint so a dead run wastes almost nothing — plus judge cheap-but-risky spot vs steady on-demand hardware.",
  prev_href="day-01-real-run.html", prev_label="Your First Real Sharded Run", next_href="review.html", next_label="Review Gate",
  sections=[
    {"title":"Money, permission, and safety","body":
     '''<div class="callout c-info"><span class="ic">🧭</span><div><b>What this assumes:</b> Day 1 — you have a trusted measured step time from a dry-run. That single number drives the whole cost estimate below.</div></div>
     <h4>Three things stand between you and the big button</h4>
     <p>Renting many accelerators costs real money, so before a large run you must clear three gates:</p>
     <ul>
       <li><b>Cost</b> — will this fit the budget? You estimate dollars from step time before you start.</li>
       <li><b>Quota</b> — a <span class="term" data-tip="A limit your cloud provider sets on how many devices (like GPUs or TPUs) you are allowed to use at once. You must request more ahead of time.">quota</span> is the cap on how many devices you may use. Big jobs need it raised, and that request takes time.</li>
       <li><b>Safety</b> — if the run dies at 90%, do you lose everything? Not if you checkpoint.</li>
     </ul>
     <h4>Cost is just step time × price</h4>
     <p>The estimate is simple arithmetic on numbers you already have: how long the run takes, how many devices, and the price per device-hour.</p>
     <h4>Ask for quota early</h4>
     <p>You cannot summon 256 accelerators on a whim. The provider must approve a <span class="term" data-tip="Asking your cloud provider, in advance, to raise your device cap so a large job is allowed to launch.">quota request</span>, which can take hours or days. Request it while you are still doing dry-runs, not the moment you want to launch.</p>''',
     "gotit":"Got the three gates"},
    {"title":"A road trip you plan before you drive","body":
     '''<div class="relate">
       <div class="card"><span class="big">⛽</span><h5>Estimate the fuel first</h5><p>Before a long drive you work out the fuel cost: distance ÷ efficiency × price. You do not just start driving and hope. A training run is the same — cost per hour × hours = the bill, worked out in advance.</p></div>
       <div class="card"><span class="big">💾</span><h5>Save your game</h5><p>In a long video game you save often. If the power cuts out, you restart from the last save, not from the very beginning. A checkpoint is exactly that save file for a training run.</p></div>
     </div>
     <p><strong>In one line:</strong> estimate the bill like fuel for a trip, and checkpoint like saving a game — so a crash costs you minutes, not the whole journey.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div>Where the analogy breaks: a game save is instant and free. A checkpoint takes real time and disk space to write, so you cannot save every single step — you pick a sensible interval, balancing safety against overhead.</div></div>''',
     "gotit":"Got the picture"},
    {"title":"Estimate the bill, in three steps","body":
     '''<p>Watch a cost estimate get built: hours from step time, dollars from device-hours, then the checkpoint math that caps your loss. <strong>Click all three.</strong></p>'''},
    {"title":"The cost estimate and the safety net","body":
     '''<h4>Step 1 — the idea in words</h4>
     <p>Total cost is the price of one device for one hour, times the number of devices, times how many hours the run lasts. Hours come from your measured step time and the number of steps. Then two safety layers: billing alerts warn you as you spend, and checkpoints cap what you lose if the run dies.</p>
     <h4>Step 2 — the formulas</h4>
     <div class="callout c-info"><span class="ic">🧮</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.9">
       hours = (steps × step_time_seconds) / 3600<br>
       cost  = price_per_device_hour × num_devices × hours<br>
       <span class="dim"># billing alert fires at 50% and 90% of your cap</span><br>
       <span class="dim"># wasted work if it dies = time since the last checkpoint</span>
     </div></div>
     <p>Symbols: <code>steps</code> = total training steps. <code>step_time_seconds</code> = your Day 1 measured step time. <code>price_per_device_hour</code> = the rent for one accelerator per hour. <code>num_devices</code> = how many you run. A <span class="term" data-tip="An automatic warning your cloud sends when your spending crosses a set level, like 50% or 90% of your budget cap.">billing alert</span> emails you before you blow the cap; a <span class="term" data-tip="A saved copy of the model weights and optimizer state at a point in training, written to disk so you can resume from there if the run stops.">checkpoint</span> lets you resume.</p>
     <h4>Step 3 — worked example</h4>
     <div class="callout c-info"><span class="ic">🔢</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.8">
       steps = 20000,  step_time = 0.5 s,  devices = 8,  price = $2/device-hr<br><br>
       hours = (20000 × 0.5) / 3600 = 10000 / 3600 ≈ <span class="hl">2.78 hr</span><br>
       cost  = 2 × 8 × 2.78 ≈ <span class="hl">$44.4</span>   <span class="dim"># under the $50 cap ✓</span><br><br>
       alerts at 50% = $25 spent, 90% = $45 spent<br>
       checkpoint every 30 min → a crash wastes ≤ 30 min ≈ $8, not $44
     </div></div>
     <p>The whole run is about $44 — safely under a $50 cap. The 90% alert at $45 would fire near the very end, exactly when you would want a heads-up. And because you checkpoint every 30 minutes, a crash at hour 2 costs you at most 30 minutes of compute, not the two hours already spent.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div><b>Silent failure — an un-checkpointed run that dies wastes the whole spend.</b> Suppose the $44 run has no checkpointing and the machine dies at 95% — a hardware fault, a preemption, a lost network link. You have no saved weights, so you must start over from step 0. The entire $44 (and the hours) are gone, and you pay it <i>again</i> to redo the same work. Nothing warned you: the run looked healthy right up to the crash. Checkpointing every 30 minutes turns that total loss into a ≤30-minute loss.</div></div>
     <div class="callout c-warn"><span class="ic">⚖️</span><div><b>Trade-off — cheap spot/preemptible vs steady on-demand.</b> <span class="term" data-tip="Spare cloud capacity rented at a big discount, but the provider can reclaim (preempt) it at any moment, killing your job.">Spot / preemptible</span> hardware is often 60–80% cheaper, but the provider can take it back with little warning, killing your run. <span class="term" data-tip="Cloud hardware you pay full price for, which the provider will not reclaim while you are using it.">On-demand</span> costs more but stays yours until you release it. Spot only makes sense <i>with good checkpointing</i>: if a preemption just means resuming from the last checkpoint, the discount is nearly free money. Without checkpointing, spot is a trap — one preemption wastes the whole spend. The trade-off is cheaper-but-interruptible against dearer-but-reliable, and checkpointing is what tips it toward spot.</div></div>''',
     "gotit":"Got cost, alerts, checkpoints"},
    {"title":"Building the cost gate","body":
     '''<p>Here is the gate assembled: the estimate, the quota request, the billing alerts, the checkpoint interval, and the go decision. <strong>Scroll slowly.</strong></p>'''},
    {"title":"Check yourself","body":'''<p>4 questions, instant feedback. <strong>Answer all four.</strong></p>'''},
    {"title":"Prove it with a cost gate","body":
     '''<p>Turn the math into a small tool you can trust. Pick one:</p>
     <h4>Option A · write it yourself</h4>
     <p>Create <code>experiments/foundations/m9d_cost_gate.py</code>. Take <code>steps</code>, <code>step_time</code>, <code>num_devices</code>, <code>price_per_device_hour</code>, and a <code>cap</code> (say $50). Print the estimated hours and cost, whether it is under the cap, and the dollar levels for 50% and 90% alerts. Given a <code>checkpoint_minutes</code>, print the worst-case wasted dollars if the run dies (time since last checkpoint × device-cost-per-hour).</p>
     <h4>Option B · let Claude build it</h4>
     <div class="prompt">
       <div class="prompt-h"><span class="prompt-l">triggers <b>frontier-experiment-lab</b></span><button class="copy" type="button" data-copy="#pp">📋 copy</button></div>
       <pre class="prompt-t" id="pp">Use /frontier-experiment-lab to build my Module 9d Day 2 artifact.
Create experiments/foundations/m9d_cost_gate.py: a cost/quota gate calculator. Inputs: steps, step_time_seconds, num_devices, price_per_device_hour, cap_dollars (default 50), checkpoint_minutes. Compute hours = steps*step_time/3600, cost = price*devices*hours. Print cost, whether cost <= cap ("GO"/"OVER BUDGET"), the 50% and 90% billing-alert dollar levels, and the worst-case wasted spend if the run dies = (checkpoint_minutes/60)*price*devices. Add a note comparing spot vs on-demand assuming checkpointing.</pre>
     </div>
     <div class="callout c-info"><span class="ic">📝</span><div>Write one line in <code>sessions/week-m9d/day-02-log.md</code>: what was your estimated cost, did it clear the $50 cap, and what is your worst-case wasted spend with checkpointing?</div></div>''',
     "gotit":"Done — on to the review gate"},
  ],
  demos={
    "hours":{"btn":"① steps → hours","html":'''<span class="prompt">&gt;&gt;&gt;</span> steps, step_time = 20000, 0.5   <span class="dim"># from Day 1</span>
<span class="prompt">&gt;&gt;&gt;</span> hours = (steps * step_time) / 3600
<span class="hl">hours = 2.78</span>   <span class="dim"># 10000 s ÷ 3600</span>''',
      "take":"<b>①  Turn steps into hours.</b> Your measured step time × the number of steps, divided by 3600, is how long the run takes: about 2.78 hours."},
    "cost":{"btn":"② hours → dollars","html":'''<span class="prompt">&gt;&gt;&gt;</span> price, devices = 2.0, 8
<span class="prompt">&gt;&gt;&gt;</span> cost = price * devices * hours
<span class="hl">cost = $44.4</span>   <span class="ok"># under the $50 cap ✓</span>''',
      "take":"<b>②  Turn hours into dollars.</b> Price per device-hour × devices × hours = about $44 — clears the $50 cap. Alerts will fire at $25 and $45."},
    "save":{"btn":"③ checkpoint caps the loss","html":'''<span class="prompt">&gt;&gt;&gt;</span> checkpoint_every = 30   <span class="dim"># minutes</span>
<span class="prompt">&gt;&gt;&gt;</span> worst_case = (30/60) * price * devices
<span class="hl">worst_case = $8</span>   <span class="dim"># if it dies, you lose ≤ this</span>
<span class="dim"># no checkpoint → you'd lose the whole $44</span>''',
      "take":"<b>③  Cap the loss.</b> Checkpointing every 30 min means a crash wastes at most 30 min of compute (~$8), not the whole $44. This is what makes cheap spot hardware safe."},
  },
  build=[
    {"viz":"<svg viewBox='0 0 520 70'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='90' y='20' width='340' height='34' rx='6' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='260' y='34' fill='#1F6280'>cost = price × devices × hours</text><text x='260' y='48' fill='#1F6280'>= 2 × 8 × 2.78 ≈ $44</text></g></svg>",
     "note":"<b>Estimate the cost first.</b> From your Day 1 step time: hours, then dollars. $44 for this run — a number you check against the budget before touching the launch button."},
    {"viz":"<svg viewBox='0 0 520 70'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='120' y='20' width='280' height='34' rx='6' fill='#EDE9F8' stroke='#7C6DAA' stroke-width='2'/><text x='260' y='34' fill='#5E5191'>request quota for 8 devices</text><text x='260' y='48' fill='#5E5191'>ahead of time (can take hours)</text></g></svg>",
     "note":"<b>Request quota early.</b> The provider caps how many devices you may use. Ask for the raise while you are still doing dry-runs, not at launch — approval is not instant."},
    {"viz":"<svg viewBox='0 0 520 80'><g font-family='monospace' font-size='11' text-anchor='middle'><line x1='60' y1='55' x2='460' y2='55' stroke='#6B645E' stroke-width='2'/><line x1='260' y1='48' x2='260' y2='62' stroke='#C99A12' stroke-width='2'/><text x='260' y='40' fill='#9A7208'>50% = $25</text><line x1='420' y1='48' x2='420' y2='62' stroke='#C93B3B' stroke-width='2'/><text x='420' y='40' fill='#C93B3B'>90% = $45</text><text x='90' y='75' fill='#6B645E'>$0</text><text x='450' y='75' fill='#6B645E'>$50 cap</text></g></svg>",
     "note":"<b>Set billing alerts.</b> Warnings at 50% ($25) and 90% ($45) of the cap. The 90% alert fires near the end of a healthy run — early enough to stop a runaway one."},
    {"viz":"<svg viewBox='0 0 520 80'><g font-family='monospace' font-size='11' text-anchor='middle'><line x1='40' y1='45' x2='480' y2='45' stroke='#6B645E' stroke-width='1.5'/><circle cx='120' cy='45' r='5' fill='#2D8B55'/><circle cx='220' cy='45' r='5' fill='#2D8B55'/><circle cx='320' cy='45' r='5' fill='#2D8B55'/><circle cx='420' cy='45' r='5' fill='#2D8B55'/><text x='260' y='30' fill='#1a5c38'>checkpoint every 30 min</text><text x='260' y='68' fill='#6B645E'>crash → resume from last ● (lose ≤ 30 min)</text></g></svg>",
     "note":"<b>Checkpoint on a schedule.</b> Save weights every 30 minutes. If the run dies, you resume from the last dot — losing at most one interval, never the whole run."},
    {"viz":"<svg viewBox='0 0 520 80'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='30' y='28' width='210' height='30' rx='6' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2'/><text x='135' y='47' fill='#1a5c38'>under cap + checkpointed → GO</text><rect x='280' y='28' width='210' height='30' rx='6' fill='#FDE8E8' stroke='#C93B3B' stroke-width='2'/><text x='385' y='47' fill='#C93B3B'>over cap or no save → STOP</text></g></svg>",
     "note":"<b>The gate decision.</b> If the estimate clears the cap, quota is approved, alerts are set, and checkpoints are on — GO. Miss any one, and you fix it before spending a cent at scale."},
  ],
  quiz=[
    {"q":"1. A run does 10,000 steps at 0.6 s/step on 10 devices at $3/device-hour. Roughly what does it cost?","opts":["about $5","about $50","about $500","about $1.7"],"ans":1,"fb":"hours = 10000×0.6/3600 ≈ 1.667; cost = 3 × 10 × 1.667 ≈ $50. Cost is just price × devices × hours."},
    {"q":"2. Why request quota before you are ready to launch?","opts":["quota makes the model better","approval can take hours or days, so asking at launch time blocks you","quota lowers the price","quota removes the need for checkpoints"],"ans":1,"fb":"A quota raise is not instant. Request it during dry-runs so the devices are available the moment your estimate clears the gate."},
    {"q":"3. A $40 run has NO checkpointing and the machine is preempted at 95% done. How much of the spend is wasted?","opts":["about $2","about $20","about $38 — nearly all of it, and you must redo the work","nothing, it auto-resumes"],"ans":2,"fb":"With no checkpoint there is no saved state, so you restart from step 0. Essentially the whole ~$40 (and the time) is lost and must be paid again. Checkpointing would cap the loss to one interval."},
    {"q":"4. When does cheap spot/preemptible hardware make sense over on-demand?","opts":["never — it is always a trap","only for tiny jobs","when you checkpoint well, so a preemption just resumes from the last save instead of losing everything","only when there is no budget cap"],"ans":2,"fb":"Spot is much cheaper but can be reclaimed anytime. With good checkpointing a preemption only costs one interval, so the discount is nearly free. Without checkpointing, one preemption wastes the whole run."},
  ],
  fin={"em":"💰","h3":"Day 2 complete — you can afford the big button!",
       "p":"You can now estimate a run's cost from step time (<code>cost = price × devices × hours</code>), request quota ahead of time, set billing alerts at 50% and 90% of a cap, and checkpoint so a killed run wastes one interval instead of everything — which is exactly what makes cheap spot hardware safe. Next: the <b>Review Gate</b> to lock it in before real orchestration in M9e."},
))
# ---------------- Review ----------------
write_review(os.path.join(OUT, "review.html"), dict(
  qid="m9d-review", title="Module 9d · Review Gate", nav_title="Spiral · M9d Review",
  eyebrow="Module 9d · The Gate — Cost & Sanity Before Scale", h1="Module 9d — Review Gate",
  lead="A short but real gate. The question: <b>before I launch a large multi-device run, can I prove it works, prove I can afford it, and prove a crash won't cost me everything?</b> Tick the self-check, then answer five mixed questions.",
  goal="<b>🎯 The rule:</b> if a box feels shaky, re-run that day. M9e (orchestration) launches real jobs at scale — it assumes you already sanity-check step time and clear a cost gate every time.",
  prev_href="day-02-cost-gate.html", prev_label="Day 2 · The Cost & Quota Gate",
  checks=[
    ["Day 1","I can run a small dry-run, measure real step time, and compute <code>ratio = measured / predicted</code> — and I know a big ratio (like 10×) usually means a silently slow compile, so I stop and debug before scaling."],
    ["Day 2","I can estimate a run's cost as <code>price × devices × hours</code>, request quota ahead of time, set billing alerts at 50% and 90% of a cap, and checkpoint so a killed run loses one interval — which is what makes cheap spot hardware safe."],
  ],
  quiz=[
    {"q":"1. After a dry-run, the first number to compare is:","opts":["the final loss","the measured step time vs the predicted step time","the number of files written","the GPU temperature"],"ans":1,"fb":"Step time is the headline. measured ÷ predicted tells you whether reality matches the M9c plan before you scale (Day 1)."},
    {"q":"2. Predicted step time is 0.4 s and the dry-run measures 4.0 s. This is most likely:","opts":["normal overhead, scale up","a silent bad compile (10× slowdown) — stop and debug","a checkpointing bug","a billing-alert misfire"],"ans":1,"fb":"ratio = 4.0/0.4 = 10. A 10× slowdown with no crash is the classic silent slow-compile. Never scale it (Day 1)."},
    {"q":"3. A run does 20,000 steps at 0.5 s/step on 8 devices at $2/device-hour. The cost is about:","opts":["$4","$44","$440","$8"],"ans":1,"fb":"hours = 20000×0.5/3600 ≈ 2.78; cost = 2 × 8 × 2.78 ≈ $44 — under a $50 cap (Day 2)."},
    {"q":"4. Why checkpoint during a long run?","opts":["it speeds up each step","if the run dies you resume from the last save, losing one interval instead of the whole spend","it lowers the device price","it removes the need for a dry-run"],"ans":1,"fb":"A checkpoint is a saved game. A crash costs you at most one interval of compute, not the entire run — and it is what makes cheap spot hardware safe (Day 2)."},
    {"q":"5. Cheap spot/preemptible hardware is a good choice when:","opts":["you never checkpoint","you checkpoint well, so a preemption just resumes from the last save","the run is under one second","there is no quota"],"ans":1,"fb":"Spot is much cheaper but can be reclaimed anytime. With good checkpointing a preemption only costs one interval, so the discount is nearly free; without it, one preemption wastes everything (Day 2)."},
  ],
  verdict_pass="you can take a plan to real hardware safely: sanity-check step time against the calculator, estimate and cap the cost, request quota early, alert on spend, and checkpoint so a crash is cheap. You're ready for M9e — real orchestration at scale.",
  verdict_fail="note which day felt shaky and re-run just that lesson. The chain — <code>dry-run</code> → <code>measured vs predicted</code> → <code>cost estimate</code> → <code>quota + alerts + checkpoints</code> → <code>GO</code> — should feel automatic before you launch anything large.",
  complete_label="Mark Module 9d complete",
  fin={"em":"🚦","h3":"Module 9d — passed!",
       "p":"You closed the gap between a sharding calculator and a real run: one small dry-run to sanity-check step time, then a cost gate — estimate, quota, billing alerts, and checkpoints — that keeps a wrong plan or a dead machine from burning your budget. This is the discipline every large run at a frontier lab lives by. Next: <b>M9e</b> — orchestrating these runs at real scale."},
  score_target=4,
))
print("M9d built: 2 lessons + review")


