---
quest_id: test-mini
mode: concept
donor: v9-base.donor
page_title: "Mini Concept Test"
module_label: "Test · Mini"
title: "Mini"
subtitle: "a tiny concept lesson"
brand_sub: "Test · Mini"
spine: "bend"
nav_prev_href: "#"
nav_prev_label: "Prev"
nav_next_href: "#"
nav_next_label: "Next"
fin_title: "Mini complete!"
fin_body: "Done."
notebook_yardstick: null
---

@@@ hero
@lede Ever wonder why a bend matters? Picture two straight rulers.
@goal By the end you can explain the bend.

@@@ concept id=c1 tag="The collapse" title="Straight + straight is still straight" gotit="Got it"
Two straight rulers stacked are still one straight ruler — a bend is missing.
%%% svg
<svg viewBox="0 0 10 10" role="img" aria-label="collapse"><path d="M0 5 L10 5"/></svg>
%%%
So depth without a bend buys nothing.

@@@ concept id=c2 tag="The bend" title="A bend between the layers" gotit="Got the bend"
Put a bend between layers and they can no longer fold flat.
%%% svg
<svg viewBox="0 0 10 10" role="img" aria-label="bend"><path d="M0 10 L5 0"/></svg>
%%%
That bend is the activation.

@@@ concept id=c3 tag="Meet ReLU" title="ReLU — a one-way valve" gotit="Met ReLU"
ReLU passes positives and zeroes negatives.
%%% svg
<svg viewBox="0 0 10 10" role="img" aria-label="relu"><path d="M0 10 L5 10 L10 0"/></svg>
%%%
%%% demo id=relu label="run it"
code: relu(np.array([-3,-1,0,2,5]))
out: array([0, 0, 0, 2, 5])
take: ReLU zeros negatives, passes positives.
%%%
One cheap max — the modern default.

@@@ concept id=c4 tag="Recap" title="Today in one page" gotit="Got the recap"
It all came down to one idea — the bend. Straight+straight stays straight; a bend makes depth matter; ReLU is the cheap default bend. Come back to this any time.
%%% svg
<svg viewBox="0 0 10 10" role="img" aria-label="recap"><path d="M0 8 L4 8 L8 2"/></svg>
%%%
%%% jargon
bend | the small non-linearity (activation) added at the end of a neuron
ReLU | a one-way valve: passes positives, zeroes negatives
%%%

@@@ quiz id=quiz tag="Check" title="Four questions" gotit="Checked"
%%% quiz
q: What does an activation add? | a:1 | params | non-linearity | speed | bias | fb: The bend.
q: Ten linear layers = ? | a:1 | more power | one linear layer | random | sigmoid | fb: One matrix.
q: ReLU(z) = ? | a:1 | 1/(1+e^-z) | max(0,z) | z^2 | -z | fb: keep positives.
q: All ReLUs output 0? | a:1 | sigmoid bug | dead ReLUs | OOM | converged | fb: dead units.
%%%

@@@ produce id=produce tag="Produce" title="Watch the collapse" gotit="Done"
Predict what `(x@W1)@W2` prints vs `x@(W1@W2)`, then run it and observe they match — until you insert a ReLU. Write it in `experiment.py` and run it.

@@@ fin
