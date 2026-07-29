# day-07-object-detection-intuition — experiment
#
# Today's big idea in two lines of output:
#   IoU turns "do these two boxes agree?" into one number — the area they SHARE
#   divided by the area they cover TOGETHER (0 = no overlap, 1 = a perfect fit).
#   NMS uses that number to collapse four boxes piled on one cat into ONE while
#   the far-away dog box survives on its own: five boxes in, two boxes out.
#
# Nothing here is random, so nothing needs a seed: every box is a fixed list of
# whole numbers and all the arithmetic is exact.
# Run it:  python3 sessions/m06-cnns-vision-encoders/day-07-object-detection-intuition/experiment.py

import numpy as np  # numpy paints the pixel masks used as an independent second opinion

IMG_H, IMG_W = 320, 400   # a tall-by-wide grid on purpose: 320 != 400, so a mixed axis shows

def box_area(b):
    # A box is [x1, y1, x2, y2]: top-left corner, then bottom-right corner.
    return (b[2] - b[0]) * (b[3] - b[1])

def overlap_rect(a, b):
    # The shared rectangle starts at the RIGHTMOST left edge and the LOWEST top edge,
    # and ends at the LEFTMOST right edge and the HIGHEST bottom edge.
    return (max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]))

def iou_parts(a, b):
    # Every intermediate value of Intersection over Union, so the story can be printed.
    x1, y1, x2, y2 = overlap_rect(a, b)
    # Boxes that miss each other give a NEGATIVE width. That is not a rectangle, so
    # clamp it to 0 — the clamp is what makes "no overlap" score exactly 0.
    w, h = max(0, x2 - x1), max(0, y2 - y1)
    inter = w * h
    # Adding both areas counts the shared middle twice, so subtract one copy of it.
    union = box_area(a) + box_area(b) - inter
    return {"rect": (x1, y1, x2, y2), "w": w, "h": h, "inter": inter, "union": union,
            "area_a": box_area(a), "area_b": box_area(b), "iou": inter / union}

def iou(a, b):
    return iou_parts(a, b)["iou"]     # the one number the lesson leans on

def pixel_iou(a, b):
    # The DEFINITION of IoU, counted one pixel at a time — no formula, no shortcut.
    def mask(box):
        m = np.zeros((IMG_H, IMG_W), dtype=bool)   # rows are y, columns are x
        m[box[1]:box[3], box[0]:box[2]] = True     # paint True on every pixel inside
        return m
    ma, mb = mask(a), mask(b)
    shared = int(np.logical_and(ma, mb).sum())     # pixels inside BOTH boxes
    together = int(np.logical_or(ma, mb).sum())    # pixels inside EITHER box
    return ma.shape, shared, together, shared / together

def nms(scored, iou_cut=0.5, verbose=False):
    # Non-Maximum Suppression: keep the loudest box, hush the boxes overlapping it, repeat.
    remaining = sorted(scored, key=lambda pair: pair[1], reverse=True)  # highest score first
    kept = []
    while remaining:
        best_box, best_score = remaining.pop(0)    # the most confident box still standing
        kept.append((best_box, best_score))
        if verbose:
            print("  keep score %.2f (highest left) -> test every other box against it"
                  % best_score)
        survivors = []
        for box, score in remaining:
            overlap = iou(best_box, box)
            # The rule is strictly ABOVE the cut-off: a box sitting exactly at it stays.
            drop = overlap > iou_cut
            if verbose:
                print("    score %.2f  iou %.4f  %s" % (score, overlap,
                      "> %.2f  -> SUPPRESS (same object)" % iou_cut if drop
                      else "<= %.2f -> survives (different object)" % iou_cut))
            if not drop:
                survivors.append((box, score))
        remaining = survivors
    return kept

def show_box(name, b):
    # One line per box: the four numbers, the size they imply, and the area.
    print("  %-11s [x1=%3d, y1=%3d, x2=%3d, y2=%3d]  ->  %3d x %3d px, area %5d"
          % (name, b[0], b[1], b[2], b[3], b[2] - b[0], b[3] - b[1], box_area(b)))

if __name__ == "__main__":
    # --- Part 1: a box is just four numbers -------------------------------
    cat_true = [50, 50, 150, 140]     # the TRUE box a human drew around the cat
    cat_pred = [60, 55, 158, 150]     # the model's GUESS at the same cat, shifted a bit
    dog = [300, 200, 380, 300]        # a second object, far away, touching neither
    print("Part 1 — three boxes written as [x1, y1, x2, y2] (two opposite corners):")
    for name, b in [("cat_true", cat_true), ("cat_pred", cat_pred), ("dog", dog)]:
        show_box(name, b)

    # --- Part 2: IoU, worked out twice by two different routes -------------
    near, far = iou_parts(cat_true, cat_pred), iou_parts(cat_true, dog)
    print("\nPart 2 — IoU = (area they SHARE) / (area they cover TOGETHER)")
    print("  cat_true vs cat_pred: overlap rect", near["rect"], "-> %d x %d = %d shared px"
          % (near["w"], near["h"], near["inter"]))
    print("  union = %d + %d - %d = %d px" % (near["area_a"], near["area_b"],
          near["inter"], near["union"]), "(subtract the shared middle, or it counts twice)")
    print("  iou(cat_true, cat_pred) =", round(near["iou"], 4))
    print("  cat_true vs dog:      overlap rect", far["rect"],
          "-> width %d and height %d are NEGATIVE, both clamped to 0"
          % (far["rect"][2] - far["rect"][0], far["rect"][3] - far["rect"][1]))
    print("  iou(cat_true, dog)      =", round(far["iou"], 4))
    # Second opinion: paint both boxes on a pixel grid and COUNT. This route shares no
    # code with the formula above, so agreement between them is real evidence.
    grid_shape, shared_px, together_px, px_iou = pixel_iou(cat_true, cat_pred)
    _, shared_far, together_far, px_iou_far = pixel_iou(cat_true, dog)
    print("  pixel grid shape:", grid_shape, "(rows = y, cols = x) — counted pixel by pixel,"
          " shared =", shared_px, " together =", together_px, " ratio =", round(px_iou, 4))
    print("  formula and pixel count land on the same float:", px_iou == near["iou"])
    # The lesson asks you to PREDICT: closer to 0, or closer to 1? Work the prediction
    # out from the pixel counts (2*shared > together means past halfway), then hold it
    # against what the formula returned. Neither answer is typed in by hand.
    side = lambda shared, together: "1" if 2 * shared > together else "0"
    near_pred, far_pred = side(shared_px, together_px), side(shared_far, together_far)
    near_real = "1" if near["iou"] > 0.5 else "0"     # what the formula actually returned
    far_real = "1" if far["iou"] > 0.5 else "0"
    print("  predicted closer to / actually closer to:  cat pair -> %s / %s" %
          (near_pred, near_real), " cat vs dog -> %s / %s" % (far_pred, far_real))

    # --- Part 3: a pile of scored boxes, the way a detector spits them out --
    boxes = [([55, 52, 152, 142], 0.95), ([58, 50, 150, 140], 0.88),
             ([52, 55, 156, 145], 0.81), ([60, 48, 148, 138], 0.77),
             ([300, 200, 380, 300], 0.90)]
    print("\nPart 3 — raw detector output: %d boxes, four on one cat plus one dog" % len(boxes))
    for b, s in boxes:
        show_box("score %.2f" % s, b)

    # --- Part 4: NMS turns the pile into one box per object ----------------
    print("\nPart 4 — NMS with IoU cut-off 0.50")
    kept = nms(boxes, iou_cut=0.5, verbose=True)
    print("  final count =", len(kept), "— four cat boxes became one, the dog survived:")
    for b, s in kept:
        show_box("score %.2f" % s, b)
    # A boundary case, because the rule says ABOVE the cut-off, not "at or above":
    tie_big, tie_small = [0, 0, 60, 100], [0, 0, 60, 50]   # 6000 px and 3000 px, nested
    tie_iou = iou(tie_big, tie_small)
    tie_kept = nms([(tie_big, 0.9), (tie_small, 0.4)], iou_cut=0.5)
    print("  boundary: two nested boxes with iou exactly", tie_iou, "-> NMS keeps",
          len(tie_kept), "boxes, because 0.5 is not ABOVE 0.5")

    # --- Part 5: the threshold is a separate dial from NMS -----------------
    strong = [(b, s) for b, s in boxes if s > 0.85]
    print("\nPart 5 — filter by score > 0.85 BEFORE NMS")
    print("  scores the threshold keeps:", [s for b, s in strong],
          " scores it drops:", [s for b, s in boxes if s <= 0.85])
    # The spec's five boxes never score exactly 0.85, so "> 0.85" and ">= 0.85" behave the
    # same on them — the strictness would go untested. One extra box sitting ON the line
    # settles it: a STRICT ">" must drop it. (Same idea as the 0.50 IoU tie in Part 4.)
    on_the_line = boxes + [([10, 10, 20, 20], 0.85)]
    edge = [s for b, s in on_the_line if s > 0.85]
    print("  a box scoring exactly 0.85:", "kept" if 0.85 in edge else "dropped",
          "-> the dial is strict, so equal-to-the-threshold does not survive")
    kept_strong = nms(strong, iou_cut=0.5)
    print("  NMS on that shorter pile keeps:", [s for b, s in kept_strong],
          "-> the dial changed the input, not the one-box-per-object result")
    print("\ntakeaway: detection outputs many boxes with confidence scores; IoU measures"
          "\n  how well two boxes overlap; NMS keeps one tidy box per object; the number"
          "\n  of final boxes is variable (0, 1, or many).")

    # --- Self-check: one boolean per claim ---------------------------------
    # Every number below was read off a real run and written down here by hand, so a
    # broken change above cannot quietly re-derive its own expected value.
    areas_ok = (near["area_a"], near["area_b"], box_area(dog)) == (9000, 9310, 8000)
    rects_ok = (near["rect"] == (60, 55, 150, 140)          # rightmost-left, lowest-top, ...
                and far["rect"] == (300, 200, 150, 140))    # x2 < x1: these boxes miss
    union_ok = near["union"] == 10660                       # 9000 + 9310 - 7650, not 18310
    iou_high_ok = round(near["iou"], 4) == 0.7176           # shared / together, not the flip
    iou_zero_ok = far["iou"] == 0.0            # fails the moment the clamp is dropped
    pixels_ok = ((shared_px, together_px) == (7650, 10660)     # the definition, by counting
                 and (shared_far, together_far) == (0, 17000)
                 and px_iou == near["iou"] and px_iou_far == far["iou"])
    # Neither branch of the prediction is dead: one pair answers "1", the other "0".
    prediction_ok = (near_pred, far_pred) == (near_real, far_real) and near_pred != far_pred
    # Kept boxes by coordinate AND score: a count alone would not notice that sorting the
    # scores the wrong way round also leaves exactly two boxes standing.
    nms_kept_ok = kept == [([55, 52, 152, 142], 0.95), ([300, 200, 380, 300], 0.90)]
    winner = kept[0][0]
    duplicates_ok = (tuple(round(iou(winner, b), 4) for b, s in boxes[1:4])
                     == (0.9082, 0.8744, 0.8333))           # all three far above 0.50
    dog_apart_ok = iou(winner, dog) == 0.0                  # why the dog is never hushed
    tie_ok = tie_iou == 0.5 and len(tie_kept) == 2           # the rule is ">", not ">="
    threshold_ok = ([s for b, s in strong] == [0.95, 0.88, 0.90]
                    and [s for b, s in kept_strong] == [0.95, 0.90]
                    and edge == [0.95, 0.88, 0.90])   # the 0.85 box was DROPPED, not kept

    if (areas_ok and rects_ok and union_ok and iou_high_ok and iou_zero_ok and pixels_ok
            and prediction_ok and nms_kept_ok and duplicates_ok and dog_apart_ok
            and tie_ok and threshold_ok):
        print("\n✅ you got it")
    else:
        print("\n❌ not yet — expected areas 9000/9310/8000, overlap rects (60,55,150,140) and "
              "(300,200,150,140), union 10660, iou(cat_true,cat_pred) = 0.7176 matching a "
              "7650/10660 pixel count, iou(cat_true,dog) = 0.0 exactly, NMS to keep "
              "[55,52,152,142]@0.95 and [300,200,380,300]@0.90 after suppressing IoUs "
              "0.9082/0.8744/0.8333, a 0.5 tie to survive, score > 0.85 to leave "
              "[0.95, 0.88, 0.90] and NMS on that to leave [0.95, 0.90]")

    # These asserts make the check hard: a wrong run stops the program right here.
    assert areas_ok, "box areas should be 9000 (cat_true), 9310 (cat_pred), 8000 (dog)"
    assert rects_ok, "overlap rects should be (60,55,150,140) and (300,200,150,140)"
    assert union_ok, "union should be 9000 + 9310 - 7650 = 10660, so subtract the overlap"
    assert iou_high_ok, "iou(cat_true, cat_pred) should be 0.7176 — shared over together"
    assert iou_zero_ok, "iou(cat_true, dog) must be exactly 0.0 — clamp negative sides to 0"
    assert pixels_ok, "counting pixels must give 7650/10660 and 0/17000, same floats"
    assert prediction_ok, "the computed prediction must match reality, and differ per pair"
    assert nms_kept_ok, "NMS should keep [55,52,152,142]@0.95 then [300,200,380,300]@0.90"
    assert duplicates_ok, "the suppressed boxes should score IoU 0.9082, 0.8744, 0.8333"
    assert dog_apart_ok, "the dog box overlaps the winner by 0.0, so it must not be hushed"
    assert tie_ok, "a box at IoU exactly 0.5 must survive — the rule is ABOVE the cut-off"
    assert threshold_ok, ("score > 0.85 should leave [0.95, 0.88, 0.90], then NMS "
                          "[0.95, 0.90], and must DROP a box scoring exactly 0.85")
