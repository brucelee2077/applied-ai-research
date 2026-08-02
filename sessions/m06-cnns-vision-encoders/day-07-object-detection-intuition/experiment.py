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

def nms(scored, iou_cut=0.5, verbose=False, log=None):
    # Non-Maximum Suppression: keep the loudest box, hush the boxes overlapping it, repeat.
    # `log` collects the exact strings the verbose lines show, so the self-check can
    # assert what the learner READ instead of computing the same numbers a second time.
    remaining = sorted(scored, key=lambda pair: pair[1], reverse=True)  # highest score first
    kept = []
    while remaining:
        best_box, best_score = remaining.pop(0)    # the most confident box still standing
        kept.append((best_box, best_score))
        shown_best = "%.2f" % best_score           # bound once: printed and checked
        if log is not None:
            log["keeps"].append(shown_best)
        if verbose:
            print("  keep score %s (highest left) -> test every other box against it"
                  % shown_best)
        survivors = []
        for box, score in remaining:
            overlap = iou(best_box, box)
            # The rule is strictly ABOVE the cut-off: a box sitting exactly at it stays.
            drop = overlap > iou_cut
            shown_score, shown_overlap = "%.2f" % score, "%.4f" % overlap
            if log is not None:
                log["tests"].append((shown_score, shown_overlap, drop))
            if verbose:
                print("    score %s  iou %s  %s" % (shown_score, shown_overlap,
                      "> %.2f  -> SUPPRESS (same object)" % iou_cut if drop
                      else "<= %.2f -> survives (different object)" % iou_cut))
            if not drop:
                survivors.append((box, score))
        remaining = survivors
    return kept

def keeps_score(score, cut):
    # The ONE score-threshold dial in this file: strictly ABOVE the cut-off, so a box
    # sitting exactly on the line is dropped. Part 5's real filter AND its on-the-line
    # probe both call this, which is what makes the strictness testable: flipping ">"
    # to ">=" here changes the probe's answer instead of passing unnoticed.
    return score > cut

def show_box(name, b):
    # One line per box: the four numbers, the size they imply, and the area.
    # Read the "97 x 90 px" part as WIDTH x HEIGHT — (x2 − x1) then (y2 − y1). That is the
    # opposite order from every grid in days 1 to 5, where a shape is (rows, cols) = height
    # first, and from the pixel masks in this very file (mask.shape is (IMG_H, IMG_W)).
    # Boxes go x-first because their coordinates are written [x1, y1, x2, y2]; grids go
    # rows-first because that is how arrays are indexed. Days 1 to 5 could never show the
    # difference — every example there is square (6x6, 28x28, 32x32, 4x4).
    # Build the line once, print that string, and hand it back so the self-check can
    # assert the very characters that reached the screen.
    line = ("  %-11s [x1=%3d, y1=%3d, x2=%3d, y2=%3d]  ->  %3d x %3d px, area %5d"
            % (name, b[0], b[1], b[2], b[3], b[2] - b[0], b[3] - b[1], box_area(b)))
    print(line)
    return line

if __name__ == "__main__":
    # --- Part 1: a box is just four numbers -------------------------------
    cat_true = [50, 50, 150, 140]     # the TRUE box a human drew around the cat
    cat_pred = [60, 55, 158, 150]     # the model's GUESS at the same cat, shifted a bit
    dog = [300, 200, 380, 300]        # a second object, far away, touching neither
    print("Part 1 — three boxes written as [x1, y1, x2, y2] (two opposite corners):")
    box_lines = [show_box(name, b) for name, b in
                 [("cat_true", cat_true), ("cat_pred", cat_pred), ("dog", dog)]]

    # --- Part 2: IoU, worked out twice by two different routes -------------
    near, far = iou_parts(cat_true, cat_pred), iou_parts(cat_true, dog)
    # Every number in the lines below is bound to a name FIRST, then printed, then
    # checked through that same name — one value, read twice, never recomputed.
    shown_near_rect, shown_far_rect = near["rect"], far["rect"]
    shown_w, shown_h, shown_inter = near["w"], near["h"], near["inter"]
    shown_area_a, shown_area_b, shown_union = near["area_a"], near["area_b"], near["union"]
    shown_near_iou, shown_far_iou = round(near["iou"], 4), round(far["iou"], 4)
    shown_far_w_raw = shown_far_rect[2] - shown_far_rect[0]   # negative before the clamp
    shown_far_h_raw = shown_far_rect[3] - shown_far_rect[1]
    print("\nPart 2 — IoU = (area they SHARE) / (area they cover TOGETHER)")
    print("  cat_true vs cat_pred: overlap rect", shown_near_rect, "-> %d x %d = %d shared px"
          % (shown_w, shown_h, shown_inter))
    print("  union = %d + %d - %d = %d px" % (shown_area_a, shown_area_b,
          shown_inter, shown_union), "(subtract the shared middle, or it counts twice)")
    print("  iou(cat_true, cat_pred) =", shown_near_iou)
    print("  cat_true vs dog:      overlap rect", shown_far_rect,
          "-> width %d and height %d are NEGATIVE, both clamped to 0"
          % (shown_far_w_raw, shown_far_h_raw))
    print("  iou(cat_true, dog)      =", shown_far_iou)
    # Second opinion: paint both boxes on a pixel grid and COUNT. This route shares no
    # code with the formula above, so agreement between them is real evidence.
    grid_shape, shared_px, together_px, px_iou = pixel_iou(cat_true, cat_pred)
    _, shared_far, together_far, px_iou_far = pixel_iou(cat_true, dog)
    shown_px_iou = round(px_iou, 4)
    # The one line whose whole job is the axis order, so its shape is a checked claim.
    shown_grid_shape = grid_shape
    shown_same_float = px_iou == near["iou"]
    print("  pixel grid shape:", shown_grid_shape,
          "(rows = y, cols = x) — counted pixel by pixel,"
          " shared =", shared_px, " together =", together_px, " ratio =", shown_px_iou)
    print("  formula and pixel count land on the same float:", shown_same_float)
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
    n_boxes = len(boxes)
    print("\nPart 3 — raw detector output: %d boxes, four on one cat plus one dog" % n_boxes)
    raw_lines = [show_box("score %.2f" % s, b) for b, s in boxes]

    # --- Part 4: NMS turns the pile into one box per object ----------------
    IOU_CUT = 0.5
    print("\nPart 4 — NMS with IoU cut-off %.2f" % IOU_CUT)
    nms_log = {"keeps": [], "tests": []}      # the strings the verbose run printed
    kept = nms(boxes, iou_cut=IOU_CUT, verbose=True, log=nms_log)
    kept_count = len(kept)
    print("  final count =", kept_count, "— four cat boxes became one, the dog survived:")
    kept_lines = [show_box("score %.2f" % s, b) for b, s in kept]
    # A boundary case, because the rule says ABOVE the cut-off, not "at or above":
    tie_big, tie_small = [0, 0, 60, 100], [0, 0, 60, 50]   # 6000 px and 3000 px, nested
    tie_iou = iou(tie_big, tie_small)
    tie_kept = nms([(tie_big, 0.9), (tie_small, 0.4)], iou_cut=IOU_CUT)
    tie_kept_count = len(tie_kept)
    print("  boundary: two nested boxes with iou exactly", tie_iou, "-> NMS keeps",
          tie_kept_count, "boxes, because %.1f is not ABOVE %.1f" % (IOU_CUT, IOU_CUT))

    # --- Part 5: the threshold is a separate dial from NMS -----------------
    SCORE_CUT = 0.85
    strong = [(b, s) for b, s in boxes if keeps_score(s, SCORE_CUT)]
    weak = [(b, s) for b, s in boxes if not keeps_score(s, SCORE_CUT)]   # the complement
    shown_strong_scores = [s for b, s in strong]
    shown_weak_scores = [s for b, s in weak]
    print("\nPart 5 — filter by score > %.2f BEFORE NMS" % SCORE_CUT)
    print("  scores the threshold keeps:", shown_strong_scores,
          " scores it drops:", shown_weak_scores)
    # The spec's five boxes never score exactly 0.85, so "> 0.85" and ">= 0.85" behave the
    # same on them — the strictness would go untested. One extra box sitting ON the line
    # settles it: a STRICT ">" must drop it. (Same idea as the 0.50 IoU tie in Part 4.)
    # The probe runs through `keeps_score`, the SAME dial `strong` used above, so it tests
    # the real filter rather than a look-alike written out a second time.
    on_the_line = boxes + [([10, 10, 20, 20], SCORE_CUT)]
    edge = [s for b, s in on_the_line if keeps_score(s, SCORE_CUT)]
    shown_edge_word = "kept" if SCORE_CUT in edge else "dropped"
    print("  a box scoring exactly %.2f:" % SCORE_CUT, shown_edge_word,
          "-> the dial is strict, so equal-to-the-threshold does not survive")
    kept_strong = nms(strong, iou_cut=IOU_CUT)
    shown_kept_strong_scores = [s for b, s in kept_strong]
    print("  NMS on that shorter pile keeps:", shown_kept_strong_scores,
          "-> the dial changed the input, not the one-box-per-object result")
    print("\ntakeaway: detection outputs many boxes with confidence scores; IoU measures"
          "\n  how well two boxes overlap; NMS keeps one tidy box per object; the number"
          "\n  of final boxes is variable (0, 1, or many).")

    # --- Self-check: one boolean per claim ---------------------------------
    # Every number below was read off a real run and written down here by hand, so a
    # broken change above cannot quietly re-derive its own expected value. Each claim
    # reads the SAME name the print statement read, so corrupting a printed value is
    # now a failing run instead of a wrong number under a green tick.
    areas_ok = (shown_area_a, shown_area_b, box_area(dog)) == (9000, 9310, 8000)
    box_lines_ok = box_lines == [
        "  cat_true    [x1= 50, y1= 50, x2=150, y2=140]  ->  100 x  90 px, area  9000",
        "  cat_pred    [x1= 60, y1= 55, x2=158, y2=150]  ->   98 x  95 px, area  9310",
        "  dog         [x1=300, y1=200, x2=380, y2=300]  ->   80 x 100 px, area  8000"]
    rects_ok = (shown_near_rect == (60, 55, 150, 140)       # rightmost-left, lowest-top, ...
                and shown_far_rect == (300, 200, 150, 140))  # x2 < x1: these boxes miss
    negative_sides_ok = (shown_far_w_raw, shown_far_h_raw) == (-150, -60)   # before the clamp
    overlap_shape_ok = (shown_w, shown_h, shown_inter) == (90, 85, 7650)    # 90*85 = 7650
    union_ok = shown_union == 10660                         # 9000 + 9310 - 7650, not 18310
    iou_high_ok = shown_near_iou == 0.7176                  # shared / together, not the flip
    iou_zero_ok = far["iou"] == 0.0 and shown_far_iou == 0.0  # fails if the clamp is dropped
    grid_ok = shown_grid_shape == (320, 400)   # rows = y = 320, cols = x = 400, in that order
    pixels_ok = ((shared_px, together_px) == (7650, 10660)     # the definition, by counting
                 and (shared_far, together_far) == (0, 17000)
                 and shown_same_float and shown_px_iou == 0.7176
                 and px_iou_far == far["iou"])
    # Neither branch of the prediction is dead: one pair answers "1", the other "0".
    prediction_ok = (near_pred, far_pred) == (near_real, far_real) and near_pred != far_pred
    raw_ok = n_boxes == 5 and raw_lines == [
        "  score 0.95  [x1= 55, y1= 52, x2=152, y2=142]  ->   97 x  90 px, area  8730",
        "  score 0.88  [x1= 58, y1= 50, x2=150, y2=140]  ->   92 x  90 px, area  8280",
        "  score 0.81  [x1= 52, y1= 55, x2=156, y2=145]  ->  104 x  90 px, area  9360",
        "  score 0.77  [x1= 60, y1= 48, x2=148, y2=138]  ->   88 x  90 px, area  7920",
        "  score 0.90  [x1=300, y1=200, x2=380, y2=300]  ->   80 x 100 px, area  8000"]
    # Kept boxes by coordinate AND score: a count alone would not notice that sorting the
    # scores the wrong way round also leaves exactly two boxes standing.
    kept_lines_expected = [
        "  score 0.95  [x1= 55, y1= 52, x2=152, y2=142]  ->   97 x  90 px, area  8730",
        "  score 0.90  [x1=300, y1=200, x2=380, y2=300]  ->   80 x 100 px, area  8000"]
    nms_kept_ok = (kept == [([55, 52, 152, 142], 0.95), ([300, 200, 380, 300], 0.90)]
                   and kept_count == 2 and nms_log["keeps"] == ["0.95", "0.90"]
                   and kept_lines == kept_lines_expected)
    winner = kept[0][0]
    # Read back from the lines the run printed, not computed a second time here.
    duplicates_ok = ([(s, o) for s, o, drop in nms_log["tests"] if drop]
                     == [("0.88", "0.9082"), ("0.81", "0.8744"),
                         ("0.77", "0.8333")])              # all three far above 0.50
    dog_apart_ok = (iou(winner, dog) == 0.0                 # why the dog is never hushed
                    and [(s, o) for s, o, drop in nms_log["tests"] if not drop]
                    == [("0.90", "0.0000")])
    tie_ok = tie_iou == 0.5 and tie_kept_count == 2           # the rule is ">", not ">="
    threshold_ok = (shown_strong_scores == [0.95, 0.88, 0.90]
                    and shown_weak_scores == [0.81, 0.77]
                    and shown_kept_strong_scores == [0.95, 0.90]
                    and shown_edge_word == "dropped"    # the 0.85 box was DROPPED, not kept
                    and edge == [0.95, 0.88, 0.90])

    if (areas_ok and box_lines_ok and rects_ok and negative_sides_ok and overlap_shape_ok
            and union_ok and iou_high_ok and iou_zero_ok and grid_ok and pixels_ok
            and prediction_ok and raw_ok and nms_kept_ok and duplicates_ok and dog_apart_ok
            and tie_ok and threshold_ok):
        print("\n✅ you got it")
    else:
        print("\n❌ not yet — expected areas 9000/9310/8000, overlap rects (60,55,150,140) and "
              "(300,200,150,140), union 10660, iou(cat_true,cat_pred) = 0.7176 matching a "
              "7650/10660 pixel count on a (320, 400) grid, iou(cat_true,dog) = 0.0 exactly, "
              "NMS to keep "
              "[55,52,152,142]@0.95 and [300,200,380,300]@0.90 after suppressing IoUs "
              "0.9082/0.8744/0.8333, a 0.5 tie to survive, score > 0.85 to leave "
              "[0.95, 0.88, 0.90] and NMS on that to leave [0.95, 0.90]")

    # These asserts make the check hard: a wrong run stops the program right here.
    assert areas_ok, "box areas should be 9000 (cat_true), 9310 (cat_pred), 8000 (dog)"
    assert box_lines_ok, "the three printed box lines must read exactly as recorded"
    assert rects_ok, "overlap rects should be (60,55,150,140) and (300,200,150,140)"
    assert negative_sides_ok, "the missing pair's raw sides should print as -150 and -60"
    assert overlap_shape_ok, "the shared rectangle should be 90 x 85 = 7650 px"
    assert union_ok, "union should be 9000 + 9310 - 7650 = 10660, so subtract the overlap"
    assert iou_high_ok, "iou(cat_true, cat_pred) should be 0.7176 — shared over together"
    assert iou_zero_ok, "iou(cat_true, dog) must be exactly 0.0 — clamp negative sides to 0"
    assert grid_ok, "the pixel grid must print as (320, 400): rows are y, columns are x"
    assert pixels_ok, "counting pixels must give 7650/10660 and 0/17000, same floats"
    assert prediction_ok, "the computed prediction must match reality, and differ per pair"
    assert raw_ok, "the raw detector dump should be the 5 recorded box lines"
    assert nms_kept_ok, "NMS should keep [55,52,152,142]@0.95 then [300,200,380,300]@0.90"
    assert duplicates_ok, "the suppressed boxes should score IoU 0.9082, 0.8744, 0.8333"
    assert dog_apart_ok, "the dog box overlaps the winner by 0.0, so it must not be hushed"
    assert tie_ok, "a box at IoU exactly 0.5 must survive — the rule is ABOVE the cut-off"
    assert threshold_ok, ("score > 0.85 should leave [0.95, 0.88, 0.90], then NMS "
                          "[0.95, 0.90], and must DROP a box scoring exactly 0.85")
