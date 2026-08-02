// shelf.js — Toy Shelf logic. Pure: no DOM, no localStorage, no rendering.
// Inlined into sessions/index.html between the SHELF-LOGIC markers; keep the two
// copies identical (sessions/_shelf_audit.py enforces it).

// A toy unlocks as soon as its owning lesson has ANY completed section.
// This is deliberately looser than the hub's pillStatus(), which requires
// done.produce || done.verdict — a learner meets a toy mid-lesson, so demanding
// completion would lock a toy they have already played with.
export function isUnlocked(raw){
  if (!raw) return false;
  try {
    var s = JSON.parse(raw);
    return !!(s && s.done && Object.keys(s.done).length > 0);
  } catch (e) { return false; }   // corrupt state must never break the hub
}
