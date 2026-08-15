// Run: node sessions/_compiler/tests/test_lang_switch.mjs   (exit 0 = pass)
//
// Tests the reading-language switcher that lives INLINE in v9-base.donor.
//
// Why it extracts from the donor instead of importing a module: shells/js/sr.js
// and shells/js/reveal.js are hand-maintained MIRRORS of code inlined in the
// donor ("mirror of shells/js/reveal.js", donor line ~602) and nothing checks
// that the two stay equal — a mirror can pass while the shipped page is broken.
// This test slices the real functions out of the donor text and runs them
// against a hand-rolled DOM stub (the same bare-node approach as test_sr.mjs,
// no jsdom in this sandbox), so drift is impossible by construction.
import fs from 'node:fs'
import assert from 'node:assert'
import path from 'node:path'

const DONOR = path.join(import.meta.dirname, '..', 'shells', 'v9-base.donor')
const donor = fs.readFileSync(DONOR, 'utf8')

// --- slice the two regions under test out of the donor ----------------------
function slice(from, to, label) {
  const a = donor.indexOf(from)
  assert.ok(a >= 0, `donor marker missing (${label} start): ${from}`)
  const b = donor.indexOf(to, a)
  assert.ok(b >= 0, `donor marker missing (${label} end): ${to}`)
  return donor.slice(a, b + to.length)
}

const checklistSrc = slice(
  "var checklist = document.getElementById('checklist')",
  'buildChecklist();',
  'checklist builder')
const langSrc = slice(
  "var langBtns = Array.prototype.slice.call(document.querySelectorAll('.lang-btn'));",
  "setLang(document.documentElement.getAttribute('data-lang') || 'en', false);",
  'language switcher')
const uiSrc = slice('var UI = {', '\nfunction ui(k){', 'ui table').replace(/\nfunction ui\(k\)\{$/, '')
const uiFnSrc = slice('function ui(k){', '\n}', 'ui lookup')

// --- DOM stub ---------------------------------------------------------------
function el(attrs = {}, text = '', children = []) {
  const classes = new Set()
  const n = {
    attrs: { ...attrs }, children, _text: text, handlers: [],
    classList: {
      add: (c) => classes.add(c), remove: (c) => classes.delete(c),
      contains: (c) => classes.has(c),
      toggle: (c, on) => (on ? classes.add(c) : classes.delete(c)),
    },
    getAttribute: (k) => (k in n.attrs ? n.attrs[k] : null),
    setAttribute: (k, v) => { n.attrs[k] = String(v) },
    addEventListener: (_ev, fn) => n.handlers.push(fn),
    click: () => n.handlers.forEach((f) => f()),
    appendChild: (c) => n.children.push(c),
    get textContent() {
      // matches the browser: display:none does NOT remove a node from textContent
      return n._text + n.children.map((c) => c.textContent).join('')
    },
    set innerHTML(v) {
      // Approximates the browser closely enough for this test: assigning
      // innerHTML replaces the node's content, and textContent then returns that
      // content with the tags stripped.
      n.children = []
      n._text = v === '' ? '' : String(v).replace(/<[^>]*>/g, '')
    },
    hasClass: (c) => classes.has(c),
  }
  n.querySelector = (sel) => n.querySelectorAll(sel)[0] || null
  n.querySelectorAll = (sel) => {
    const want = sel.replace(/^\./, '')
    const out = []
    const walk = (m) => (m.children || []).forEach((ch) => {
      if ((ch.attrs.class || '').split(' ').includes(want)) out.push(ch)
      walk(ch)
    })
    walk(n)
    return out
  }
  return n
}

function harness({ withZh, sections, storedLang }) {
  const store = {}
  if (storedLang !== undefined) store['frontier-lang'] = storedLang
  const localStorage = {
    getItem: (k) => (k in store ? store[k] : null),
    setItem: (k, v) => { store[k] = String(v) },
  }
  const htmlEl = el({ 'data-lang': storedLang === 'zh' && withZh ? 'zh' : 'en' })
  const enBtn = el({ class: 'lang-btn', 'data-lang-set': 'en' })
  const zhBtn = el({ class: 'lang-btn', 'data-lang-set': 'zh' })
  const checklist = el({ class: 'checklist' })
  // `secs` must hold ONLY .module-section nodes, exactly as the donor's
  // querySelectorAll('.module-section') would. The Chinese body content lives in
  // the page but is not a section, so it goes in the tree and not in secs.
  const extra = withZh ? [el({ class: 'lang-zh' }, '中文内容')] : []
  const root = el({}, '', [enBtn, zhBtn, checklist, ...sections, ...extra])

  const document = {
    documentElement: htmlEl,
    getElementById: (id) => (id === 'checklist' ? checklist : null),
    createElement: () => el(),
    querySelector: (sel) => root.querySelector(sel),
    querySelectorAll: (sel) => root.querySelectorAll(sel),
  }
  const refreshCalls = { n: 0 }
  const src = `
    ${checklistSrc}
    ${langSrc}
    return { setLang: setLang, secLabel: secLabel, buildChecklist: buildChecklist,
             lbl: function(){ return _lblLang }, items: function(){ return checkItems } };
  `
  const factory = new Function('document', 'localStorage', 'secs', 'RM', 'shorten', 'refresh', src)
  const api = factory(document, localStorage, sections, false,
                      (t) => (t.length > 34 ? t.slice(0, 32).trim() + '…' : t),
                      () => { refreshCalls.n++ })
  return { api, htmlEl, enBtn, zhBtn, checklist, store, refreshCalls }
}

// a section whose .sec-h carries BOTH languages, as the compiler will emit
function pairedSection(dataSec, num, en, zh) {
  const secH = el({ class: 'sec-h' }, '', [
    el({ class: 'lang-en' }, en),
    el({ class: 'lang-zh' }, zh),
  ])
  const secNum = el({ class: 'sec-num' }, num)
  return el({ class: 'module-section', 'data-sec': dataSec }, '', [secNum, secH])
}

function englishOnlySection(dataSec, num, en) {
  return el({ class: 'module-section', 'data-sec': dataSec }, '', [
    el({ class: 'sec-num' }, num),
    el({ class: 'sec-h' }, en),
  ])
}

// ============================================================================
let n = 0
const t = (name, fn) => { fn(); n++ }

// --- the toggle moves BOTH attributes ---------------------------------------
// data-lang alone is not enough: a screen reader on a page still marked
// lang="en" reads CJK with an English voice, which is worse than not
// translating it at all.
t('setLang sets data-lang AND html lang', () => {
  const h = harness({ withZh: true, sections: [pairedSection('c1', '1', 'The bend', '这个弯')] })
  h.api.setLang('zh')
  assert.equal(h.htmlEl.getAttribute('data-lang'), 'zh')
  assert.equal(h.htmlEl.getAttribute('lang'), 'zh-Hans')
  h.api.setLang('en')
  assert.equal(h.htmlEl.getAttribute('data-lang'), 'en')
  assert.equal(h.htmlEl.getAttribute('lang'), 'en')
})

t('the pre-paint IIFE in the donor also sets both attributes', () => {
  const m = donor.match(/<script>\/\* set reading language before paint[\s\S]*?<\/script>/)
  assert.ok(m, 'no reading-language pre-paint script in the donor')
  assert.ok(m[0].includes("setAttribute('data-lang'"), 'pre-paint does not set data-lang')
  assert.ok(m[0].includes("setAttribute('lang'"), 'pre-paint does not set lang — first paint would be mis-announced')
  assert.ok(m[0].includes('zh-Hans'), 'pre-paint does not use a BCP-47 Chinese tag')
})

// --- persistence ------------------------------------------------------------
t('choosing a language persists it', () => {
  const h = harness({ withZh: true, sections: [pairedSection('c1', '1', 'a', '甲')] })
  h.api.setLang('zh')
  assert.equal(h.store['frontier-lang'], 'zh')
})

t('a page with NO Chinese must not persist the forced fallback', () => {
  // Without this guard, opening one untranslated day silently resets the
  // reader's language for the whole site.
  const h = harness({ withZh: false, storedLang: 'zh', sections: [englishOnlySection('c1', '1', 'a')] })
  h.api.setLang('zh')
  assert.equal(h.htmlEl.getAttribute('data-lang'), 'en', 'should fall back to English')
  assert.equal(h.store['frontier-lang'], 'zh', 'the stored preference was overwritten')
})

t('init does not persist either', () => {
  const h = harness({ withZh: true, sections: [pairedSection('c1', '1', 'a', '甲')] })
  assert.equal('frontier-lang' in h.store, false, 'init wrote localStorage on a page nobody clicked')
})

// --- the disabled state -----------------------------------------------------
t('the 中文 button is marked disabled on a page with no Chinese', () => {
  const h = harness({ withZh: false, sections: [englishOnlySection('c1', '1', 'a')] })
  assert.equal(h.zhBtn.getAttribute('aria-disabled'), 'true')
  assert.ok((h.zhBtn.getAttribute('title') || '').includes('中文'))
  assert.equal(h.enBtn.getAttribute('aria-disabled'), null)
})

t('the 中文 button is NOT disabled once the page has Chinese', () => {
  const h = harness({ withZh: true, sections: [pairedSection('c1', '1', 'a', '甲')] })
  assert.equal(h.zhBtn.getAttribute('aria-disabled'), null)
})

t('clicking a button switches the language', () => {
  const h = harness({ withZh: true, sections: [pairedSection('c1', '1', 'a', '甲')] })
  h.zhBtn.click()
  assert.equal(h.htmlEl.getAttribute('data-lang'), 'zh')
  assert.ok(h.zhBtn.hasClass('active'))
  assert.ok(!h.enBtn.hasClass('active'))
})

// --- checklist labels -------------------------------------------------------
// display:none hides a span visually but textContent STILL returns it, so a
// naive read of .sec-h splices the two languages into one label.
t('a checklist label never splices the two languages', () => {
  const secs = [pairedSection('c1', '1', 'The bend', '这个弯')]
  const h = harness({ withZh: true, sections: secs })
  const label = () => h.checklist.children[0].textContent
  assert.ok(label().includes('The bend'), label())
  assert.ok(!label().includes('这个弯'), 'English label contains Chinese: ' + label())
  h.api.setLang('zh')
  assert.ok(label().includes('这个弯'), label())
  assert.ok(!label().includes('The bend'), 'Chinese label contains English: ' + label())
})

t('an unpaired .sec-h still yields its label in both languages', () => {
  const h = harness({ withZh: true, sections: [englishOnlySection('c1', '1', 'Arrays')] })
  assert.ok(h.checklist.children[0].textContent.includes('Arrays'))
  h.api.setLang('zh')
  assert.ok(h.checklist.children[0].textContent.includes('Arrays'), 'fallback label lost')
})

t('switching language rebuilds the checklist exactly once and refreshes', () => {
  const h = harness({ withZh: true, sections: [pairedSection('c1', '1', 'a', '甲')] })
  const before = h.refreshCalls.n
  h.api.setLang('zh')
  assert.equal(h.refreshCalls.n, before + 1)
  h.api.setLang('zh')                       // same language again
  assert.equal(h.refreshCalls.n, before + 1, 'rebuilt on a no-op switch')
  assert.equal(h.checklist.children.length, 1, 'checklist duplicated on rebuild')
})

t('the checklist keys stay the data-sec values across a switch', () => {
  // progress is stored per data-sec; a rebuild must not change the keys or the
  // reader loses their ticks on every toggle.
  const h = harness({ withZh: true, sections: [pairedSection('c1', '1', 'a', '甲')] })
  assert.deepEqual(Object.keys(h.api.items()), ['c1'])
  h.api.setLang('zh')
  assert.deepEqual(Object.keys(h.api.items()), ['c1'])
})

// --- the hub / roadmap shape: no checklist, no refresh() ---------------------
// index.html and roadmap.html carry the same sidebar shell but have no #checklist
// and no refresh(). The sweep pastes the language controller onto them verbatim,
// so setLang must survive both being absent — an unguarded call would throw and
// kill the whole IIFE, taking the theme switcher down with it.
t('the language controller works on a page with no checklist and no refresh', () => {
  const store = {}
  const localStorage = {
    getItem: (k) => (k in store ? store[k] : null),
    setItem: (k, v) => { store[k] = String(v) },
  }
  const htmlEl = el({ 'data-lang': 'en' })
  const enBtn = el({ class: 'lang-btn', 'data-lang-set': 'en' })
  const zhBtn = el({ class: 'lang-btn', 'data-lang-set': 'zh' })
  const zhBody = el({ class: 'lang-zh' }, '中文内容')
  const root = el({}, '', [enBtn, zhBtn, zhBody])
  const document = {
    documentElement: htmlEl,
    getElementById: () => null,
    createElement: () => el(),
    querySelector: (sel) => root.querySelector(sel),
    querySelectorAll: (sel) => root.querySelectorAll(sel),
  }
  // NOTE: no `secs`, no `shorten`, no `refresh`, and langSrc only — exactly the
  // scope those two pages provide.
  const factory = new Function('document', 'localStorage',
    `${langSrc}\nreturn { setLang: setLang };`)
  const api = factory(document, localStorage)
  api.setLang('zh')
  assert.equal(htmlEl.getAttribute('data-lang'), 'zh')
  assert.equal(htmlEl.getAttribute('lang'), 'zh-Hans')
  assert.equal(store['frontier-lang'], 'zh')
  api.setLang('en')
  assert.equal(htmlEl.getAttribute('data-lang'), 'en')
})

// --- runtime UI strings ------------------------------------------------------
// The strings no CSS toggle can reach, because the code REPLACES textContent. If
// these do not follow the language, the page flips back to English the moment the
// reader presses anything.
t('ui() returns the string for the current language', () => {
  const htmlEl = el({ 'data-lang': 'en' })
  const document = { documentElement: htmlEl }
  const api = new Function('document', `${uiSrc}\n${uiFnSrc}\nreturn {ui: ui, UI: UI};`)(document)
  const keys = ['reveal_done', 'all_answered', 'hints_end', 'hint_more',
                'copied', 'copy_manual', 'reset_confirm', 'sections_done']
  for (const k of keys) {
    assert.ok(api.UI.en[k], `en table missing ${k}`)
    assert.ok(api.UI.zh[k], `zh table missing ${k}`)
    htmlEl.setAttribute('data-lang', 'en')
    assert.equal(api.ui(k), api.UI.en[k], k)
    htmlEl.setAttribute('data-lang', 'zh')
    assert.equal(api.ui(k), api.UI.zh[k], k)
    assert.notEqual(api.UI.en[k], api.UI.zh[k], `${k} is the same in both languages`)
  }
})

t('ui() falls back to English for an unknown language or key', () => {
  const htmlEl = el({ 'data-lang': 'de' })
  const document = { documentElement: htmlEl }
  const api = new Function('document', `${uiSrc}\n${uiFnSrc}\nreturn {ui: ui, UI: UI};`)(document)
  assert.equal(api.ui('copied'), api.UI.en.copied)
  assert.equal(api.ui('no_such_key'), undefined)
})

console.log(`ok: language switcher (${n} assertions groups, extracted live from v9-base.donor)`)
