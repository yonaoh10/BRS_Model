#!/usr/bin/env python3
"""Automated QA for the dashboard prototype.

Checks the RENDERED page, not the source: horizontal overflow, computed
contrast of every visible text node against its actual painted background,
focus visibility, RTL/bidi correctness, tap-target size, and that no view is
left empty. Runs light+dark x desktop+phone.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

CHROME = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"
PAGE = Path(__file__).parent / "prototype.html"
OUT = Path(__file__).parent / "qa-output"
OUT.mkdir(exist_ok=True)

# Computes contrast against the nearest opaque painted ancestor background,
# which is what the eye actually sees (a transparent card over a page colour).
AUDIT_JS = r"""
() => {
  const problems = [];
  const lum = (r,g,b) => {
    const f = v => { v/=255; return v<=0.04045 ? v/12.92 : Math.pow((v+0.055)/1.055,2.4); };
    return 0.2126*f(r)+0.7152*f(g)+0.0722*f(b);
  };
  const parse = c => {
    const m = c.match(/rgba?\(([\d.]+),\s*([\d.]+),\s*([\d.]+)(?:,\s*([\d.]+))?\)/);
    return m ? {r:+m[1],g:+m[2],b:+m[3],a:m[4]===undefined?1:+m[4]} : null;
  };
  const bgOf = el => {
    let n = el;
    while (n && n !== document.documentElement) {
      const c = parse(getComputedStyle(n).backgroundColor);
      if (c && c.a > 0.9) return c;
      n = n.parentElement;
    }
    const c = parse(getComputedStyle(document.body).backgroundColor);
    return c || {r:255,g:255,b:255,a:1};
  };
  const ratio = (f,b) => {
    const L1 = lum(f.r,f.g,f.b), L2 = lum(b.r,b.g,b.b);
    const hi = Math.max(L1,L2), lo = Math.min(L1,L2);
    return (hi+0.05)/(lo+0.05);
  };

  // 1. contrast of every visible text node
  document.querySelectorAll('body *').forEach(el => {
    const txt = Array.from(el.childNodes)
      .filter(n => n.nodeType === 3).map(n => n.textContent.trim()).join('');
    if (!txt) return;
    const r = el.getBoundingClientRect();
    if (r.width === 0 || r.height === 0) return;
    const cs = getComputedStyle(el);
    if (cs.visibility === 'hidden' || cs.display === 'none' || +cs.opacity === 0) return;
    const fg = parse(cs.color); if (!fg) return;
    const bg = bgOf(el);
    const cr = ratio(fg,bg);
    const size = parseFloat(cs.fontSize);
    const bold = (parseInt(cs.fontWeight)||400) >= 700;
    const large = size >= 24 || (size >= 18.66 && bold);
    const need = large ? 3.0 : 4.5;
    if (cr < need) problems.push({
      kind:'contrast', need, got:+cr.toFixed(2),
      text: txt.slice(0,40), sel: el.tagName.toLowerCase()+'.'+(el.className||'').toString().split(' ')[0],
      fontSize: size, color: cs.color, bg: `rgb(${bg.r},${bg.g},${bg.b})`
    });
  });

  // 2. horizontal overflow
  const de = document.documentElement;
  if (de.scrollWidth > de.clientWidth + 1) {
    const wide = [];
    document.querySelectorAll('body *').forEach(el => {
      const r = el.getBoundingClientRect();
      if (r.right > de.clientWidth + 1 || r.left < -1) {
        const cs = getComputedStyle(el);
        if (cs.position === 'fixed') return;
        wide.push({sel: el.tagName.toLowerCase()+'.'+(el.className||'').toString().split(' ')[0],
                   right: Math.round(r.right), left: Math.round(r.left)});
      }
    });
    problems.push({kind:'overflow-x', scrollWidth: de.scrollWidth,
                   clientWidth: de.clientWidth, offenders: wide.slice(0,6)});
  }

  // 3. tap targets below 24px (WCAG 2.2 target size minimum)
  document.querySelectorAll('button, a, [tabindex]:not([tabindex="-1"])').forEach(el => {
    const r = el.getBoundingClientRect();
    if (r.width === 0 || r.height === 0) return;
    if (r.height < 24 || r.width < 24) problems.push({
      kind:'tap-target', w: Math.round(r.width), h: Math.round(r.height),
      text: (el.textContent||'').trim().slice(0,25),
      sel: el.tagName.toLowerCase()+'.'+(el.className||'').toString().split(' ')[0]
    });
  });

  // 4. bidi: latin/digit runs inside RTL that are NOT isolated will reorder
  const unisolated = [];
  document.querySelectorAll('td, dd, .val, .fig, .v, .amt, .sub, .foot').forEach(el => {
    const cs = getComputedStyle(el);
    const own = Array.from(el.childNodes).filter(n=>n.nodeType===3)
      .map(n=>n.textContent).join('');
    // a bare latin/number run directly in an RTL box with no isolation
    if (/[A-Za-z0-9]{2,}/.test(own) && cs.direction === 'rtl' &&
        cs.unicodeBidi !== 'isolate' && cs.unicodeBidi !== 'plaintext') {
      unisolated.push({sel: el.tagName.toLowerCase()+'.'+(el.className||'').toString().split(' ')[0],
                       text: own.trim().slice(0,30)});
    }
  });
  if (unisolated.length) problems.push({kind:'bidi-unisolated', items: unisolated.slice(0,8)});

  // 5. every nav destination renders something
  const empties = [];
  document.querySelectorAll('.view').forEach(v => {
    if (v.hidden) return;
    if (v.innerText.trim().length < 40) empties.push(v.id);
  });
  if (empties.length) problems.push({kind:'empty-view', views: empties});

  return problems;
}
"""

FOCUS_JS = r"""
() => {
  // Tab to the first few focusables and confirm each shows a visible ring.
  const els = Array.from(document.querySelectorAll('button, a, [tabindex]:not([tabindex="-1"])'))
    .filter(e => e.getBoundingClientRect().width > 0).slice(0, 12);
  const bad = [];
  els.forEach(e => {
    e.focus();
    const cs = getComputedStyle(e);
    const hasRing = (cs.boxShadow && cs.boxShadow !== 'none') ||
                    (cs.outlineStyle !== 'none' && parseFloat(cs.outlineWidth) > 0);
    if (!hasRing) bad.push({sel: e.tagName.toLowerCase()+'.'+(e.className||'').toString().split(' ')[0],
                            text:(e.textContent||'').trim().slice(0,20)});
  });
  document.activeElement && document.activeElement.blur();
  return bad;
}
"""

# The money meter is the one element that must never scroll out of view -
# the whole point of the state strip. Guarded so it cannot silently regress.
METER_JS = """
() => {
  const m = document.querySelector('.meter .amt');
  if (!m) return false;
  const r = m.getBoundingClientRect();
  return r.bottom > 0 && r.top < innerHeight;
}
"""

MODES = [
    ("light-desktop", {"width": 1280, "height": 900}, "light"),
    ("dark-desktop",  {"width": 1280, "height": 900}, "dark"),
    ("light-phone",   {"width": 390,  "height": 844}, "light"),
    ("dark-phone",    {"width": 390,  "height": 844}, "dark"),
]

def main() -> int:
    findings: dict[str, list] = {}
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=CHROME)
        for name, vp, theme in MODES:
            ctx = browser.new_context(viewport=vp, color_scheme=theme,
                                      device_scale_factor=2, locale="he-IL")
            page = ctx.new_page()
            page.goto(PAGE.as_uri())
            page.wait_for_timeout(1200)  # let webfonts settle
            probs = page.evaluate(AUDIT_JS)
            focus_bad = page.evaluate(FOCUS_JS)
            if focus_bad:
                probs.append({"kind": "focus-invisible", "items": focus_bad})
            page.mouse.wheel(0, 900)
            page.wait_for_timeout(300)
            if not page.evaluate(METER_JS):
                probs.append({'kind': 'meter-scrolled-away',
                              'note': 'cost meter left the viewport after scrolling'})
            page.mouse.wheel(0, -900)
            page.wait_for_timeout(200)
            findings[name] = probs
            page.screenshot(path=str(OUT / f"{name}.png"), full_page=True)

            # also capture the detail drawer, which the default view never shows
            if name == "light-desktop":
                page.click(".navlink[data-view='calls']")
                page.wait_for_timeout(300)
                page.click("#allTable tr[data-call='CALL003']")
                page.wait_for_timeout(400)
                page.screenshot(path=str(OUT / "drawer.png"))
                drawer_probs = page.evaluate(AUDIT_JS)
                findings["drawer-light"] = drawer_probs
            ctx.close()
        browser.close()

    # ---- report -------------------------------------------------------
    total = 0
    for mode, probs in findings.items():
        # dedupe contrast findings by (selector, text)
        seen, uniq = set(), []
        for pr in probs:
            key = (pr.get("kind"), pr.get("sel"), pr.get("text"))
            if key in seen:
                continue
            seen.add(key)
            uniq.append(pr)
        findings[mode] = uniq
        total += len(uniq)
        print(f"\n=== {mode}: {len(uniq)} finding(s) ===")
        for pr in uniq[:14]:
            k = pr["kind"]
            if k == "contrast":
                print(f"  CONTRAST {pr['got']}:1 (need {pr['need']}) "
                      f"{pr['sel']:24} {pr['fontSize']:.0f}px  {pr['color']} on {pr['bg']}  "
                      f"'{pr['text']}'")
            elif k == "overflow-x":
                print(f"  OVERFLOW-X page scrolls {pr['scrollWidth']}px in {pr['clientWidth']}px")
                for o in pr["offenders"]:
                    print(f"      -> {o['sel']} spans {o['left']}..{o['right']}")
            elif k == "tap-target":
                print(f"  TAP-TARGET {pr['w']}x{pr['h']}px  {pr['sel']:22} '{pr['text']}'")
            elif k == "bidi-unisolated":
                for i in pr["items"]:
                    print(f"  BIDI not isolated: {i['sel']:20} '{i['text']}'")
            elif k == "focus-invisible":
                for i in pr["items"]:
                    print(f"  FOCUS invisible: {i['sel']:20} '{i['text']}'")
            else:
                print(f"  {k}: {pr}")

    (OUT / "findings.json").write_text(json.dumps(findings, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    print(f"\n{'='*64}\nTOTAL: {total} finding(s). Screenshots + findings.json in {OUT}")
    return 0 if total == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
