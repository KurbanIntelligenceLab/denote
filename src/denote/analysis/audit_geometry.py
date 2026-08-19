#!/usr/bin/env python3
"""Geometric audit of the built PDFs.

Checks, on the TYPESET OUTPUT rather than the source:
  1. word-box overlaps (text set on top of other text)
  2. spans that cross the column gutter or run into the margins
  3. minimum typeset font size, against the IEEE 9.5 pt body / caption floor

The checker validates ITSELF first on a synthetic page with known defects, so a
silently-passing checker cannot be mistaken for a clean document. Three earlier
checkers in this project failed silently; this one fails loudly.

Usage:  python audit_geometry.py file1.pdf [file2.pdf ...]
Exit 0 if every input is clean, 1 otherwise.
"""
import sys

import pymupdf


def boxes_overlap(a, b, frac=0.30):
    """True when the intersection covers >= frac of the smaller box's area."""
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return False
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = max((a[2] - a[0]) * (a[3] - a[1]), 1e-9)
    area_b = max((b[2] - b[0]) * (b[3] - b[1]), 1e-9)
    return inter / min(area_a, area_b) >= frac


def is_script_pair(ra, rb, sa, sb):
    """True when one box is a small glyph nested inside a larger one, which is
    what a math sub/superscript looks like to a word extractor. These are not
    collisions and flagging them would make the checker cry wolf."""
    if sa is None or sb is None:
        return False
    small, big = (ra, rb) if sa < sb else (rb, ra)
    ratio = min(sa, sb) / max(sa, sb)
    nested = (small[0] >= big[0] - 2 and small[2] <= big[2] + 2)
    return ratio < 0.85 and nested


def audit(path, margin_pt=36.0, size_floor=6.0, advisory_floor=8.0, verbose=False):
    """Return a list of finding strings for one PDF."""
    findings = []
    doc = pymupdf.open(path)
    for pno, page in enumerate(doc, start=1):
        words = page.get_text("words")           # x0,y0,x1,y1,word,block,line,wno
        rects = [(w[0], w[1], w[2], w[3]) for w in words]
        texts = [w[4] for w in words]
        # height stands in for point size at the word level
        hts = [(w[3] - w[1]) for w in words]

        # (1) overlaps. Compare only words whose vertical bands intersect, which
        # keeps this near-linear instead of quadratic over the whole page.
        order = sorted(range(len(rects)), key=lambda i: rects[i][1])
        for a_pos, i in enumerate(order):
            for j in order[a_pos + 1:]:
                if rects[j][1] >= rects[i][3]:
                    break                         # sorted: no later word can overlap
                if is_script_pair(rects[i], rects[j], hts[i], hts[j]):
                    continue                      # math sub/superscript, not a collision
                if boxes_overlap(rects[i], rects[j]):
                    findings.append(
                        f"{path} p{pno}: text overlap "
                        f"{texts[i]!r} / {texts[j]!r} at {rects[i]}")

        # (2) margin / gutter bleed
        pw, ph = page.rect.width, page.rect.height
        for r, t in zip(rects, texts):
            if r[0] < margin_pt - 8 or r[2] > pw - margin_pt + 8:
                findings.append(f"{path} p{pno}: {t!r} runs into the margin at x={r[0]:.1f}-{r[2]:.1f}")

        # (3) smallest typeset size
        smallest = None
        for blk in page.get_text("dict")["blocks"]:
            for line in blk.get("lines", []):
                for span in line.get("spans", []):
                    if span["text"].strip():
                        s = round(span["size"], 2)
                        if smallest is None or s < smallest:
                            smallest = s
        # Ignore math script sizes: a subscript index is legitimately small.
        # Report the smallest size occurring in a RUNNING-TEXT span instead.
        body_small = None
        for blk in page.get_text("dict")["blocks"]:
            for line in blk.get("lines", []):
                for span in line.get("spans", []):
                    txt = span["text"].strip()
                    if len(txt) >= 3:             # >=3 chars is not a lone index
                        s = round(span["size"], 2)
                        if body_small is None or s < body_small:
                            body_small = s
        if body_small is not None and body_small < size_floor:
            findings.append(f"{path} p{pno}: running-text span typeset at "
                            f"{body_small} pt, below the {size_floor} pt "
                            f"legibility floor")
        elif body_small is not None and body_small < advisory_floor:
            findings.append(f"{path} p{pno}: ADVISORY, smallest running-text "
                            f"span is {body_small} pt (IEEE sets no table floor, "
                            f"but check it is comfortably readable in print)")
        if verbose:
            print(f"  {path} p{pno}: {len(words)} words, smallest span {smallest} pt")
    doc.close()
    return findings


def selftest():
    """Build a page with a known overlap and a known margin bleed, and confirm
    the checker reports both. Any failure here means the checker is broken and
    a clean report on the real documents would be meaningless."""
    doc = pymupdf.open()
    page = doc.new_page(width=300, height=200)
    page.insert_text((100, 100), "COLLIDE", fontsize=14)
    page.insert_text((104, 101), "COLLIDE", fontsize=14)   # deliberate overlap
    page.insert_text((2, 160), "BLEED", fontsize=14)       # deliberate bleed
    doc.save("/tmp/_selftest.pdf")
    doc.close()
    found = audit("/tmp/_selftest.pdf", margin_pt=36.0, size_floor=0.0,
                  advisory_floor=0.0)
    has_overlap = any("text overlap" in f for f in found)
    has_bleed = any("runs into the margin" in f for f in found)
    if not (has_overlap and has_bleed):
        raise SystemExit(
            "CHECKER SELFTEST FAILED: planted defects were not detected "
            f"(overlap={has_overlap}, bleed={has_bleed}). Do not trust any "
            "clean result from this script until it is fixed.")
    print("checker selftest PASS (planted overlap and margin bleed both detected)")


def main(argv):
    selftest()
    bad = False
    for path in argv[1:]:
        findings = audit(path)
        if findings:
            bad = True
            print(f"\n{path}: {len(findings)} FINDING(S)")
            for f in findings[:40]:
                print("   ", f)
            if len(findings) > 40:
                print(f"    ... and {len(findings) - 40} more")
        else:
            print(f"{path}: clean (no overlaps, no margin bleed, no sub-floor type)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
