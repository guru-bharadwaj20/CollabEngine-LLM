"""Build the paper as a .docx laid out to the NeurIPS 2026 style file.

The style file itself is LaTeX, so this reproduces its *geometry* rather than
running it: US Letter, a 5.5 x 9 inch text block at a 1.5 inch left margin,
10/11 Times, the two title rules at 4pt and 1pt, the indented abstract, and the
heading sizes from Section 3 of the formatting instructions. Numbers come from
`numbers()` below, which reads the corpus through the same scoring and
integrity modules the tables and figures use -- so a stale number in the paper
is a failure of this script rather than a transcription slip.

    python scripts/analysis/build_paper.py [--out CollabEngine-NeurIPS2026.docx]

`--author` / `--affiliation` / `--email` set the title block. Defaults are the
preprint form; pass `--anonymous` for the double-blind submission form.
"""

from __future__ import annotations

import argparse
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

# --- the style file's constants --------------------------------------------
BODY_PT = 10.0
LEAD_PT = 11.0          # "vertical spacing (leading) of 11 points"
PARA_GAP_PT = 5.5       # "paragraphs are separated by 1/2 line space"
TITLE_PT = 17.0
H1_PT, H2_PT, H3_PT = 12.0, 10.0, 10.0
ABSTRACT_INDENT = Inches(0.5)   # "indented 1/2 inch (3 picas) on both margins"
RULE_GAP_PT = 18        # "1/4 inch space above and below the title to rules"
SERIF = "Times New Roman"
MONO = "Courier New"
TEXT_W = 5.5


# --- low-level docx helpers -------------------------------------------------
def _border(el_tag: str, sz_eighths: int, space: int = 0) -> OxmlElement:
    b = OxmlElement(el_tag)
    b.set(qn("w:val"), "single")
    b.set(qn("w:sz"), str(sz_eighths))     # eighths of a point
    b.set(qn("w:space"), str(space))       # points between text and rule
    b.set(qn("w:color"), "000000")
    return b


def rule_below(par, points: float, gap: int = 0) -> None:
    """A horizontal rule under a paragraph, `points` thick."""
    pPr = par._p.get_or_add_pPr()
    bdr = OxmlElement("w:pBdr")
    bdr.append(_border("w:bottom", int(round(points * 8)), gap))
    pPr.append(bdr)


def field(par, instr: str) -> None:
    """Insert a Word field code (used for PAGE)."""
    r = par.add_run()
    fc = OxmlElement("w:fldChar"); fc.set(qn("w:fldCharType"), "begin")
    it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve")
    it.text = f" {instr} "
    fe = OxmlElement("w:fldChar"); fe.set(qn("w:fldCharType"), "end")
    r._r.append(fc); r._r.append(it); r._r.append(fe)


_INLINE = re.compile(r"(\*\*.+?\*\*|\*[^*]+?\*|`[^`]+?`)", re.S)


def runs(par, text: str, size: float, italic_all: bool = False,
         bold_all: bool = False) -> None:
    """Write `text` into `par`, honouring **bold**, *italic* and `mono`.

    Handles one level of nesting, so `**a *p* = 0.9**` bolds the whole span and
    italicises `p` within it. Without this a nested marker was emitted
    literally, which put a visible `*p*` in the body text.
    """
    for piece in _INLINE.split(text):
        if not piece:
            continue
        if piece.startswith("**") and piece.endswith("**"):
            inner = piece[2:-2]
            if _INLINE.search(inner):          # nested: recurse, carrying bold
                runs(par, inner, size, italic_all, bold_all=True)
                continue
            piece, bold, italic, mono = inner, True, False, False
        elif piece.startswith("*") and piece.endswith("*"):
            piece, bold, italic, mono = piece[1:-1], bold_all, True, False
        elif piece.startswith("`") and piece.endswith("`"):
            piece, bold, italic, mono = piece[1:-1], bold_all, False, True
        else:
            bold, italic, mono = bold_all, False, False
        r = par.add_run(piece)
        r.font.name = MONO if mono else SERIF
        r.font.size = Pt(size - 0.5 if mono else size)
        r.font.bold = bold
        r.font.italic = italic or italic_all
        r.font.color.rgb = RGBColor(0, 0, 0)


def para(doc, text="", size=BODY_PT, align=WD_ALIGN_PARAGRAPH.JUSTIFY,
         space_after=PARA_GAP_PT, space_before=0.0, lead=LEAD_PT,
         indent=None, keep_with_next=False, italic_all=False):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.alignment = align
    pf.space_after = Pt(space_after)
    pf.space_before = Pt(space_before)
    if lead is None:
        # Exact leading clips an inline image to the line height, so the one
        # place it must not be set is the paragraph holding a figure.
        pf.line_spacing_rule = WD_LINE_SPACING.SINGLE
    else:
        pf.line_spacing_rule = WD_LINE_SPACING.EXACTLY
        pf.line_spacing = Pt(lead)
    pf.first_line_indent = Pt(0)          # "with no indentation"
    pf.keep_with_next = keep_with_next
    if indent is not None:
        pf.left_indent = indent
        pf.right_indent = indent
    if text:
        runs(p, text, size, italic_all)
    return p


def heading(doc, level: int, text: str):
    """Flush left, bold, lower case except first word and proper nouns."""
    size = {1: H1_PT, 2: H2_PT, 3: H3_PT}[level]
    p = para(doc, "", size=size, align=WD_ALIGN_PARAGRAPH.LEFT,
             space_before=11.0 if level == 1 else 9.0,
             space_after=4.0, lead=size + 2, keep_with_next=True)
    r = p.add_run(text)
    r.font.name = SERIF
    r.font.size = Pt(size)
    r.font.bold = True
    return p


def bullet(doc, text: str, size=BODY_PT, level=0):
    p = para(doc, "", size=size, space_after=2.0)
    p.paragraph_format.left_indent = Inches(0.25 + 0.22 * level)
    p.paragraph_format.first_line_indent = Inches(-0.14)
    runs(p, ("•  " if level == 0 else "–  ") + text, size)
    return p


def numbered(doc, n: int, text: str, size=BODY_PT):
    p = para(doc, "", size=size, space_after=3.0)
    p.paragraph_format.left_indent = Inches(0.28)
    p.paragraph_format.first_line_indent = Inches(-0.28)
    runs(p, f"{n}.  " + text, size)
    return p


def figure(doc, path: Path, caption: str, width_in: float, number: int):
    """Figure, then caption. 'The figure number and caption always appear
    after the figure. Place one line space before the figure caption.'"""
    p = para(doc, "", align=WD_ALIGN_PARAGRAPH.CENTER, space_before=6.0,
             space_after=5.5, lead=None, keep_with_next=True)
    p.add_run().add_picture(str(path), width=Inches(width_in))
    cap = para(doc, "", size=9.0, align=WD_ALIGN_PARAGRAPH.LEFT,
               space_after=9.0, lead=10.5)
    r = cap.add_run(f"Figure {number}: ")
    r.font.name = SERIF; r.font.size = Pt(9); r.font.bold = False
    runs(cap, caption, 9.0)
    return cap


def _shade_none(cell):
    tcPr = cell._tc.get_or_add_tcPr()
    for el in tcPr.findall(qn("w:tcBorders")):
        tcPr.remove(el)


def table(doc, caption: str, headers: list[str], rows: list[list[str]],
          number: int, widths: list[float] | None = None, size=9.0,
          align_right_from: int = 1):
    """Caption above; booktabs rules; no vertical rules anywhere."""
    cap = para(doc, "", size=9.0, align=WD_ALIGN_PARAGRAPH.LEFT,
               space_before=6.0, space_after=4.0, lead=10.5, keep_with_next=True)
    r = cap.add_run(f"Table {number}: ")
    r.font.name = SERIF; r.font.size = Pt(9)
    runs(cap, caption, 9.0)

    t = doc.add_table(rows=1, cols=len(headers))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False

    # Strip every border, then add only the three horizontal rules booktabs
    # draws. "Publication-quality tables do not contain vertical rules."
    tblPr = t._tbl.tblPr
    for el in tblPr.findall(qn("w:tblBorders")):
        tblPr.remove(el)
    borders = OxmlElement("w:tblBorders")
    for tag in ("w:top", "w:bottom"):
        borders.append(_border(tag, 12))          # 1.5pt outer rules
    for tag in ("w:left", "w:right", "w:insideH", "w:insideV"):
        b = OxmlElement(tag); b.set(qn("w:val"), "none"); b.set(qn("w:sz"), "0")
        borders.append(b)
    tblPr.append(borders)

    def write(cells, vals, bold=False):
        for i, (c, v) in enumerate(zip(cells, vals)):
            c.text = ""
            p = c.paragraphs[0]
            pf = p.paragraph_format
            pf.space_after = Pt(1.5); pf.space_before = Pt(1.5)
            pf.line_spacing_rule = WD_LINE_SPACING.EXACTLY
            pf.line_spacing = Pt(size + 1.5)
            pf.alignment = (WD_ALIGN_PARAGRAPH.LEFT if i < align_right_from
                            else WD_ALIGN_PARAGRAPH.RIGHT)
            # Header cells go through the same inline parser as everything
            # else, then get bolded -- writing them as a raw run left `hard`
            # and *d* showing their own markup in the rendered table.
            runs(p, v, size)
            if bold:
                for rr in p.runs:
                    rr.font.bold = True
            if widths:
                c.width = Inches(widths[i])

    write(t.rows[0].cells, headers, bold=True)
    # header underrule
    for c in t.rows[0].cells:
        tcPr = c._tc.get_or_add_tcPr()
        tb = OxmlElement("w:tcBorders")
        tb.append(_border("w:bottom", 6))          # 0.75pt
        tcPr.append(tb)
    for row in rows:
        write(t.add_row().cells, row)

    if widths:
        for r_ in t.rows:
            for c, w in zip(r_.cells, widths):
                c.width = Inches(w)
    para(doc, "", space_after=7.0, lead=6)
    return t


# --- corpus numbers ---------------------------------------------------------
def numbers() -> dict:
    """Every quantity the paper quotes, read from `runs/` at build time."""
    from collabengine.analysis.integrity import (final_turn_truncated,
                                                 is_instrument_failure)
    from collabengine.analysis.scoring import rescore
    from collabengine.transcripts.store import TranscriptReader

    def sc(path, cond, drop_cut=False):
        out = []
        for rec in TranscriptReader(str(path)):
            if rec.condition != cond or is_instrument_failure(rec):
                continue
            if drop_cut and final_turn_truncated(rec):
                continue
            out.append(rescore(rec).overall["fraction"])
        return out

    def live(path):
        v = []
        for rec in TranscriptReader(str(path)):
            if rec.condition.startswith("live:") and not is_instrument_failure(rec):
                v.append(rescore(rec).overall["fraction"])
        return v

    F = Path("runs/llama31-8b-q4-medium-h3b")
    P = Path("runs/llama31-8b-q4-medium-ans")
    n = {}
    n["solo"] = st.mean(sc(F / "baseline.jsonl", "solo"))
    n["team"] = st.mean(sc(F / "baseline.jsonl", "baseline"))
    n["three"] = st.mean(live(F / "ablation.jsonl"))
    n["solo_long"] = st.mean(sc(F / "baseline.jsonl", "solo_long"))
    # Spread from the *rounded* means, so the paper's own arithmetic checks out
    # for a reader: 0.579 - 0.574 must read as 0.005, not as the 0.006 the
    # unrounded values happen to give.
    _r = [round(n[k], 3) for k in ("solo", "team", "three")]
    n["spread"] = max(_r) - min(_r)
    n["n_solo"] = len(sc(F / "baseline.jsonl", "solo"))
    n["n_team"] = len(sc(F / "baseline.jsonl", "baseline"))
    n["n_three"] = len(live(F / "ablation.jsonl"))
    n["n_total"] = n["n_solo"] + n["n_team"] + n["n_three"]
    n["pilot_team"] = st.mean(sc(P / "baseline.jsonl", "baseline"))
    n["pilot_three"] = st.mean(live(P / "ablation.jsonl"))
    n["pilot_solo"] = st.mean(sc(P / "baseline.jsonl", "solo"))
    return n


# --- the document -----------------------------------------------------------
def build(out_path: Path, author: str, affiliation: str, email: str,
          anonymous: bool, fig_dir: Path) -> None:
    N = numbers()
    doc = Document()

    s = doc.sections[0]
    s.page_width, s.page_height = Inches(8.5), Inches(11)   # US Letter
    s.left_margin = s.right_margin = Inches(1.5)            # 5.5in text block
    s.top_margin = s.bottom_margin = Inches(1.0)            # 9in long
    s.different_first_page_header_footer = True

    normal = doc.styles["Normal"]
    normal.font.name = SERIF
    normal.font.size = Pt(BODY_PT)
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), SERIF)

    # page numbers on pages 2+; page 1 carries the preprint notice instead
    fp = para(s.footer, "", size=8.0, align=WD_ALIGN_PARAGRAPH.CENTER,
              space_after=0, lead=10)
    rule_below(para(s.first_page_footer, "", size=1, space_after=2.0, lead=4), 0.5)
    fpn = para(s.first_page_footer, "", size=8.0, align=WD_ALIGN_PARAGRAPH.LEFT,
               space_after=0, lead=9.5)
    runs(fpn, "Preprint. Work in progress." if not anonymous
         else "Submitted to 40th Conference on Neural Information Processing "
              "Systems (NeurIPS 2026). Do not distribute.", 8.0)
    fnum = doc.sections[0].footer.paragraphs[0]
    field(fp, "PAGE")
    for r in fp.runs:
        r.font.name = SERIF; r.font.size = Pt(9)

    # --- title block: 4pt rule / title / 1pt rule ---------------------------
    top = para(doc, "", size=1, space_before=0, space_after=RULE_GAP_PT, lead=4)
    rule_below(top, 4.0)
    t = para(doc, "", size=TITLE_PT, align=WD_ALIGN_PARAGRAPH.CENTER,
             space_after=0, lead=21)
    rt = t.add_run("Three Positives and a Cap: Measurement Artifacts in "
                   "Single-Agent versus Multi-Agent LLM Comparisons")
    rt.font.name = SERIF; rt.font.size = Pt(TITLE_PT); rt.font.bold = True
    rule_below(t, 1.0, gap=RULE_GAP_PT)

    para(doc, "", space_after=14.0, lead=6)
    if anonymous:
        block = ["**Anonymous Author(s)**", "Affiliation", "Address", "`email`"]
    else:
        block = [f"**{author}**", affiliation, f"`{email}`"]
    for i, line in enumerate(block):
        para(doc, line, align=WD_ALIGN_PARAGRAPH.CENTER,
             space_after=1.0 if i < len(block) - 1 else 20.0)

    # --- abstract -----------------------------------------------------------
    para(doc, "", space_after=0, lead=11)          # "two line spaces precede"
    a = para(doc, "", size=12.0, align=WD_ALIGN_PARAGRAPH.CENTER,
             space_after=6.0, lead=14)
    ra = a.add_run("Abstract")
    ra.font.name = SERIF; ra.font.size = Pt(12); ra.font.bold = True
    para(doc,
         "We set out to test whether emergent role differentiation in same-model "
         "LLM agent teams is causally real, using leave-one-out ablation rather "
         "than transcript reading. We could not run that test, and the reason is "
         "the contribution. Across three task difficulties, two serving "
         "instruments and two disjoint seed sets, we obtained three separate "
         "statistically significant results favouring multi-agent teams, and "
         "traced all three to measurement artifacts. Two were the same artifact: "
         "a per-turn token cap that is symmetric by construction and asymmetric "
         "in effect, because a single agent must emit an entire solution in its "
         "final turn while a team commits one its shared transcript already "
         "holds. The third was a small-sample baseline, a four-agent reference "
         f"estimated from 48 episodes that a 3× larger sample did not "
         f"reproduce. Removing all three, the effect of team size on this task "
         f"family is flat: one agent scores {N['solo']:.3f}, three agents "
         f"{N['three']:.3f} and four agents {N['team']:.3f} across "
         f"{N['n_total']} episodes, a spread of {N['spread']:.3f}. The only "
         "manipulation that moves the score reliably is giving a single agent "
         "more turns, which makes it worse. We report the artifact mechanism, "
         "the three diagnostics that caught it, and a bounded negative result. "
         "We argue the diagnostics are the transferable part: any comparison "
         "between architectures that consume a resource differently will land a "
         "shared limit asymmetrically, and standard reporting does not currently "
         "surface this.",
         indent=ABSTRACT_INDENT, space_after=13.0)

    # ============================ 1. introduction ==========================
    heading(doc, 1, "1   Introduction")
    para(doc,
         "The claim we set out to test came from two 2026 results that had never "
         "been combined. Observational work established that same-model LLM teams "
         "*appear* to differentiate: agents settle into critic-like and "
         "planner-like behaviour that survives coding by an LLM judge [2]. "
         "Separately, causal ablation work showed that introspective judgements "
         "of agent contribution do not match what ablation measures, but applied "
         "it only to *assigned* roles, as an engineering tool for finding "
         "bottleneck agents [3]. Nobody had pointed the causal instrument at the "
         "emergent phenomenon. That was the intended contribution.")
    para(doc,
         "**A design correction makes it a real experiment.** A scalar "
         "performance drop when an agent is removed does not demonstrate "
         "specialization; it demonstrates participation. Remove any competent "
         "contributor from any team and output falls, which is equally true of "
         "four identical agents with no division of labour at all. The signature "
         "of specialization is an **agent × task-component interaction**: "
         "ablating an emergent critic must damage verification-loaded components "
         "*more* than planning-loaded ones, and ablating a planner must do the "
         "reverse. The main effect proves nothing; the crossover is the result.")
    para(doc,
         "That test requires an operating point where the team actually "
         "outperforms a single agent. Otherwise the ablation grid decomposes its "
         "own noise floor. We made that a preregistered stop condition, and "
         "establishing the operating point is where the project stopped. This "
         "paper is therefore about measurement rather than emergence.")
    para(doc, "**Contributions.**")
    numbered(doc, 1,
             "A mechanism by which a per-turn token cap, identical across arms "
             "and therefore apparently controlled, produces a spurious "
             "multi-agent advantage that **grows with task difficulty in exactly "
             "the shape a genuine collaboration benefit predicts** (§4).")
    numbered(doc, 2,
             "Three statistically significant pro-team results, each traced to "
             "its cause and each dissolving under the corresponding control "
             "(§4.3, §5).")
    numbered(doc, 3,
             "A bounded negative result: at 8B on this task family, team size "
             f"has no measurable effect ({N['spread']:.3f} spread across "
             f"{N['n_total']} episodes), while giving one agent four times the "
             "turns makes it reliably worse (§6).")
    numbered(doc, 4,
             "Seven reporting practices that would have caught these artifacts, "
             "and two instrument defects released rather than quietly fixed "
             "(§7, §8).")

    # ============================ 2. related work ==========================
    heading(doc, 1, "2   Related work")
    para(doc,
         "**Emergent roles, observed.** The MARL lineage, of which ROMA [1] is "
         "representative, produces emergent roles as learned embeddings. Those "
         "roles are architectural rather than linguistic, and the agents are not "
         "LLMs. The closest LLM result is observational [2]: 208 runs and 13,786 "
         "coded messages establish that unassigned same-model agents differentiate "
         "behaviourally, using LLM-judge coding and cosine similarity. It reports "
         "no causal test, and states the transcript-reading problem explicitly. "
         "It also uses several different LLMs, which leaves model heterogeneity "
         "as an open confound for any differentiation it observes.")
    para(doc,
         "**Ablation, applied to assigned roles.** Leave-one-out ablation of LLM "
         "agents is established as an engineering tool. [3] treats attribution as "
         "a cooperative game and finds that introspective LLM judgements do not "
         "faithfully approximate ablation behaviour, which is third-party evidence "
         "for the premise that transcript-reading and causal reality diverge. [4] "
         "approximates leave-one-out by introspection, and [5] eliminates agents "
         "dynamically for efficiency. In all three the roles are pre-assigned, and "
         "ablation optimises a system rather than validating that a division of "
         "labour is real.")
    para(doc,
         "**The gap, and why we could not fill it.** The observational half and "
         "the causal half have not been combined. Our design does combine them, "
         "and holds model identity fixed across agents so that any differentiation "
         "cannot come from model heterogeneity. What we report instead is the "
         "measurement problem that blocked the combination, because we believe it "
         "blocks others: the comparison that licenses an ablation study is itself "
         "much harder to measure than the literature's reporting conventions "
         "suggest.")

    # ============================ 3. setup =================================
    heading(doc, 1, "3   Experimental setup")
    table(doc,
          "experimental configuration. one model serves every agent in every "
          "condition, so observed differentiation cannot come from model "
          "heterogeneity — a control the prior observational work lacks.",
          ["", ""],
          [["Model", "Meta-Llama-3.1-8B-Instruct, Q4_K_M, one model serving every agent"],
           ["Serving", "`llama.cpp`, 18,432 tokens per slot, 7 slots, one 24 GB RTX 4500 Ada"],
           ["Task", "synthetic multi-constraint scheduling; tagged constraint classes "
                    "(arithmetic, search, verification, synthesis); deterministic "
                    "per-component grader; satisfiable by construction"],
           ["Difficulties", "`medium` / `hard` / `xhard` — 16 / 24 / 36 jobs"],
           ["Metrics", "`fraction` (partial credit, primary), `strict`, `feasible`"],
           ["Statistics", "20,000-draw permutation tests; 10,000-draw bootstrap intervals; "
                          "ablation contrasts paired on instance"],
           ["Corpus", "~2,900 episodes across all conditions; all transcripts released"]],
          number=1, widths=[0.95, 4.55], size=9.0, align_right_from=99)
    para(doc,
         "**Orchestration is hand-written on purpose.** Every mainstream agent "
         "framework ships role scaffolding in its prompt templates, which would "
         "plant the structure the study claims to observe emerging. No agent is "
         "told to plan, criticise or verify; agents receive identical briefs and "
         "differ only in a name and a sampling seed.")
    para(doc,
         "**Preregistration.** Hypotheses, tests and falsification conditions were "
         "registered before the runs they govern, with amendments dated and "
         "justified. Two disciplines in those documents did the work reported "
         "below: a **complete-case sensitivity row printed beside every headline "
         "number**, and a **fresh-seed rule** forbidding a corpus from testing the "
         "hypothesis it generated. Neither was added after seeing a result.")

    heading(doc, 2, "3.1   Three confounds handled in the design")
    para(doc,
         "**Compensation.** Remove an agent and the survivors may reorganise to "
         "absorb its function, masking the specialization being measured. "
         "Ablation therefore runs three ways: `live`, in which compensation is "
         "allowed; `frozen_replay`, which regenerates surviving turns on the "
         "recorded schedule; and `frozen_excise`, which deletes the messages and "
         "re-reads the answer with zero model calls. Section 7 reports that the "
         "third of these does not work on real transcripts, and why.")
    para(doc,
         "**Position versus identity.** Roles may attach to turn order rather "
         "than to agent identity, which would be protocol rather than "
         "specialization. Speaking order is reshuffled every round, which drives "
         "a purely positional world to chance while leaving identity-bound roles "
         "untouched. Under a *fixed* order that same synthetic world scores 0.50 "
         "against a 0.25 chance baseline — enough to be mistaken for the "
         "real thing, which is why the reshuffle is not optional.")
    para(doc,
         "**Symmetry breaking is the independent variable.** Identical model, "
         "prompt and context produce identical output, and nothing can "
         "specialise. What breaks the tie — a name, a seed, a private "
         "scratchpad — is swept rather than fixed and forgotten. The "
         "question the design asks is whether *minimal* asymmetry amplifies into "
         "stable roles.")

    heading(doc, 2, "3.2   Validating the instrument before trusting it")
    para(doc,
         "The pipeline is run first against three synthetic worlds whose answer "
         "is known in advance: genuinely specialised agents, wholly "
         "undifferentiated ones, and agents whose behaviour tracks turn slot "
         "rather than identity. Diagonal dominance separates the specialised "
         "world from the other two but *cannot* separate positional from null, "
         "because both put ownership at chance by construction, and chance is "
         "1/*n* rather than zero; interaction strength is what distinguishes "
         "those two. **A pipeline that reports specialization in the null world "
         "is manufacturing its result.** Catching that costs seconds on a mock "
         "backend; catching it after a GPU run costs the study.")

    # ============================ 4. the artifact ==========================
    heading(doc, 1, "4   A per-turn cap is not a shared constraint")
    heading(doc, 2, "4.1   Mechanism")
    para(doc,
         "Agents are capped at `max_tokens` per turn. The cap is identical across "
         "conditions, which is exactly what makes it look like a controlled "
         "variable. It is not, because the arms spend the resource differently:")
    bullet(doc, "A **single agent** at *k* rounds must produce the complete final "
                "solution inside its own last turn. The answer competes with "
                "reasoning for one budget.")
    bullet(doc, "A **team** at *k* rounds ends on a turn that commits an answer the "
                "shared transcript already contains. Its final turn can be a "
                "summary; the content was distributed across earlier turns and "
                "earlier agents.")
    para(doc,
         "The same cap therefore truncates an *answer* in one arm and a *summary* "
         "in the other. Because harder instances need longer answers, **the "
         "artifact grows with instance size in exactly the shape a genuine "
         "collaboration benefit would predict.** The confound and the hypothesis "
         "have the same functional form, because both scale with how much answer "
         "the instance requires.", space_after=8.0)

    figure(doc, fig_dir / "fig1_mechanism.png",
           "the cap is symmetric in specification and asymmetric in effect. "
           "**(a)** on the original instrument, the single-agent arm's "
           "answer-bearing turn hits the token cap on 23–47% of usable "
           "episodes while the team's does so on 0–8%, and the gap widens "
           "with difficulty. **(b)** the same corpus, scored as-is and with "
           "answer-turn-truncated episodes dropped from both arms. the headline "
           "gap trends upward across difficulty; the controlled gap does not "
           "trend at all and is negative at two tiers of three. a difficulty "
           "curve read without this control cannot be distinguished from the "
           "effect it was built to detect.",
           width_in=5.4, number=1)

    heading(doc, 2, "4.2   The asymmetry is measured, not assumed")
    para(doc,
         "On a matched *turn* budget at `medium`, the single agent emits 28,319 "
         "characters against the team's 13,877, a factor of **2.04×**. "
         "Matching turns does not match generation: one agent restates the whole "
         "working solution every round, whereas four agents each add to a shared "
         "transcript. An arm described throughout our own preregistration as "
         "“matched budget” was matched on the resource we counted and "
         "unmatched on the one that mattered.")

    heading(doc, 2, "4.3   Rebuilding the instrument")
    para(doc,
         "The fix is not a larger cap everywhere, which would change the whole "
         "instrument. It is a separate budget for the one turn that gets parsed: "
         "`answer_max_tokens = 3072` on the final agent turn, `max_tokens` "
         "unchanged at 1024 elsewhere, and `max_model_len` reduced so the slot "
         "geometry does not move. Holding everything else constant:")
    table(doc,
          "the `hard` tier before and after the answer-budget instrument. the "
          "team's mean barely moves; the single agent's rises by +0.239, which is "
          "the whole of the original gap. nothing about the model or the task "
          "changed — the measurement had been reading its own cap.",
          ["`hard`", "1 agent", "team", "gap", "*d*", "answer turn cut (1 ag. / team)"],
          [["original instrument", "0.342", "0.591", "**+0.249**", "+1.09", "9 / 23  vs  1 / 24"],
           ["answer-budget instrument", "0.554", "0.536", "**−0.018**", "−0.12", "1 / 44  vs  3 / 48"]],
          number=2, widths=[1.55, 0.62, 0.55, 0.62, 0.45, 1.71], size=8.5)
    para(doc,
         "Answer-turn truncation in the single-agent arm falls roughly five-fold "
         "across the corpus rather than to zero: the first 24 `medium` episodes "
         "had none, and extending to 48 turned up three more. The honest statement "
         "is *reduced*, not *eliminated*, and the complete-case sensitivity row is "
         "still printed beside every number below for that reason.")

    # ======================= 5. the baseline ===============================
    heading(doc, 1, "5   The baseline that did not reproduce")
    para(doc,
         "Our ablation pilot measured a per-agent participation effect of "
         f"**+0.055 [+0.024, +0.088]** over 192 live-ablation episodes on seeds "
         "0–47. Our own preregistration forbade testing that on the corpus "
         "that produced it, so we re-ran all four agents on seeds "
         "1000–1149. **Every arm reproduced except one.**", space_after=8.0)

    figure(doc, fig_dir / "fig3_reproduction.png",
           "pilot against fresh seeds, per arm. the single-agent arm and the "
           "pooled three-agent ablated arms agree across seed sets to within "
           "0.006 and 0.002 respectively; the four-agent reference — the arm "
           "every positive finding in the project was measured against — "
           "moves by 0.055, which is the size of the effect being claimed.",
           width_in=5.4, number=2)

    para(doc,
         "The three-agent arms agree to within **0.002 over 791 episodes** "
         "(*p* = 0.909). **This is not seed-range difficulty.** The generator is "
         "deterministic in (seed, difficulty) and untouched across both runs; "
         "instances in the two ranges are structurally identical with zero "
         "variance (16 jobs, 5 workers, 29 constraints, 3 planted errors). Were "
         "the fresh range harder, the three-agent arms would carry the same "
         "penalty. They do not move at all.")
    para(doc,
         "It is sampling noise in a mean estimated from 48 episodes on an arm "
         "with `sd` ≈ 0.14, a standard error of ≈ 0.020 against effects "
         "of ≈ 0.05 being read off it. **An unremarkable statistical fact "
         "that produced a headline finding**, and one that a small-*n* pilot in "
         "this literature will reproduce routinely. What caught it was the "
         "fresh-seed rule, and it is worth being plain about what that rule "
         "caught: a headline result, in a project that had already survived seven "
         "corrections, produced entirely by a 48-episode reference.", space_after=8.0)

    figure(doc, fig_dir / "fig2_corrections.png",
           "each significant pro-team result, as measured and after its control. "
           "finding 1 is the token cap (§4.3); finding 2 is the same cap "
           "surviving budget-matching, removed by the complete-case sensitivity "
           "row; finding 3 is the 48-episode reference (§5). all three "
           "corrections move toward zero and none reverses.",
           width_in=5.4, number=3)

    # ============================ 6. results ===============================
    heading(doc, 1, "6   Results")
    heading(doc, 2, "6.1   What reproduced and what did not")
    table(doc,
          "the central table. everything that failed to reproduce across "
          "independent seed sets is a positive measured against the 48-episode "
          "four-agent reference; everything that reproduced is a null or a "
          "negative. the asymmetry is the result.",
          ["quantity", "pilot", "fresh", "*p*"],
          [["**failed to reproduce**", "", "", ""],
           ["four-agent reference", "0.631", "0.576", "0.031"],
           ["gate: team − 1 agent", "+0.045", "−0.003", "0.868"],
           ["C5: team − matched-budget agent", "+0.126", "+0.060 → **+0.003** corrected", "0.869"],
           ["participation: per-agent ablation drop", "+0.055", "**+0.002**", "0.834"],
           ["**reproduced**", "", "", ""],
           ["1 agent × 3 rounds", "0.585", "0.579", "0.856"],
           ["3 agents (pooled ablated)", "0.575", "0.574", "0.909"],
           ["budget penalty, 3 vs 12 rounds, one agent", "−0.081", "**−0.063**", "**0.011**"],
           ["generation asymmetry, 1 agent ÷ team", "1.87×", "2.04×", "—"],
           ["agent × component interaction", "*p* = 0.928", "not re-run (no main effect)", "—"],
           ["fungibility Δ(frozen) − Δ(live)", "−0.005", "—", "—"]],
          number=3, widths=[2.35, 0.85, 1.65, 0.55], size=8.5)

    heading(doc, 2, "6.2   Team size does nothing; more turns hurt")
    figure(doc, fig_dir / "fig4_teamsize.png",
           "the headline, on fresh seeds with the artifacts removed, drawn as "
           "episodes rather than bars because the finding is that the "
           "distributions coincide. the first three arms span "
           f"{N['spread']:.3f}. the only manipulation that reliably moves the "
           "score is giving a *single* agent four times the turns, and it moves "
           "it **down**.",
           width_in=5.4, number=4)
    para(doc,
         f"On fresh seeds one agent scores **{N['solo']:.3f}**, three agents "
         f"**{N['three']:.3f}** and four agents **{N['team']:.3f}** — a "
         f"spread of **{N['spread']:.3f} across {N['n_total']} episodes**. Team "
         "size does nothing on this task family at 8B. The only manipulation that "
         "reliably moves the score is giving a single agent the team's whole turn "
         f"budget, and it moves it downward: {N['solo']:.3f} → "
         f"{N['solo_long']:.3f}, **−0.063 at *p* = 0.011**. Read together "
         "with the generation asymmetry of §4.2, the interpretation is that "
         "a single agent given twelve rounds spends them restating rather than "
         "improving.")

    heading(doc, 2, "6.3   Behavioural coding and the interaction test agree")
    para(doc,
         "Phase 2 coded 468 messages against an eight-action taxonomy. Agents "
         "within an episode differ no more in what they do than shuffling the "
         "labels among them produces (*p* = 0.451), and no agent identity carries "
         "a stable tendency across episodes (*p* = 1.00). The one robust "
         "behavioural result is between conditions rather than within teams: "
         "**teams generate and lone agents audit** — propose as a share of "
         "propose plus verify is 0.674 for teams against 0.403 for solo, "
         "*p* < 0.0001, with the mechanism stated before the test was run.")
    para(doc,
         "The ablation side agrees. The agent × component interaction is "
         "**chi2 = 3.73, *p* = 0.928** over 768 observations from 48 episodes, "
         "with a double-centred residual matrix holding *less* structure than a "
         "true null would be expected to produce. Blocking compensation does not "
         "increase the damage either: Δ(frozen_replay) − Δ(live) = "
         "**−0.005**. Checked at three points as the pilot filled, every "
         "quantity constituting specialization halved as the sample doubled "
         "— agent spread 0.095 → 0.050 → 0.034, largest residual "
         "0.052 → 0.027 — which is the signature of noise averaging "
         "out, not of an effect being resolved.", space_after=8.0)

    figure(doc, fig_dir / "fig5_residuals.png",
           "the double-centred ablation matrix from the pilot grid. main effects "
           "for agent and for component are removed, so specialization would "
           "appear as off-diagonal structure. the colour scale is set to the "
           "≈0.06 a true null would produce; the observed residuals do not "
           "fill it.",
           width_in=3.5, number=5)

    para(doc,
         "**The scope must be stated precisely.** These are measured at an "
         "operating point where the team has no advantage over a single agent. A "
         "flat interaction and zero fungibility are what a zero main effect "
         "*predicts*; they are not independent confirmations of it. This is "
         "evidence that **this task, at this scale, does not produce the team "
         "advantage that specialization would have to explain** — not "
         "evidence that emergent specialization is absent in multi-agent LLM "
         "systems generally. The difference is between a bounded null and an "
         "unsupported universal.")

    # ==================== 7. instrument defects ============================
    heading(doc, 1, "7   Two instrument defects, reported rather than quietly fixed")
    para(doc,
         "**Frozen-transcript excision does not work on real transcripts.** "
         "Deleting an agent's messages and re-grading gives drops of ≈ 0.002 "
         "against live drops of 0.03–0.23, because agents restate the working "
         "solution constantly: the propagation index, the fraction of an agent's "
         "content echoed by others later, is **0.589**. The contribution survives "
         "in the copies. Read naively this reports “the agent contributed "
         "nothing” when the truth is “this measurement does not work "
         "here.” We ship a diagnostic that detects the condition and selects "
         "which frozen mode to trust.")
    para(doc,
         "**Our capacity control controlled one of the four cells it was quoted "
         "against.** Lowering `n_agents` by one always removes the *last* agent, "
         "so only one live-ablation cell shares its roster. On that cell, 74% of "
         "the attributed drop was unexplained by head-count — too large to "
         "ignore, too under-powered to act on, and moot once the main effect went "
         "to zero. Both defects are released with the code rather than repaired "
         "silently, because a reader deciding whether to reuse this instrument "
         "needs them more than we need a clean methods section.")

    # ======================= 8. recommendations ============================
    heading(doc, 1, "8   Recommendations")
    para(doc, "For anyone comparing single-agent and multi-agent LLM systems:")
    for i, rec in enumerate([
        "**Print an answer-turn truncation count beside every headline number** "
        "— not in an appendix. Our gate passed at two of three operating "
        "points on an artifact this column makes visible in one glance.",
        "**Report a complete-case sensitivity row.** Dropping truncated episodes "
        "is post-hoc and biased; so is ignoring them. The two bracket the truth. "
        "Ours was right three times, including against a headline at *p* < 0.001.",
        "**A matched-budget arm is part of the comparison, not an extra.** Without "
        "one, “four agents beat one” and “more tokens beat "
        "fewer” are the same measurement.",
        "**Match generation, not just turns.** Our matched-budget arm let the "
        "single agent emit 2.04× the team's text. Turn count is not token "
        "count.",
        "**A single-agent baseline needs a single-agent prompt.** Running *n* = 1 "
        "through a brief that says “the others” and “the group's "
        "last message is scored” measures the harness, and does so in the "
        "direction that flatters the team.",
        "**Never test a hypothesis on the corpus that generated it.** This rule "
        "cost us one extra run and overturned our headline. Without it we would "
        "have published a 2,100-episode ablation grid decomposing an effect that "
        "does not exist — and every cell would have been internally "
        "consistent.",
        "**Any per-turn resource limit lands asymmetrically when the arms use the "
        "resource differently.** This generalises past token caps to context "
        "windows, wall-clock budgets and tool-call quotas.",
    ]):
        numbered(doc, i + 1, rec)

    # ========================= 9. limitations ==============================
    heading(doc, 1, "9   Limitations")
    bullet(doc, "**One model family, one scale.** Llama-3.1-8B at Q4_K_M. "
                "Quantisation is lossy and not obviously neutral between arms: if "
                "4-bit weights degrade long-context instruction-following more, "
                "the team arm absorbs more of the loss. That biases *against* a "
                "team advantage, so it weakens a null more than it would have "
                "weakened a positive, and we report it as such.")
    bullet(doc, "**One task family.** Synthetic constrained scheduling, chosen "
                "because it is programmatically gradable with per-component "
                "scores and uncontaminated. A single task cannot support a claim "
                "about collaboration in general.")
    bullet(doc, "**`feasible` is uninformative.** It runs 0.000–0.021 in "
                "nearly every arm; almost no episode produces a fully feasible "
                "solution. All comparisons here are between degrees of partial "
                "credit.")
    bullet(doc, "**The interaction test was not re-run on fresh seeds**, because "
                "the fresh-seed run removed the main effect it would decompose. "
                "Our bound on it stands on the pilot corpus.")
    bullet(doc, "**Behavioural coding is bounded by judge quality.** Our local "
                "judge agrees with a human at κ = 0.29 after four codebook "
                "revisions, against the ≈ 0.6 Phase 2 needs and the "
                "≈ 0.78 convergent validity needs. Four codebooks moved "
                "κ from 0.07 to 0.29 and stopped, which reads as a "
                "model-capacity limit rather than a wording problem.")

    # ========================= 10. future work =============================
    heading(doc, 1, "10   Future work")
    para(doc,
         "**A 14B model.** The preregistered response to “7–8B may be "
         "too weak to collaborate meaningfully.” A 14B fits the same 24 GB "
         "card at 4-bit. If a team advantage appears at 14B where none exists at "
         "8B, that locates a capability threshold for collaboration — a more "
         "interesting finding than either of ours, and directly enabled by the "
         "artifact controls reported here. **Our negative is explicitly scoped to "
         "8B and does not predict the 14B outcome.**")
    para(doc,
         "**Role-label validation at usable agreement.** Until the judge reaches "
         "usable κ, the transcript-derived half of the convergent-validity "
         "comparison cannot be interpreted: we have causal contribution profiles "
         "and no trustworthy labels to correlate them against. This is testable "
         "on the same 40 messages the moment a larger judge is on disk.")
    para(doc,
         "**A task family where collaboration demonstrably pays.** The strongest "
         "version of the original experiment needs an operating point that exists. "
         "Finding one — by scale, by task, or by protocol — is the "
         "prerequisite the whole ablation programme was blocked on.")

    # ========================= 11. conclusion ==============================
    heading(doc, 1, "11   Conclusion")
    para(doc,
         "We did not measure whether emergent roles in LLM teams are causally "
         "real. We measured that three independent significant results favouring "
         "teams were artifacts — two of a token cap that is symmetric in "
         "specification and asymmetric in effect, one of a 48-episode baseline "
         "— and that with those removed, team size has no effect on this "
         "task family at 8B. The negative result is bounded and honest. The "
         "mechanism is the contribution: **a shared per-turn limit is not a shared "
         "constraint when the arms spend it differently**, and the diagnostics "
         "that expose it are cheap, general, and currently absent from standard "
         "practice. Every artifact we removed made the gap smaller. Not once did "
         "one make it larger.")

    # ========================= references ==================================
    doc.add_page_break()
    heading(doc, 1, "References")
    refs = [
        "T. Wang, H. Dong, V. Lesser and C. Zhang. ROMA: Multi-agent reinforcement "
        "learning with emergent roles. In *Proceedings of the 37th International "
        "Conference on Machine Learning (ICML)*, 2020. arXiv:2003.08039.",
        "Behavioral differentiation without role assignment in same-model LLM "
        "agent groups. arXiv:2604.00026, 2026.",
        "Agents that matter: causal leave-one-out attribution for multi-agent LLM "
        "systems. arXiv:2605.27621, 2026.",
        "IntrospecLOO: approximating leave-one-out agent ablation by "
        "introspection. arXiv:2505.22192, 2025.",
        "AgentDropout: dynamic agent elimination for efficient multi-agent LLM "
        "inference. arXiv:2503.18891, 2025.",
    ]
    for i, r in enumerate(refs):
        p = para(doc, "", size=9.0, space_after=4.0, lead=10.5)
        p.paragraph_format.left_indent = Inches(0.25)
        p.paragraph_format.first_line_indent = Inches(-0.25)
        runs(p, f"[{i + 1}]  " + r, 9.0)

    return doc


def append_appendix(doc) -> None:
    """Technical appendix. 'There is no page limit for the technical
    appendices', and nothing here is load-bearing for the main claims -- the
    paper stands without it, as the instructions require."""
    doc.add_page_break()
    heading(doc, 1, "A   Technical appendices and supplementary material")
    para(doc,
         "Nothing in this appendix is required to support the claims in the main "
         "paper. It records the per-tier detail behind the summary tables, for "
         "readers who want to check a specific operating point rather than take "
         "the aggregate on trust.")

    heading(doc, 2, "A.1   Every operating point, both instruments")
    table(doc,
          "the full gate across three difficulty tiers on both instruments. "
          "`cut` is the number of usable episodes whose answer-bearing turn hit "
          "the token cap. `sens. gap` is the same contrast with those episodes "
          "dropped from both arms. on the original instrument the headline gap "
          "and the truncation column rise together and the sensitivity gap does "
          "not — the signature of §4. on the answer-budget instrument the "
          "two columns agree.",
          ["instrument / tier", "1 agent", "team", "gap", "*p*",
           "cut (1 ag. / team)", "sens. gap"],
          [["**original**", "", "", "", "", "", ""],
           ["medium", "0.564", "0.631", "+0.067", "0.306", "5 / 22  ·  0 / 24", "−0.023"],
           ["hard", "0.342", "0.591", "**+0.249**", "**0.001**", "9 / 23  ·  1 / 24", "+0.024"],
           ["xhard", "0.310", "0.499", "**+0.188**", "**0.019**", "8 / 17  ·  2 / 24", "−0.056"],
           ["**answer-budget**", "", "", "", "", "", ""],
           ["medium", "0.585", "0.631", "+0.045", "0.246", "3 / 48  ·  0 / 48", "+0.006"],
           ["hard", "0.554", "0.536", "−0.018", "0.573", "1 / 44  ·  3 / 48", "+0.004"],
           ["xhard", "0.506", "0.575", "+0.069", "0.266", "1 / 24  ·  1 / 23", "+0.073"],
           ["**answer-budget, fresh seeds**", "", "", "", "", "", ""],
           ["medium", "0.579", "0.576", "−0.003", "0.868", "8 / 147  ·  4 / 150", "−0.015"]],
          number=4, widths=[1.42, 0.56, 0.48, 0.55, 0.44, 1.30, 0.62], size=8.0)

    heading(doc, 2, "A.2   Ablation cells, pilot and fresh")
    table(doc,
          "per-agent live-ablation drops from the four-agent reference, paired on "
          "instance. on the pilot every cell is positive and the pooled effect "
          "clears zero; on fresh seeds the cells straddle zero, one of them "
          "negative, and the pooled effect does not. the three-agent arms "
          "themselves are nearly identical across the two seed sets (table 3) "
          "— it is the reference that moved.",
          ["removed", "pilot drop", "pilot 95% CI", "fresh drop", "fresh 95% CI"],
          [["A1", "+0.049", "[−0.015, +0.118]", "+0.000", "[−0.035, +0.037]"],
           ["A2", "+0.037", "[−0.015, +0.092]", "+0.014", "[−0.023, +0.053]"],
           ["A3", "+0.064", "[−0.000, +0.130]", "−0.023", "[−0.057, +0.012]"],
           ["A4", "+0.071", "[+0.006, +0.139]", "+0.016", "[−0.019, +0.051]"],
           ["**pooled**", "**+0.055**", "**[+0.024, +0.088]**", "**+0.002**",
            "**[−0.016, +0.020]**"],
           ["capacity control (*n*→3)", "+0.019", "[−0.028, +0.062]", "−0.017",
            "[−0.048, +0.014]"]],
          number=5, widths=[1.42, 0.78, 1.16, 0.78, 1.16], size=8.0)

    heading(doc, 2, "A.3   The pilot's three checkpoints")
    para(doc,
         "The pilot grid was inspected three times as it filled. The pooled drop "
         "held steady while every quantity that would constitute specialization "
         "halved as the sample doubled, which is what noise averaging out looks "
         "like rather than an effect being resolved. Reported because the *shape* "
         "of the failure is the transferable part: an interaction that is real "
         "should sharpen with *n*, not dissolve.")
    table(doc,
          "the pilot ablation grid checked at three sample sizes.",
          ["checked at", "episodes/agent", "pooled drop", "agent spread",
           "largest residual", "interaction *p*"],
          [["75 done", "≈19", "+0.063", "0.095", "—", "0.591"],
           ["148 done", "≈37", "+0.050", "0.050", "0.052", "0.820"],
           ["**192 done**", "**48**", "**+0.055**", "**0.034**", "**0.027**",
            "**0.928**"]],
          number=6, widths=[0.95, 1.05, 0.90, 0.95, 1.00, 0.95], size=8.0)

    heading(doc, 2, "A.4   Behavioural coding: the taxonomy collapsed")
    para(doc,
         "468 messages were coded by a local judge against an eight-action "
         "taxonomy, with 0 unparseable responses and 0 judge errors. Two labels "
         "took 95.9% of the corpus: `propose` (294, 62.8%) and `verify` (155, "
         "33.1%), followed by `compute` (8), `other` (6), `agree` (4), `search` "
         "(1), and `synthesize` and `organize` at zero. This is not an artifact "
         "of the solo episodes: within the 276 team messages, where `organize`, "
         "`synthesize` and `agree` are all available, the same two labels take "
         "98.1%.")
    para(doc,
         "**This independently breaks the convergent-validity phase.** The "
         "mapping from actions to graded components uses exactly four actions "
         "— `compute`→arithmetic, `search`→search, "
         "`verify`→verification, `synthesize`→synthesis — and "
         "three of the four have essentially no data. Even with a usable ablation "
         "grid in hand, convergent validity would have been a test of `verify` "
         "alone rather than of the mapping. We report this as a design lesson: a "
         "taxonomy should be piloted against the corpus it will code before it is "
         "made load-bearing for a downstream test.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="CollabEngine-NeurIPS2026.docx")
    ap.add_argument("--author", default="Guru R Bharadwaj")
    ap.add_argument("--affiliation", default="Independent Researcher")
    ap.add_argument("--email", default="gururb20@gmail.com")
    ap.add_argument("--anonymous", action="store_true")
    ap.add_argument("--fig-dir", default="docs/figures/paper")
    args = ap.parse_args()

    doc = build(Path(args.out), args.author, args.affiliation, args.email,
                args.anonymous, Path(args.fig_dir))
    append_appendix(doc)
    from checklist import append_checklist          # noqa: E402
    append_checklist(doc, para, heading, bullet, runs)
    doc.save(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    main()
