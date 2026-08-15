"""The NeurIPS 2026 paper checklist, answered for this paper.

Kept separate from `build_paper.py` because it is a fixed instrument: the
questions and guidelines are the conference's and must not be reworded, while
the answers and justifications are ours. Mixing the two in one file invites
editing the questions by accident, which the instructions forbid.

The instruction block that precedes the checklist in the template is deleted
here, as the template directs; the section heading, the subsection headings,
the questions, the answers and the guidelines are kept.
"""

from __future__ import annotations

from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt

# (title, question, answer, justification, [guidelines])
ITEMS: list[tuple[str, str, str, str, list[str]]] = [
    ("1. Claims",
     "Do the main claims made in the abstract and introduction accurately "
     "reflect the paper's contributions and scope?",
     "Yes",
     "The abstract states the artifact mechanism, the three corrected findings "
     "and the bounded negative result, and it names the scope limit explicitly "
     "(one task family, 8B, Q4_K_M). Section 6.3 states in the body that a flat "
     "interaction is what a zero main effect predicts rather than an "
     "independent confirmation of it, and Section 1 states that the intended "
     "experiment was not run.",
     ["The answer [N/A] means that the abstract and introduction do not include "
      "the claims made in the paper.",
      "The abstract and/or introduction should clearly state the claims made, "
      "including the contributions made in the paper and important assumptions "
      "and limitations. A [No] or [N/A] answer to this question will not be "
      "perceived well by the reviewers.",
      "The claims made should match theoretical and experimental results, and "
      "reflect how much the results can be expected to generalize to other "
      "settings.",
      "It is fine to include aspirational goals as motivation as long as it is "
      "clear that these goals are not attained by the paper."]),

    ("2. Limitations",
     "Does the paper discuss the limitations of the work performed by the "
     "authors?",
     "Yes",
     "Section 9 is a dedicated limitations section covering model family and "
     "scale, the single task family, an uninformative third metric, the "
     "interaction test not being re-run on fresh seeds, and the judge-quality "
     "ceiling on behavioural coding. Section 7 additionally reports two defects "
     "in our own instrument rather than repairing them silently.",
     ["The answer [N/A] means that the paper has no limitation while the answer "
      "[No] means that the paper has limitations, but those are not discussed "
      "in the paper.",
      "The authors are encouraged to create a separate “Limitations” "
      "section in their paper.",
      "The paper should point out any strong assumptions and how robust the "
      "results are to violations of these assumptions.",
      "The authors should reflect on the scope of the claims made, e.g., if the "
      "approach was only tested on a few datasets or with a few runs.",
      "The authors should reflect on the factors that influence the performance "
      "of the approach.",
      "The authors should discuss the computational efficiency of the proposed "
      "algorithms and how they scale with dataset size.",
      "If applicable, the authors should discuss possible limitations of their "
      "approach to address problems of privacy and fairness.",
      "While the authors might fear that complete honesty about limitations "
      "might be used by reviewers as grounds for rejection, a worse outcome "
      "might be that reviewers discover limitations that aren't acknowledged in "
      "the paper. Reviewers will be specifically instructed to not penalize "
      "honesty concerning limitations."]),

    ("3. Theory assumptions and proofs",
     "For each theoretical result, does the paper provide the full set of "
     "assumptions and a complete (and correct) proof?",
     "N/A",
     "The paper contains no theorems or formal proofs. Its claims are empirical "
     "and are supported by permutation tests, bootstrap intervals and a "
     "mixed-effects joint Wald test, all specified in Section 3 and reproducible "
     "from the released corpus.",
     ["The answer [N/A] means that the paper does not include theoretical "
      "results.",
      "All the theorems, formulas, and proofs in the paper should be numbered "
      "and cross-referenced.",
      "All assumptions should be clearly stated or referenced in the statement "
      "of any theorems.",
      "The proofs can either appear in the main paper or the supplemental "
      "material, but if they appear in the supplemental material, the authors "
      "are encouraged to provide a short proof sketch to provide intuition.",
      "Inversely, any informal proof provided in the core of the paper should be "
      "complemented by formal proofs provided in appendix or supplemental "
      "material.",
      "Theorems and Lemmas that the proof relies upon should be properly "
      "referenced."]),

    ("4. Experimental result reproducibility",
     "Does the paper fully disclose all the information needed to reproduce the "
     "main experimental results of the paper to the extent that it affects the "
     "main claims and/or conclusions of the paper (regardless of whether the "
     "code and data are provided or not)?",
     "Yes",
     "Section 3 gives the model, quantisation, serving configuration, slot "
     "geometry, task generator, difficulty tiers, metrics and statistical "
     "procedures. Instance generation is deterministic in (seed, difficulty) and "
     "the exact seed ranges are stated in Section 5. The token budgets that "
     "constitute the corrected instrument are given numerically in Section 4.3. "
     "Every figure and table in the paper is regenerated from the released "
     "transcripts by a script in the repository.",
     ["The answer [N/A] means that the paper does not include experiments.",
      "If the paper includes experiments, a [No] answer to this question will "
      "not be perceived well by the reviewers: Making the paper reproducible is "
      "important, regardless of whether the code and data are provided or not.",
      "If the contribution is a dataset and/or model, the authors should "
      "describe the steps taken to make their results reproducible or "
      "verifiable.",
      "Depending on the contribution, reproducibility can be accomplished in "
      "various ways. In general, releasing code and data is often one good way "
      "to accomplish this, but reproducibility can also be provided via detailed "
      "instructions for how to replicate the results.",
      "While NeurIPS does not require releasing code, the conference does "
      "require all submissions to provide some reasonable avenue for "
      "reproducibility, which may depend on the nature of the contribution."]),

    ("5. Open access to data and code",
     "Does the paper provide open access to the data and code, with sufficient "
     "instructions to faithfully reproduce the main experimental results, as "
     "described in supplemental material?",
     "Yes",
     "The full framework, the configuration files for every operating point, the "
     "episode transcripts for every condition, and the analysis and figure "
     "scripts are released together. A mock backend runs the entire pipeline "
     "end to end with no GPU, so the instrument can be validated before any "
     "compute is spent. A single command regenerates every number quoted in the "
     "paper from the transcripts.",
     ["The answer [N/A] means that paper does not include experiments requiring "
      "code.",
      "Please see the NeurIPS code and data submission guidelines "
      "(https://neurips.cc/public/guides/CodeSubmissionPolicy) for more details.",
      "While we encourage the release of code and data, we understand that this "
      "might not be possible, so [No] is an acceptable answer.",
      "The instructions should contain the exact command and environment needed "
      "to run to reproduce the results.",
      "The authors should provide instructions on data access and preparation.",
      "The authors should provide scripts to reproduce all experimental results "
      "for the new proposed method and baselines.",
      "At submission time, to preserve anonymity, the authors should release "
      "anonymized versions (if applicable)."]),

    ("6. Experimental setting/details",
     "Does the paper specify all the training and test details (e.g., data "
     "splits, hyperparameters, how they were chosen, type of optimizer) "
     "necessary to understand the results?",
     "Yes",
     "No training is performed; all agents are one frozen quantised checkpoint. "
     "The relevant settings are decoding and orchestration parameters "
     "— temperature, nucleus threshold, per-turn and answer-turn token "
     "budgets, agent count, round count, turn-order randomisation and "
     "symmetry-breaking mode — and these are given in Section 3 and "
     "Section 4.3, with the full resolved configuration released alongside each "
     "run directory.",
     ["The answer [N/A] means that the paper does not include experiments.",
      "The experimental setting should be presented in the core of the paper to "
      "a level of detail that is necessary to appreciate the results and make "
      "sense of them.",
      "The full details can be provided either with the code, in appendix, or as "
      "supplemental material."]),

    ("7. Experiment statistical significance",
     "Does the paper report error bars suitably and correctly defined or other "
     "appropriate information about the statistical significance of the "
     "experiments?",
     "Yes",
     "Every headline contrast is reported with a 95% percentile bootstrap "
     "interval over 10,000 draws and a two-sided permutation p-value over 20,000 "
     "draws; both are stated in Section 3 and shown as intervals in Figures 2 "
     "and 3. Ablation contrasts are paired on instance, because the same "
     "instance is played by the intact team and by every ablated variant, and "
     "treating those rows as independent would shrink the standard errors. The "
     "interaction test uses episode as a random effect and is joint across all "
     "interaction coefficients rather than scanning for the smallest p.",
     ["The answer [N/A] means that the paper does not include experiments.",
      "The authors should answer [Yes] if the results are accompanied by error "
      "bars, confidence intervals, or statistical significance tests, at least "
      "for the experiments that support the main claims of the paper.",
      "The factors of variability that the error bars are capturing should be "
      "clearly stated.",
      "The method for calculating the error bars should be explained (closed "
      "form formula, call to a library function, bootstrap, etc.)",
      "The assumptions made should be given (e.g., Normally distributed errors).",
      "It should be clear whether the error bar is the standard deviation or the "
      "standard error of the mean.",
      "It is OK to report 1-sigma error bars, but one should state it.",
      "For asymmetric distributions, the authors should be careful not to show "
      "in tables or figures symmetric error bars that would yield results that "
      "are out of range.",
      "If error bars are reported in tables or plots, the authors should explain "
      "in the text how they were calculated and reference the corresponding "
      "figures or tables in the text."]),

    ("8. Experiments compute resources",
     "For each experiment, does the paper provide sufficient information on the "
     "computer resources (type of compute workers, memory, time of execution) "
     "needed to reproduce the experiments?",
     "Yes",
     "All experiments run on a single 24 GB NVIDIA RTX 4500 Ada workstation GPU "
     "serving one 4-bit GGUF checkpoint to every agent, as stated in Section 3. "
     "The paper also discloses compute spent on work that does not appear in the "
     "results: three corpora lost to preflight-detectable conditions, a "
     "2,100-episode ablation grid that was halted at its preregistered branch "
     "rule, and a full pilot grid superseded by the fresh-seed re-run of "
     "Section 5.",
     ["The answer [N/A] means that the paper does not include experiments.",
      "The paper should indicate the type of compute workers CPU or GPU, "
      "internal cluster, or cloud provider, including relevant memory and "
      "storage.",
      "The paper should provide the amount of compute required for each of the "
      "individual experimental runs as well as estimate the total compute.",
      "The paper should disclose whether the full research project required more "
      "compute than the experiments reported in the paper (e.g., preliminary or "
      "failed experiments that didn't make it into the paper)."]),

    ("9. Code of ethics",
     "Does the research conducted in the paper conform, in every respect, with "
     "the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?",
     "Yes",
     "The work uses a synthetic, programmatically generated task family and an "
     "openly available model checkpoint. It involves no human subjects, no "
     "personal data, no scraped corpora and no deployment. The one human-rater "
     "comparison reported is the author's own coding of forty of the "
     "system's own generated messages.",
     ["The answer [N/A] means that the authors have not reviewed the NeurIPS "
      "Code of Ethics.",
      "If the authors answer [No], they should explain the special circumstances "
      "that require a deviation from the Code of Ethics.",
      "The authors should make sure to preserve anonymity (e.g., if there is a "
      "special consideration due to laws or regulations in their jurisdiction)."]),

    ("10. Broader impacts",
     "Does the paper discuss both potential positive societal impacts and "
     "negative societal impacts of the work performed?",
     "Yes",
     "The positive impact is direct and is the paper's purpose: the "
     "diagnostics in Section 8 reduce the rate at which multi-agent LLM systems "
     "are reported as beneficial on the basis of measurement artifacts, which "
     "bears on procurement and deployment decisions that cite such comparisons. "
     "The plausible negative impact is that a bounded negative result is quoted "
     "beyond its scope as evidence that multi-agent systems do not work; "
     "Section 6.3 states the scope limit explicitly in anticipation of exactly "
     "that misreading. The work releases no model, no dataset of human origin "
     "and no capability uplift.",
     ["The answer [N/A] means that there is no societal impact of the work "
      "performed.",
      "If the authors answer [N/A] or [No], they should explain why their work "
      "has no societal impact or why the paper does not address societal impact.",
      "Examples of negative societal impacts include potential malicious or "
      "unintended uses, fairness considerations, privacy considerations, and "
      "security considerations.",
      "The conference expects that many papers will be foundational research and "
      "not tied to particular applications, let alone deployments.",
      "The authors should consider possible harms that could arise when the "
      "technology is being used as intended and functioning correctly, harms "
      "that could arise when the technology is being used as intended but gives "
      "incorrect results, and harms following from misuse of the technology.",
      "If there are negative societal impacts, the authors could also discuss "
      "possible mitigation strategies."]),

    ("11. Safeguards",
     "Does the paper describe safeguards that have been put in place for "
     "responsible release of data or models that have a high risk for misuse "
     "(e.g., pre-trained language models, image generators, or scraped "
     "datasets)?",
     "N/A",
     "The paper releases no model weights and no scraped data. The released "
     "artifacts are an experimental harness, configuration files, and "
     "transcripts of a synthetic scheduling task generated by an already-public "
     "checkpoint. None of these carries a misuse risk requiring gated release.",
     ["The answer [N/A] means that the paper poses no such risks.",
      "Released models that have a high risk for misuse or dual-use should be "
      "released with necessary safeguards to allow for controlled use of the "
      "model, for example by requiring that users adhere to usage guidelines or "
      "restrictions to access the model or implementing safety filters.",
      "Datasets that have been scraped from the Internet could pose safety "
      "risks. The authors should describe how they avoided releasing unsafe "
      "images.",
      "We recognize that providing effective safeguards is challenging, and many "
      "papers do not require this, but we encourage authors to take this into "
      "account and make a best faith effort."]),

    ("12. Licenses for existing assets",
     "Are the creators or original owners of assets (e.g., code, data, models), "
     "used in the paper, properly credited and are the license and terms of use "
     "explicitly mentioned and properly respected?",
     "Yes",
     "The model checkpoint is Meta-Llama-3.1-8B-Instruct, used under the Llama "
     "3.1 Community License and named with its exact quantisation (Q4_K_M) in "
     "Section 3. Serving uses llama.cpp, which is MIT licensed and vendored with "
     "its license intact. Prior work whose findings motivate the design is cited "
     "in Section 2. The task instances are generated by our own code and are not "
     "derived from any existing dataset.",
     ["The answer [N/A] means that the paper does not use existing assets.",
      "The authors should cite the original paper that produced the code package "
      "or dataset.",
      "The authors should state which version of the asset is used and, if "
      "possible, include a URL.",
      "The name of the license (e.g., CC-BY 4.0) should be included for each "
      "asset.",
      "For scraped data from a particular source, the copyright and terms of "
      "service of that source should be provided.",
      "If assets are released, the license, copyright information, and terms of "
      "use in the package should be provided.",
      "For existing datasets that are re-packaged, both the original license and "
      "the license of the derived asset should be provided.",
      "If this information is not available online, the authors are encouraged "
      "to reach out to the asset's creators."]),

    ("13. New assets",
     "Are new assets introduced in the paper well documented and is the "
     "documentation provided alongside the assets?",
     "Yes",
     "The released framework carries a research log recording every measurement, "
     "pivot and failure with its cost; preregistration documents with dated "
     "amendments; serving arithmetic for the quantised instrument; and per-module "
     "documentation of the analysis path. The two known instrument defects of "
     "Section 7 are documented in place rather than omitted, so that a reader "
     "reusing the harness meets them in the documentation rather than in their "
     "results.",
     ["The answer [N/A] means that the paper does not release new assets.",
      "Researchers should communicate the details of the dataset/code/model as "
      "part of their submissions via structured templates. This includes details "
      "about training, license, limitations, etc.",
      "The paper should discuss whether and how consent was obtained from people "
      "whose asset is used.",
      "At submission time, remember to anonymize your assets (if applicable)."]),

    ("14. Crowdsourcing and research with human subjects",
     "For crowdsourcing experiments and research with human subjects, does the "
     "paper include the full text of instructions given to participants and "
     "screenshots, if applicable, as well as details about compensation (if "
     "any)?",
     "N/A",
     "The paper involves no crowdsourcing and no human subjects. The single "
     "human-coded comparison is the author coding forty machine-generated "
     "messages against the project's own published codebook, which is released "
     "with the code.",
     ["The answer [N/A] means that the paper does not involve crowdsourcing nor "
      "research with human subjects.",
      "Including this information in the supplemental material is fine, but if "
      "the main contribution of the paper involves human subjects, then as much "
      "detail as possible should be included in the main paper.",
      "According to the NeurIPS Code of Ethics, workers involved in data "
      "collection, curation, or other labor should be paid at least the minimum "
      "wage in the country of the data collector."]),

    ("15. Institutional review board (IRB) approvals or equivalent for research "
     "with human subjects",
     "Does the paper describe potential risks incurred by study participants, "
     "whether such risks were disclosed to the subjects, and whether "
     "Institutional Review Board (IRB) approvals (or an equivalent "
     "approval/review based on the requirements of your country or institution) "
     "were obtained?",
     "N/A",
     "The paper involves no human subjects and no participants, so no IRB "
     "approval or equivalent review was required.",
     ["The answer [N/A] means that the paper does not involve crowdsourcing nor "
      "research with human subjects.",
      "Depending on the country in which research is conducted, IRB approval (or "
      "equivalent) may be required for any human subjects research. If you "
      "obtained IRB approval, you should clearly state this in the paper.",
      "We recognize that the procedures for this may vary significantly between "
      "institutions and locations, and we expect authors to adhere to the "
      "NeurIPS Code of Ethics and the guidelines for their institution.",
      "For initial submissions, do not include any information that would break "
      "anonymity (if applicable), such as the institution conducting the "
      "review."]),

    ("16. Declaration of LLM usage",
     "Does the paper describe the usage of LLMs if it is an important, original, "
     "or non-standard component of the core methods in this research? Note that "
     "if the LLM is used only for writing, editing, or formatting purposes and "
     "does not impact the core methodology, scientific rigor, or originality of "
     "the research, declaration is not required.",
     "Yes",
     "LLMs are the object of study rather than a tool applied to it, and their "
     "role is declared throughout. One quantised open-weights checkpoint "
     "produces every agent turn in every condition, which is the identical-"
     "weights control the design rests on. A separate local LLM judge performs "
     "behavioural coding of finished transcripts only; Section 9 reports that "
     "judge's agreement with a human rater and treats it as a bound on the "
     "coding results. No LLM was used to generate agent turns for a condition it "
     "also judged.",
     ["The answer [N/A] means that the core method development in this research "
      "does not involve LLMs as any important, original, or non-standard "
      "components.",
      "Please refer to our LLM policy in the NeurIPS handbook for what should or "
      "should not be described."]),
]


def append_checklist(doc, para, heading, bullet, runs) -> None:
    """Append the answered checklist. Callers pass the builder's own helpers so
    the checklist inherits the document's metrics exactly."""
    doc.add_page_break()

    p = para(doc, "", size=12.0, space_before=0, space_after=8.0, lead=14)
    r = p.add_run("NeurIPS Paper Checklist")
    r.font.name = "Times New Roman"
    r.font.size = Pt(12)
    r.font.bold = True

    for title, question, answer, justification, guidelines in ITEMS:
        h = para(doc, "", size=10.0, align=WD_ALIGN_PARAGRAPH.LEFT,
                 space_before=9.0, space_after=3.0, lead=12)
        h.paragraph_format.left_indent = Inches(0.28)
        h.paragraph_format.first_line_indent = Inches(-0.28)
        hr = h.add_run(title)
        hr.font.name = "Times New Roman"
        hr.font.size = Pt(10)
        hr.font.bold = True

        for label, text in (("Question: ", question),
                            ("Answer: ", f"**[{answer}]**"),
                            ("Justification: ", justification)):
            q = para(doc, "", size=10.0, space_after=3.0)
            q.paragraph_format.left_indent = Inches(0.28)
            runs(q, label + text, 10.0)

        g = para(doc, "", size=10.0, space_after=2.0)
        g.paragraph_format.left_indent = Inches(0.28)
        runs(g, "Guidelines:", 10.0)
        for item in guidelines:
            b = para(doc, "", size=9.0, space_after=1.5, lead=10.5)
            b.paragraph_format.left_indent = Inches(0.70)
            b.paragraph_format.first_line_indent = Inches(-0.16)
            runs(b, "•  " + item, 9.0)
