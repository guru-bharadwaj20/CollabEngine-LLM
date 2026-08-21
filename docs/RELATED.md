# Related work

Raw material for two things: the paper's Related Work section (Tier 1.9), and the
Tier 2.1 audit of a public multi-agent system for the same artifact. Every entry
below was verified against the arXiv abstract page or the publisher record on
2026-08-20 — title, authors, year and identifier are checked, not recalled.
Anything that could not be verified is listed at the bottom under **Dropped**
rather than included.

Each entry is three things: what it claims, and how it touches this paper's
argument — that a per-turn generation cap is symmetric in specification and
asymmetric in effect, because a solo agent must emit the whole answer in its
final turn while a team commits one its transcript already holds.

**Read the scooping check first.** There is a 2026 paper that reaches a
neighbouring conclusion by a different route, and the paper cannot be written as
if it does not exist.

---

## 1. Multi-agent LLM frameworks

The point of contact is the last column. Every one of these reports a gain over
a single-agent baseline; the question is what the baseline was allowed to spend.

| System | Single-agent baseline used | Matched on |
|---|---|---|
| AutoGen | vanilla GPT-4, ChatGPT+Code Interpreter, LangChain ReAct, DPR+GPT-3.5, ReAct | **Nothing.** Task and model only; extra LLM calls are reported as a feature |
| MetaGPT | CodeX, GPT-4, PaLM, CodeGen, AlphaCode single-call pass@1 | **Nothing.** Token cost reported *post hoc* and only against ChatDev, not against the single-call baselines |
| CAMEL | `gpt-3.5-turbo` **single-shot** | **Nothing.** Multi-agent arm runs to a 40-message limit; solution is a *summary* of that conversation |
| ChatDev | GPT-Engineer (single agent), MetaGPT | **Hyperparameters, not budget.** Token counts reported (7.2k vs 22.9k) and explicitly accepted as a cost |
| AgentVerse | CoT agent; `Solo` = AgentVerse with one agent | **Turns/modules, not tokens.** `Solo` keeps the recruitment/execution/evaluation modules; no token or call normalisation |
| Magentic-One | Prior systems on GAIA / AssistantBench / WebArena | **Not a solo-vs-team comparison.** Ships AutoGenBench for repetition and isolation control |

- **AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation**
  (Wu, Bansal, Zhang, Wu, Li, Zhu, Jiang, Zhang, Zhang, Liu, Awadallah, White,
  Burger, Wang; arXiv:2308.08155, 2023) — a conversable-agent framework whose
  multi-agent configurations beat single-call and ReAct baselines across math,
  QA, ALFWorld and decision-making. **Baseline matched on: nothing beyond task
  and model.** The paper reports that ~19.4% of Natural Questions items trigger
  an extra "Update Context" LLM call, i.e. the winning arm is explicitly allowed
  more generation than the baseline, and no result is re-reported at equal
  budget. This is the standard practice recommendation 3 of the paper is
  addressed to.

- **MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework**
  (Hong, Zhuge, Chen, Zheng, Cheng, Zhang, Wang, Wang, Yau, Lin, Zhou, Ran, Xiao,
  Wu, Schmidhuber; ICLR 2024, arXiv:2308.00352) — encodes SOPs into prompt
  sequences and reports 85.9% / 87.7% on HumanEval / MBPP against single-call
  baselines. **Baseline matched on: nothing.** Token accounting exists (24,613 or
  31,255 tokens per SoftwareDev task against ChatDev's 19,292) but is used to
  argue efficiency *between two multi-agent systems*, never to equalise against
  the single-call arm. Also the clearest example of the README's closing point:
  the role scaffolding is in the prompt templates, so any "emergent" structure
  observed in MetaGPT was planted.

- **CAMEL: Communicative Agents for "Mind" Exploration of Large Language Model
  Society** (Li, Hammoud, Itani, Khizbullin, Ghanem; NeurIPS 2023,
  arXiv:2303.17760) — inception-prompted role-playing between two agents;
  human raters preferred CAMEL over `gpt-3.5-turbo` single-shot 76.3% of the
  time, GPT-4 raters 73.0%. **Baseline matched on: nothing.** This is the
  sharpest instance of the cap asymmetry in the framework literature: the paper
  states the conversation runs to "a maximum limit of 40 messages" and the
  evaluated artifact is a *summarised* CAMEL solution, judged against one
  single-shot generation. The team's answer is assembled from 40 turns' worth of
  budget; the baseline's answer is one turn's worth. That is §3.1 of the paper,
  in a published system, at a 40:1 ratio.

- **ChatDev: Communicative Agents for Software Development** (Qian, Liu, Liu,
  Chen, Dang, Li, Yang, Chen, Su, Cong, Xu, Li, Liu, Sun; ACL 2024,
  arXiv:2307.07924) — a chat chain across design, coding and testing that beats
  GPT-Engineer, "a fundamental single-agent approach." **Baseline matched on:
  hyperparameters and temperature, explicitly not budget.** The paper reports
  7,182 tokens for GPT-Engineer against 22,949 for ChatDev — a 3.2× generation
  advantage held by the winning arm — and states that "the multi-agent paradigm,
  despite being slower and consuming more tokens than the single-agent method,
  yields a greater number of code files and a larger codebase." The honesty is
  real; the comparison is still "more tokens beat fewer" and "four agents beat
  one" measured as one number.

- **AgentVerse: Facilitating Multi-Agent Collaboration and Exploring Emergent
  Behaviors** (Chen, Su, Zuo, Yang, Yuan, Chan, Yu, Lu, Hung, Qian, Qin, Cong,
  Xie, Liu, Sun, Zhou; arXiv:2308.10848, 2023) — a
  recruit/decide/execute/evaluate loop, claimed to be "a greater-than-the-sum-of-
  its-parts system." **Baseline matched on: modules and turns, not tokens.** Its
  `Solo` condition is the best-designed solo baseline in this cluster — it keeps
  the same scaffolding modules, so it isolates group size rather than harness —
  and it is still not budget-matched, which is exactly the gap between this
  paper's C4/C5 arms and its gate arm. AgentVerse is the closest published
  analogue to the C5 design and the most useful contrast to draw.

- **Magentic-One: A Generalist Multi-Agent System for Solving Complex Tasks**
  (Fourney, Bansal, Mozannar, Tan, Salinas, Zhu, Niedtner, Proebsting, Bassman,
  Gerrits, Alber, Chang, Loynd, West, Dibia, Awadallah, Kamar, Hosn, Amershi;
  arXiv:2411.04468, 2024) — an orchestrator plus specialised agents, reported as
  "statistically competitive" on GAIA, AssistantBench and WebArena. **Not a
  solo-vs-team comparison at all**, which is worth saying: it ships AutoGenBench
  with "built-in controls for repetition and isolation," so the instrument
  discipline exists in this literature — it is just aimed at variance, not at
  budget parity.

- **Why Do Multi-Agent LLM Systems Fail?** (Cemri, Pan, Yang, Agrawal, Chopra,
  Tiwari, Keutzer, Parameswaran, Klein, Ramchandran, Zaharia, Gonzalez, Stoica;
  NeurIPS 2025 Datasets and Benchmarks Track, arXiv:2503.13657) — a 14-mode
  failure taxonomy (MAST) built from grounded-theory analysis of 150 traces at
  κ = 0.88, plus MAST-Data, 1,642 annotated traces from 7 frameworks. **The most
  important entry in this cluster for Tier 2.1.** Its taxonomy is behavioural —
  system design, inter-agent misalignment, task verification — and contains no
  category for *the harness truncated the answer*. That is the gap this paper
  fills, and MAST-Data is the corpus in which to look for it (see §6).

---

## 2. Debate, self-consistency and reflection

This is where a controlled solo-vs-multi comparison is most likely to exist, and
where the re-evaluations have already started arriving.

- **Improving Factuality and Reasoning in Language Models through Multiagent
  Debate** (Du, Li, Torralba, Tenenbaum, Mordatch; arXiv:2305.14325, 2023) —
  multiple model instances propose and debate over several rounds; reported gains
  in arithmetic, strategic reasoning and factuality. The canonical claim this
  cluster's critics test. Its debate arm spends *n_agents × n_rounds* generations
  against a single-generation baseline; the paper's recommendation 4 ("match
  generation, not just turns") applies to it directly.

- **Self-Consistency Improves Chain of Thought Reasoning in Language Models**
  (Wang, Wei, Schuurmans, Le, Chi, Narang, Chowdhery, Zhou; ICLR 2023,
  arXiv:2203.11171) — sample many reasoning paths, marginalise, take the modal
  answer; +17.9 on GSM8K. The correct competing explanation for most reported
  multi-agent gains: it obtains them from extra sampled generations with no
  agents, no roles and no communication. If a team beats a solo agent but not
  self-consistency at the same generation count, the team contributed nothing
  that sampling did not.

- **Reflexion: Language Agents with Verbal Reinforcement Learning** (Shinn,
  Cassano, Berman, Gopinath, Narasimhan, Yao; arXiv:2303.11366, 2023) —
  self-reflection stored in an episodic memory buffer; 91% pass@1 on HumanEval.
  A single-agent method that consumes multi-agent-scale generation, which makes
  it the natural matched-budget control that the framework papers above omit.

- **Large Language Models Cannot Self-Correct Reasoning Yet** (Huang, Chen,
  Mishra, Zheng, Yu, Song, Zhou; ICLR 2024, arXiv:2310.01798) — intrinsic
  self-correction without external feedback does not help and sometimes hurts.
  Structurally the same result as this paper's within-agent budget penalty: more
  turns given to one agent moved `fraction` **down** by 0.063 (*p* = 0.012). Two
  independent findings that additional self-directed turns are not free.

- **Should we be going MAD? A Look at Multi-Agent Debate Strategies for LLMs**
  (Smit, Duckworth, Grinsztajn, Barrett, Pretorius; arXiv:2311.17371, 2023) —
  debate "does not reliably outperform other proposed prompting strategies, such
  as self-consistency and ensembling," though tuned agreement levels can recover
  it. The first of the re-evaluations, and it reports cost alongside accuracy.

- **Stop Overvaluing Multi-Agent Debate — We Must Rethink Evaluation and Embrace
  Model Heterogeneity** (Zhang, Cui, Chen, Wang, Zhang, Wang, Wu, Hu;
  arXiv:2502.08788, 2025) — 5 MAD methods × 9 benchmarks × 4 models; MAD "often
  fail to outperform simple single-agent baselines such as Chain-of-Thought and
  Self-Consistency, **even when consuming significantly more inference-time
  computation**," and heterogeneity is the fix. The closest position-paper
  ancestor of this paper's argument, arrived at from evaluation coverage rather
  than from an instrument defect. Its diagnosis is *weak baselines*; this paper's
  is *a baseline the instrument silently penalises*, which is a different and
  narrower failure.

- **When and Why Does Multi-Agent Debate Fail and Does It Really Underperform?**
  (Chen, Niu, Cheng, Han, Sugiyama; arXiv:2510.20963, 2025, rev. 2026) — argues
  competitive MAD degenerates into cheap talk and consensus-seeking discards
  useful disagreement, and proposes ColMAD, which it reports outperforms
  single-agent scaling at comparable token usage. **The counter-example to hold
  the paper's claim against.** It is the one result in this cluster claiming a
  multi-agent win that survives a token comparison, so the paper's scope
  statement ("this task family, at 8B") must not be written in a way that ColMAD
  contradicts.

- **The Cost of Consensus: Isolated Self-Correction Prevails Over Unguided
  Homogeneous Multi-Agent Debate** (Bertalanič, Fortuna; ACM Conference on AI and
  Agentic Systems, arXiv:2605.00914, 2026) — 10-agent teams across three models;
  "debate consumes 2.1–3.4× more tokens (up to 28,631 tokens per problem) than
  self-correction for equal or lower accuracy," with isolated self-correction
  winning across all 7–8B configurations. **Same model scale as this paper, same
  direction, independent instrument.** Their 2.1–3.4× token ratio is the mirror
  image of this paper's measured 2.04× solo-to-team generation ratio — they find
  the team spending more, this paper finds the solo agent forced to spend more
  in one turn. Both are the same underlying fact: turn count is not token count.

- **Statistical Scouting Finds Debate-Safe but Not Debate-Useful Cases: A
  Matched-Ceiling Study of Open-Weight LLM Reasoning Protocols** (Hu, Shen,
  Lakshmipathi; arXiv:2605.09618, 2026) — routes between direct answering,
  voting and debate under an equal budget of 960 tokens per example; vote entropy
  predicts where debate is *safe*, not where it is *needed*, and self-critique
  probes fail as routing signals "due to format-compliance artifacts."
  Independent evidence that at open-weight scale, format and budget artifacts
  contaminate the signal before the protocol does.

---

## 3. Mixture-of-agents and aggregation

The competing explanation for team advantage: aggregation over samples, not
division of labour.

- **Mixture-of-Agents Enhances Large Language Model Capabilities** (Wang, Wang,
  Athiwaratkun, Zhang, Zou; arXiv:2406.04692, 2024) — layered aggregation where
  each agent sees all previous-layer outputs; 65.1% on AlpacaEval 2.0 against
  GPT-4 Omni's 57.5%. A multi-model, multi-generation system compared against
  single generations from one model — the compute confound in its purest form,
  and the paper does not rest on any claim about roles.

- **Rethinking Mixture-of-Agents: Is Mixing Different Large Language Models
  Beneficial?** (Li, Lin, Xia, Jin; arXiv:2502.00674, 2025) — Self-MoA
  aggregates outputs from only the single best model and beats standard MoA by
  6.6% on AlpacaEval 2.0 and 3.8% on average. **The cleanest ablation of "multi"
  in this literature**: hold the aggregation and the sample count, remove the
  heterogeneity, and performance goes *up*. Directly parallel to this paper's
  finding that the team-size manipulation contributes nothing once the resource
  it was confounded with is held fixed.

- **More Agents Is All You Need** (Li, Zhang, Yu, Fu, Ye; TMLR 2024,
  arXiv:2402.05120) — sampling-and-voting scales with agent count, and "the
  degree of enhancement is correlated to the task difficulty." **The strongest
  competing explanation for a difficulty-scaled team advantage**, and therefore
  the entry the paper must engage most carefully: this paper's §3.1 shows the
  cap artifact *also* grows with instance size. Two mechanisms predicting the
  same curve, one of which requires no collaboration and one of which requires no
  real effect at all.

- **Are More LLM Calls All You Need? Towards Scaling Laws of Compound Inference
  Systems** (Chen, Davis, Hanin, Bailis, Stoica, Zaharia, Zou;
  arXiv:2403.02419, 2024) — performance of Vote and Filter-Vote "can first
  increase but then decrease as a function of the number of LM calls," because
  extra calls help easy items and hurt hard ones. A non-monotone budget response
  in a compound system, which is what this paper measured within a single agent:
  3 rounds → 12 rounds moved the score down.

- **Large Language Monkeys: Scaling Inference Compute with Repeated Sampling**
  (Brown, Juravsky, Ehrlich, Clark, Le, Ré, Mirhoseini; arXiv:2407.21787, 2024)
  and **Scaling LLM Test-Time Compute Optimally can be More Effective than
  Scaling Model Parameters** (Snell, Lee, Xu, Kumar; arXiv:2408.03314, 2024) —
  both establish that generation budget is a first-class performance axis, with
  Snell et al. explicitly evaluating "on equivalent computational budgets."
  Together they are the citation for why an unmatched generation budget is not a
  minor confound: it is *the* known scaling axis, and any architecture comparison
  that leaves it free is measuring it instead.

---

## 4. Emergent roles and the MARL lineage

Extending what `docs/PLAN.md` already cites.

- **ROMA: Multi-Agent Reinforcement Learning with Emergent Roles** (Wang, Dong,
  Lesser, Zhang; ICML 2020, arXiv:2003.08039) — role embeddings emerge from a
  regulariser rather than assignment; agents with similar roles share learning.
  Already cited. Roles here are architectural and continuous; the emergence is
  demonstrated by a learned latent space, not read out of language.

- **RODE: Learning Roles to Decompose Multi-Agent Tasks** (Wang, Gupta, Mahajan,
  Peng, Whiteson, Zhang; arXiv:2010.01523, 2020) — clusters actions by
  environmental effect into restricted role action spaces, giving a bi-level
  hierarchy; beats SOTA on 10 of 14 SMAC maps and transfers to 3× more agents.
  **The extension ROMA needs**, because it makes the decomposition explicit and
  testable rather than latent — the MARL analogue of demanding an agent × task-
  component interaction rather than a scalar drop.

- **Generative Agents: Interactive Simulacra of Human Behavior** (Park, O'Brien,
  Cai, Ringel Morris, Liang, Bernstein; arXiv:2304.03442, 2023) — 25 agents in a
  sandbox produce coordinated emergent behaviour (the Valentine's party), with
  ablations confirming each architectural component matters for *believability*.
  The canonical emergent-behaviour result in LLM agents, and the canonical
  instance of the problem this project was built to address: the outcome measure
  is human-judged believability, so "emergence" is established by reading rather
  than by ablating a scored quantity.

- **"Who Am I, and Who Else Is Here?" Behavioral Differentiation Without Role
  Assignment in Multi-Agent LLM Systems** (El Kandoussi; arXiv:2604.00026, 2026)
  — 7 LLMs, 12 configurations, 208 runs, ~14,000 coded messages; heterogeneous
  groups differentiate substantially more than homogeneous ones, and removing
  prompt structure collapses differentiation to homogeneous levels.
  **Already cited, and the verified title differs from the one currently in
  `scripts/analysis/build_paper.py`** — the reference list says "Behavioral
  differentiation without role assignment in same-model LLM agent groups," which
  is neither the real title nor accurate about the paper, since its headline
  contrast is *heterogeneous vs homogeneous*. That matters: its own result says
  same-model groups differentiate least, which is the prior this project's null
  is consistent with.

- **Agents that Matter: Optimizing Multi-Agent LLMs via Removal-Based
  Attribution** (Lu, Huang, Lin, Lee; arXiv:2605.27621, 2026) — formalises credit
  assignment as a cooperative game; LOO identifies bottleneck agents as well as
  combinatorial methods at a fraction of the cost, giving up to 17% improvement
  at 35% lower cost, and finds contributions to diagnostic accuracy and ethical
  behaviour are decoupled. **Also cited under a title that is not its title** —
  the current reference reads "causal leave-one-out attribution for multi-agent
  LLM systems." Fix both before submission.

---

## 5. Measurement validity, reproducibility and evaluation critique

This is the literature the paper is a member of, and it is currently entirely
uncited. Ordered by how directly each maps onto the paper's own move.

- **Are Emergent Abilities of Large Language Models a Mirage?** (Schaeffer,
  Miranda, Koyejo; arXiv:2304.15004, 2023) — emergent abilities "appear due to
  the researcher's choice of metric rather than due to fundamental changes in
  model behavior with scale," demonstrated by predicting where emergence appears,
  auditing BIG-Bench claims, and *inducing* emergence in vision models by metric
  choice. **The closest structural analogue in the whole survey.** A celebrated
  phenomenon, traced to a property of the instrument, with the diagnostic offered
  as the transferable contribution. The difference to state: their artifact lives
  in the scoring function, this paper's lives in the generation budget — a
  metric can be recomputed offline, a truncated answer cannot.

- **Deep Reinforcement Learning that Matters** (Henderson, Islam, Bachman,
  Pineau, Precup, Meger; AAAI 2018, arXiv:1709.06560) — reproducing SOTA deep RL
  is "seldom straightforward"; reported gains fall inside the variance induced by
  seeds and hyperparameters. **The ancestor of §4 of the paper.** Their finding
  that a small number of seeds manufactures rankings is precisely what a
  four-agent baseline estimated from 48 episodes at `sd` ≈ 0.15 did here, and
  their prescription — report across many seeds, report variance — is what the
  fresh-seed rule enforces.

- **Are We Really Making Much Progress? A Worrying Analysis of Recent Neural
  Recommendation Approaches** (Ferrari Dacrema, Cremonesi, Jannach; RecSys 2019,
  arXiv:1907.06902) — of 18 neural recommenders from top venues, 7 reproduce,
  6 of those are beaten by simple heuristics, and the survivor does not
  consistently beat a well-tuned linear method. **The canonical under-tuned-
  baseline result**, and the model for the paper's recommendation 5: a
  single-agent baseline given the team's prompt is an under-tuned baseline by
  construction.

- **With Little Power Comes Great Responsibility** (Card, Henderson, Khandelwal,
  Jia, Mahowald, Jurafsky; EMNLP 2020, arXiv:2010.06595) — underpowered
  experiments are common in NLP; most SOTA comparisons on several GLUE tasks
  lack the power to detect the differences they report. **The citation for §4.**
  A 48-episode arm with `sd` ≈ 0.15 has a standard error of ~0.021 against
  effects of ~0.05 — this paper is an instance of their argument that produced a
  headline finding and then unproduced it.

- **Show Your Work: Improved Reporting of Experimental Results** (Dodge,
  Gururangan, Card, Schwartz, Smith; arXiv:1909.03004, 2019) — test-set numbers
  alone cannot establish that one model beats another; report expected validation
  performance *as a function of compute budget*, since conclusions flip with the
  budget available. **The direct methodological precedent for recommendation 3.**
  They asked for performance-vs-budget curves in hyperparameter search; this
  paper asks for the same thing across architectures at inference.

- **A Metric Learning Reality Check** (Musgrave, Belongie, Lim; arXiv:2003.08505,
  2020) — reported improvements over years of metric learning are "marginal at
  best" once baselines are tuned and evaluation is held fixed. A second
  independent instance of the under-tuned-baseline pattern, in vision.

- **Annotation Artifacts in Natural Language Inference Data** (Gururangan,
  Swayamdipta, Levy, Schwartz, Bowman, Smith; NAACL 2018, arXiv:1803.02324) — a
  hypothesis-only classifier gets ~67% on SNLI and ~53% on MultiNLI, so "the
  success of natural language inference models to date has been overestimated."
  The origin of *artifact* in this sense. Theirs is in the data, this paper's is
  in the harness, but the argument shape — a shortcut that mimics the target
  phenomenon — is identical, and §3.1's "the artifact grows with instance size in
  exactly the shape a genuine collaboration benefit would predict" is the same
  observation.

- **Measurement and Fairness** (Jacobs, Wallach; FAccT 2021, arXiv:1912.05511) —
  imports construct reliability and construct validity from the social sciences:
  harms arise from mismatch between an unobservable construct and its
  operationalisation. **The vocabulary the paper is missing.** "Collaboration
  benefit" is the construct; "team score minus solo score at matched turns" is
  the operationalisation; the cap artifact is a validity failure in that
  operationalisation, and saying so places the contribution in an existing
  framework rather than presenting it as a one-off bug report.

- **Underspecification Presents Challenges for Credibility in Modern Machine
  Learning** (D'Amour, Heller, Moldovan, Adlam, Alipanahi, Beutel, Chen, Deaton,
  Eisenstein, Hoffman, Hormozdiari, Houlsby, Hou, Jerfel, Karthikesalingam,
  Lucic, Ma, McLean, Mincu, Mitani, Montanari, Nado, Natarajan, Nielson, Osborne,
  Raman, Ramasamy, Sayres, Schrouff, Seneviratne, Sequeira, Suresh, Veitch,
  Vladymyrov, Wang, Webster, Yadlowsky, Yun, Zhai, Sculley; arXiv:2011.03395,
  2020) — pipelines that agree on held-out performance diverge in deployment,
  because the pipeline does not pin down the properties that matter. The general
  form of "identical specification, divergent behaviour," which is what a shared
  per-turn cap is.

- **Leakage and the Reproducibility Crisis in ML-based Science** (Kapoor,
  Narayanan; arXiv:2207.07048, 2022) — 329 papers across 17 fields affected by 8
  types of leakage; in their civil-war case study, *all* papers claiming complex
  ML beats logistic regression fail to reproduce. Their remedy — model info
  sheets, a mandatory disclosure form — is the same instrument as this paper's
  recommendations 1 and 2: make the diagnostic a required field rather than an
  optional appendix.

- **What Will it Take to Fix Benchmarking in Natural Language Understanding?**
  (Bowman, Dahl; NAACL 2021, arXiv:2104.02145 — note the arXiv comments field
  misstates the venue year as 2020) — four criteria a benchmark must satisfy;
  most do not, and adversarial data collection obscures measurement rather than
  fixing it. Useful for framing the paper's recommendations as benchmark-design
  criteria rather than as advice.

- **NLP Evaluation in trouble: On the Need to Measure LLM Data Contamination for
  each Benchmark** (Sainz, Campos, García-Ferrero, Etxaniz, Lopez de Lacalle,
  Agirre; arXiv:2310.18018, 2023) and **Don't Make Your LLM an Evaluation
  Benchmark Cheater** (Zhou, Zhu, Chen, Chen, Zhao, Chen, Lin, Wen, Han;
  arXiv:2311.01964, 2023) — benchmark contamination inflates reported capability
  and undermines comparison. The reason this project uses a synthetic,
  programmatically graded task family (Limitations, §8), and the citation that
  turns that choice from a limitation into a control.

- **Lessons from the Trenches on Reproducible Evaluation of Language Models**
  (Biderman, Schoelkopf, Sutawika, Gao, Tow *et al.*; arXiv:2405.14782, 2024) —
  documents model sensitivity to evaluation setup, the difficulty of comparing
  across methods, and the absence of reproducibility, from building the LM
  Evaluation Harness. The most concrete statement that *harness details decide
  results*, which is this paper's finding stated as folk knowledge; this paper
  supplies a measured instance with a mechanism and an effect size.

- **The False Promise of Imitating Proprietary LLMs** (Gudibande, Wallace, Snell,
  Geng, Liu, Abbeel, Levine, Song; arXiv:2305.15717, 2023) — imitation models win
  crowd ratings by matching ChatGPT's *style* while closing little of the
  capability gap. A reported gain that was an evaluation artifact, found by
  changing the measurement rather than the model — the same move as replacing a
  turn-matched arm with a generation-matched one.

- **MemDelta: Controlled Baselines and Hidden Confounds in Agent Memory
  Evaluation** (Wang; arXiv:2606.29914, 2026) — in agent memory evaluation,
  uncontrolled variables (embedding choice, model family, write-path cost)
  account for effect sizes as large as the architectural differences claimed;
  recommends reporting write-path cost before attributing gains to architecture.
  **The same argument in an adjacent subfield, published this year.** Strong
  evidence that "control the resource before crediting the architecture" is a
  live and general problem, not a quirk of one scheduling task.

- **Baselines Before Architecture: Evaluating Coding Agents for Autonomous
  Penetration Testing** (Dhakal, Neupane, Chaudhary; arXiv:2607.13085, 2026) —
  published autonomous-pentest systems change both harness and backbone at once,
  so a controlled study on the 104-task XBOW benchmark supplies "model-matched
  plain-agent baselines" to separate the two. Same discipline, different domain:
  build the baseline the original paper should have run, then see what is left.

---

## 6. Candidates for the Tier-2 audit

Ranked by whether the artifact could actually be *measured* in the released
material, not just argued about. The audit needs three things per episode: the
final answer text, a signal of whether generation stopped at the cap, and the
arm it belongs to.

| Source | What is public | Usable for the audit? |
|---|---|---|
| **MAST-Data** — [`huggingface.co/datasets/mcemri/MAD`](https://huggingface.co/datasets/mcemri/MAD), code at [`github.com/multi-agent-systems-failure-taxonomy/MAST`](https://github.com/multi-agent-systems-failure-taxonomy/MAST) | `MAD_full_dataset.json`: 1,642 annotated traces, 7 frameworks (AG2, MetaGPT, ChatDev, Magentic, AppWorld, HyperAgent, OpenManus), 8 benchmarks, 5 LLMs, with a `trajectory` field per trace. `MAD_human_labelled_dataset.json`: 19 traces | ~~**Best candidate.**~~ **Corrected 2026-08-21: not a candidate at all.** MAST-Data has **no single-agent arm**. Its `mas_name` field separates AG2 from MetaGPT from ChatDev — every one of them multi-agent — so there is no solo baseline to compare against and the contrast this audit needs cannot be formed from it. The dataset card names no `finish_reason` and no token counts either. Cross-framework annotation made it look like the best candidate; the audit question is a *within-comparison* question, and no amount of cross-framework coverage supplies a missing arm. See `docs/AUDIT-TARGETS.md` for what replaced it |
| **ChatDev** — [`github.com/OpenBMB/ChatDev`](https://github.com/OpenBMB/ChatDev) | A `WareHouse/` directory of generated software projects with per-run logs; the paper reports per-run token totals (GPT-Engineer 7,182 vs ChatDev 22,949) | **Strong.** The paper already publishes the 3.2× generation gap against its single-agent baseline in Table 3; the audit is to check whether GPT-Engineer runs terminate at the cap. Needs a repo walk to confirm the WareHouse logs retain finish reasons |
| **CAMEL** — [`github.com/camel-ai/camel`](https://github.com/camel-ai/camel) | Released role-playing conversation datasets; the 40-message limit is a documented parameter | **Highest prior of finding the artifact**, because the design is a 40-message conversation summarised into one answer versus a `gpt-3.5-turbo` single shot. The single-shot arm is the one to instrument |
| **AgentVerse** — [`github.com/OpenBMB/AgentVerse`](https://github.com/OpenBMB/AgentVerse) | Framework plus task configurations for the `Solo` and `Group` conditions | **Best re-run candidate rather than best log candidate.** `Solo` and `Group` share the scaffolding, so re-running both under a matched generation budget is a clean, cheap replication of this paper's C5 contrast on someone else's system |
| **Du et al. multi-agent debate** — [`composable-models.github.io/llm_debate/`](https://composable-models.github.io/llm_debate/) | Project page with code and generated debate data | Worth checking. Debate rounds have fixed prompts across tasks, so the cap lands uniformly — which makes it a good place to test the *absence* of the artifact as a control |
| **AutoGen / Magentic-One** — AutoGenBench | Benchmark runner with repetition and isolation controls | Instrumentation exists; released *logs* do not, as far as could be verified. Would require re-running |
| **AlpacaEval `results/`** — [`github.com/tatsu-lab/alpaca_eval`](https://github.com/tatsu-lab/alpaca_eval) | 228 model entries, each with `model_outputs.json`. `Together-MoA` and `SelfMoA_gemma-2-9b-it-SimPO` hold 805 records each against their own single-model baselines on a matched task set | **Added 2026-08-21, and now the best candidate.** The only source found with two arms on one task set and raw final generations released. ~7 MB over plain HTTPS, no clone, no auth. Yields the verbosity ratio only — no `finish_reason`, no token counts — so it bounds artifact 2 and says nothing about truncation. Detail in `docs/AUDIT-TARGETS.md` |
| **Tran & Kiela** (arXiv:2604.02460) | No public code, data or transcript release could be verified from the abstract page | Cannot be audited from the outside as of 2026-08-20; worth an email |

**What the audit is looking for, stated so it is falsifiable:** in any released
multi-agent corpus, the distribution of final-turn generation length in the
single-agent arm should pile up at the configured cap while the multi-agent arm's
does not. If it does, the reported gain is partly this artifact. If both arms sit
well below the cap, the artifact is absent and the reported gain stands — and
that is a publishable negative for the audit too.

---

## 7. Scooping check

**Result: not scooped, but the paper has a close neighbour that it must cite and
explicitly distinguish itself from, and one more that reports the same ratio at
the same model scale.**

### The neighbour

**Single-Agent LLMs Outperform Multi-Agent Systems on Multi-Hop Reasoning Under
Equal Thinking Token Budgets** (Dat Tran, Douwe Kiela; arXiv:2604.02460,
submitted 2 April 2026, revised 11 April 2026). Verified on the arXiv abstract
page. From the abstract: "Recent work reports strong performance from multi-agent
LLM systems (MAS), but these gains are often confounded by increased test-time
computation. When computation is normalized, single-agent systems (SAS) can match
or outperform MAS." They give an information-theoretic argument from the Data
Processing Inequality, test on FRAMES and MuSiQue with Qwen3-30B-A3B,
DeepSeek-R1-Distill-Llama-70B and Gemini 2.5 Flash/Pro, and conclude that "many
reported advantages of multi-agent systems are better explained by unaccounted
computation and context effects rather than inherent architectural benefits."

**This is the same conclusion.** It must be cited in the abstract-adjacent part
of the introduction, not buried in Related Work.

**Four things separate this paper from it, all checkable:**

1. **Opposite direction of the asymmetry.** Tran & Kiela match a *global*
   thinking-token budget *B* and split it across agents, so their concern is that
   MAS gets more total compute. Their identified artifact runs the same way: in
   Appendix G they report that for Gemini 2.5, visible thought text from SAS
   "tends to plateau well below the requested budget, while MAS surfaces more
   visible thought content under the same requested budget B, due to multiple
   calls." Their solo arm *under-spends*. This paper's solo arm is *forced to
   over-spend in one turn and is truncated for it* — 34% truncation, 20 cut
   answers out of 150, against the team's 8% and 4. Same headline, opposite
   mechanism.

2. **They normalise total budget; this paper's cap is per-turn.** A total-budget
   protocol does not surface the failure reported here, because the failure is
   about *where in the episode* the budget is spent. This paper's C5 arm gave one
   agent the team's entire turn budget and the artifact survived — the solo agent
   still emitted 2.04× the team's text and still hit the per-turn cap at twelve
   rounds. A global normalisation would have scored that as matched.

3. **They do not report differential final-answer truncation.** The diagnostic
   sections do not discuss truncation of the final answer turn or which arm it
   damages. That mechanism, the truncation counts, and the fix (a separate
   `answer_max_tokens` budget on the parsed turn, with slot geometry held
   constant) are not in their paper.

4. **Different regime.** They run 30B–70B open models and a frontier API at
   multi-hop QA; this paper runs a single 8B model at Q4_K_M on a synthetic
   constrained-scheduling family with per-component grading, and reports a
   bounded null on team size rather than a single-agent win. Their result does
   not predict this one, and vice versa — which is worth saying, because two
   independent instruments finding the same confound in different regimes is
   stronger than either alone.

**Do not soften the framing to protect novelty.** The correct move is to lead
with agreement — the confound is real and now independently reported — and claim
the mechanism, the diagnostics, and the truncation counts, which are this paper's
and are not in Tran & Kiela.

### The second-closest

**The Cost of Consensus** (Bertalanič, Fortuna; arXiv:2605.00914, 2026) reports
that debate "consumes 2.1–3.4× more tokens (up to 28,631 tokens per problem) than
self-correction for equal or lower accuracy," across 7–8B models. Same model
scale as this paper, and a token ratio in the same range as this paper's measured
2.04×. It is a cost-effectiveness argument rather than an artifact argument — it
does not claim any reported result was an artifact of an instrument limit — but
it is close enough that omitting it would look like a gap in the reading.

### The strongest public statement of the confound is not a paper

Anthropic's engineering post *How we built our multi-agent research system*
states: "In our data, agents typically use about 4× more tokens than chat
interactions, and multi-agent systems use about 15× more tokens than chats";
"token usage by itself explains 80% of the variance" on BrowseComp; and "a
multi-agent system with Claude Opus 4 as the lead agent and Claude Sonnet 4
subagents outperformed single-agent Claude Opus 4 by 90.2% on our internal
research eval." A production team reporting a 90.2% multi-agent win, a 15×
token multiplier, and 80% of variance explained by tokens — in the same document,
without matching the arms. Worth quoting once, as evidence that the confound is
load-bearing in deployed systems and not a small-scale academic curiosity.
Not peer-reviewed; cite as an engineering report and label it as such.

### What was searched, and did not turn up

No paper was found that reports **per-turn** budget asymmetry between
single-agent and multi-agent arms, or that identifies final-answer truncation as
the mechanism behind a reported multi-agent gain. Searches covered token/turn
budget asymmetry, `max_tokens` truncation artifacts in agent evaluation,
matched-compute multi-agent comparisons, and evaluation-artifact framings of
multi-agent results. The prior work normalises *total* compute
(Tran & Kiela), compares *cost* at fixed accuracy (Bertalanič & Fortuna;
Smit et al.), or criticises *baseline strength and benchmark coverage*
(Zhang et al.). None of them locates the failure in where inside an episode a
symmetric limit binds. **That mechanism, and the three diagnostics, remain
this paper's.**

---

## Dropped — could not be verified

- **RODE venue.** Widely referred to as ICLR 2021; the arXiv record checked here
  does not state a venue, so it is cited above as arXiv:2010.01523, 2020 only.
  Confirm against OpenReview before the bibliography is final.
- **Venues for Du et al. (2305.14325), Reflexion (2303.11366), AgentVerse
  (2308.10848), AutoGen (2308.08155), Show Your Work (1909.03004), Schaeffer
  et al. (2304.15004), Generative Agents (2304.03442) and Sainz et al.
  (2310.18018)** — arXiv identifiers, titles, authors and years are verified;
  the peer-reviewed venue was not confirmed from the pages checked and is
  therefore omitted rather than guessed. Sainz et al. in particular returned
  conflicting venue years (EMNLP 2023 vs 2024 Findings) and needs one lookup.
- **MAST-Data human-labelled subset size.** The dataset card lists 19 traces in
  `MAD_human_labelled_dataset.json` against 150 traces described in the paper.
  An open issue on the MAST repository raises the same discrepancy. Resolve
  before relying on the human-annotated subset for the audit; the 1,642-trace
  LLM-annotated file is unaffected.
- **No fabricated entries.** Several search results pointed to blog posts and
  secondary write-ups of Tran & Kiela (Substack, DEV, Medium, personal blogs).
  None are cited; all claims attributed to that paper above come from the arXiv
  abstract page and the paper's own HTML.
