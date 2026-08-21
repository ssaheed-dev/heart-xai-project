# Technical Report Template

> **How to use this file:** every section below has a short guidance note in
> *[square italics]* describing what belongs there. Delete the guidance note as you
> replace it with your own writing. Anywhere you see `[YOUR RESULT]`, that number/plot
> must come from actually running your own pipeline — do not fill these in from this
> template, from example numbers you've seen elsewhere, or by guessing.

---

## Title Page
Project title, your name, programme, supervisor (if assigned), date.

## Abstract
*[150–250 words. Write this LAST. One or two sentences each on: motivation, method,
what you built, your headline results, and what it means for your proposed MSc
direction.]*

## 1. Introduction
*[Why this problem matters (healthcare + AI adoption), what the project does at a
glance, and how it connects to your proposed MSc research on human-centred XAI. End
with a one-paragraph roadmap of the report.]*

## 2. Literature / Background
*[Briefly cover: (a) ML for cardiovascular risk prediction — what's been tried, roughly
what performance is typical; (b) explainability methods (SHAP, LIME) — what they are,
what problem they solve; (c) the human-centred XAI gap — evidence that technical
explanation quality and human understanding/trust are not the same thing, and why that
matters for healthcare specifically. Cite properly — paraphrase sources in your own
words, use quotation marks and citations only for load-bearing exact phrases.]*

## 3. Problem Statement
*[Restate, in your own words, the specific gap this project addresses — see Section 3
of `project_proposal.md` as a starting point, but make it your own.]*

## 4. Aim and Objectives
*[Restate your aim (one sentence) and your 4–6 measurable objectives. Mark each
objective as met / partially met / not met once the project is complete, with a
one-line justification for each.]*

## 5. Methodology
*[Describe, in past tense (this is a completed pipeline), what you actually did: data
source and access method, cleaning decisions and why, EDA approach, train/test split
and why, models compared and why, evaluation metrics and why, XAI method and why, human
evaluation design and why. This should read as a methodological justification, not just
a list of steps.]*

## 6. System Design
*[Include your system architecture diagram (Section 12 of `project_proposal.md`, or
your own redrawn version) and a short explanation of each component and how data flows
between them.]*

## 7. Implementation
*[Describe key implementation decisions and any deviations from the original plan —
e.g. did you have to change an imputation strategy, did a package version cause an
issue, did you change which model you selected as "best" and why. Include short code
excerpts (a few lines) where they illustrate a specific decision, not full file dumps —
your code is already in the repository.]*

## 8. Results
*[This section is 100% your own numbers and plots. Suggested structure:]*

### 8.1 Data & EDA
`[YOUR RESULT: class balance, 1–2 EDA plots, 2–3 sentence summary]`

### 8.2 Model Performance
`[YOUR RESULT: your metrics table — accuracy/precision/recall/F1/ROC-AUC per model,
your ROC curve plot, your confusion matrix for the selected model]`

### 8.3 Explainability
`[YOUR RESULT: your SHAP global summary plot, your SHAP local waterfall plot(s) for
1–2 example patients, with your own written interpretation of what each shows]`

### 8.4 Human Evaluation
`[YOUR RESULT: number of participants recruited, comprehension accuracy per
explanation format, mean trust/clarity ratings per format, 2–3 representative
(anonymised) participant comments]`

## 9. Discussion
*[Answer your RQ1–RQ4 directly and explicitly, referring back to Section 8. Compare
your findings to the literature reviewed in Section 2 — where do they agree/disagree,
and why might that be? Discuss what your small, informal evaluation can and cannot tell
you.]*

## 10. Limitations
*[Be specific and honest: sample size (both dataset and evaluation study), single-site
data (Cleveland only, if that's what you used), non-clinical evaluation participants,
threshold choices, any model/library version dependencies, anything you would do
differently with more time.]*

## 11. Conclusion
*[Summarise what was achieved against the objectives, and — this is the part a
prospective MSc supervisor will care about most — explicitly state 2–3 concrete ways
this project's findings and limitations inform your proposed MSc research direction.]*

## 12. References
*[Full reference list in your department's required style. At minimum include: the UCI
Heart Disease dataset citation (see the notebook's final cell), the original SHAP paper
(Lundberg & Lee, 2017, "A Unified Approach to Interpreting Model Predictions"), and
whatever literature you drew on in Section 2.]*

---

## Interview Prep Checklist

Before your supervisor meeting, make sure you can answer each of these **without notes**,
because every answer should be something you actually did and can point to in your own
repository:

- Why this dataset, and what are its known limitations?
- Why these four models, and what did comparing them actually show you?
- What preprocessing/cleaning decisions did you make, and why (not just "how")?
- Why SHAP specifically, and what does a SHAP value mathematically represent?
- Walk through one local explanation plot in detail — what does it mean for that
  specific input?
- What did your human evaluation actually find, and how confident can you be in it given
  the sample size?
- What was your personal contribution, end to end?
- If you had another 3 months, what would you build next — and how does that map onto
  your proposed MSc research?
