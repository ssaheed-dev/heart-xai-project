# Human-Centred Evaluation Protocol

A small, informal, within-subjects usability study comparing two explanation formats
produced by the Streamlit prototype. Target: **5–10 participants**, ~10–15 minutes each.

> **Check your university's ethics process first.** Even a low-risk study like this one,
> involving human participants, usually needs at least a lightweight departmental
> sign-off before you recruit anyone. This document is a starting protocol for you to
> adapt to whatever your department requires — it is not a substitute for that approval.

---

## 1. Before you recruit anyone

- Confirm the prototype runs correctly end-to-end (Section 13, step 4 of
  `project_proposal.md`).
- Prepare 2–3 fixed example patient profiles (fictional/synthetic — never use a real
  person's medical information) that you will enter into the app for every participant,
  so all participants see the same predictions. Write these down so your procedure is
  reproducible.
- Decide your design: **within-subjects** is recommended — every participant sees BOTH
  formats (visual-only, then text-only, or vice versa — alternate the order between
  participants to control for order effects).

## 2. Consent script (read or share with each participant before starting)

> *"Thank you for helping with my final-year project. I've built a prototype that
> predicts heart disease risk from example patient data and explains its predictions.
> I'd like you to look at a few example predictions and their explanations, answer a
> few short questions about each, and share your honest feedback. This takes about
> 10–15 minutes. Your responses are anonymous — please use a participant code rather
> than your name. You can stop at any time without giving a reason. This is a student
> research prototype, not a real medical tool, and no real patient data is used.
> Are you happy to continue?"*

Record only a participant code (e.g. P1, P2, ...) — never a name — in the app's
"Participant ID" field.

## 3. Procedure (repeat per participant)

1. Assign a condition order (alternate: odd-numbered participants see **visual first,
   then text**; even-numbered participants see **text first, then visual**).
2. For the first format: set the sidebar toggle accordingly, enter your first fixed
   example patient profile, click **Predict risk**, and let the participant read the
   result and explanation at their own pace.
3. Ask the participant to complete the in-app comprehension question, trust rating, and
   clarity rating (in the "Research evaluation" expander) for this example.
4. Repeat step 2–3 for your second fixed example, same format.
5. Switch the sidebar toggle to the other explanation format and repeat steps 2–4 with
   your third example (and a fourth if you want more data per participant).
6. Ask the two open-ended debrief questions below out loud and note the answers (pen and
   paper, or a shared doc — not the app).

## 4. In-app items (already built into `app/app.py`)

For each prediction shown, the participant answers:

- **Comprehension:** "Which factor had the SINGLE largest effect on this prediction?"
  (multiple choice: the true top SHAP feature plus two distractors, randomised order).
  Logged automatically as correct/incorrect.
- **Trust:** "I trust this prediction" (1 = strongly disagree, 5 = strongly agree).
- **Clarity:** "This explanation was easy to understand" (1 = strongly disagree,
  5 = strongly agree).
- **Optional comment:** free text.

These are logged automatically to `app/artifacts/evaluation_log.csv`, one row per
submitted response, with timestamp, participant ID, explanation format shown, and all
ratings.

## 5. Verbal debrief questions (ask after both formats, note answers separately)

1. "Which of the two explanation styles did you personally prefer, and why?"
2. "Was there anything about either explanation that was confusing or that you didn't
   trust?"

These are for qualitative colour in your Discussion section — a few representative,
anonymised quotes are enough; you don't need to formally code all responses unless your
supervisor asks for that level of rigour.

## 6. Analysing the log afterwards

A few lines of pandas is enough — do not over-build this part:

```python
import pandas as pd

log = pd.read_csv("app/artifacts/evaluation_log.csv")

summary = log.groupby("explanation_format").agg(
    n_responses=("participant_id", "count"),
    comprehension_accuracy=("comprehension_correct", "mean"),
    mean_trust=("trust_rating", "mean"),
    mean_clarity=("clarity_rating", "mean"),
)
print(summary)
```

Report this table directly in Section 8.4 of your report (`docs/report_template.md`).
With a sample this small, describe differences ("format X had higher self-reported
clarity in this sample") — do not claim statistical significance.

## 7. What NOT to do

- Do not invent or "tidy up" participant responses.
- Do not run the study on yourself and present it as multiple participants.
- Do not collect real patient/medical data from participants about themselves — all
  inputs to the model must be the fixed, synthetic example profiles you prepared.
- Do not skip the consent script, even for friends/coursemates.
