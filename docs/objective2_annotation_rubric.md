# Objective 2 manual correctness annotation rubric

## Scope and cautions

Annotate each generated response independently for the question it answers.
The `reference_answer` is copied from IndicLegalQA as a reference for review; it
is not guaranteed ground truth. Reviewers should use their legal knowledge and,
where appropriate, authoritative legal sources rather than treating the
reference answer as conclusive.

Agreement between generated responses is not evidence of legal correctness by
itself. Similar or repeated responses can share the same error, and a response
that differs from the others can still be legally correct. Similarity and
uncertainty metrics are descriptive signals only and must not determine a
correctness label.

## Labels

Use exactly one of these labels:

- **`correct`**: The response answers the question accurately and contains no
  material legal error. Minor wording or formatting issues do not change the
  answer's meaning.
- **`incorrect`**: The response is materially wrong, answers a different
  question, relies on a false legal proposition, or omits information necessary
  for the requested answer.
- **`partially_correct`**: The response contains a materially correct component
  but is incomplete, imprecise, or also contains a material error. Explain
  which part is correct and what is missing or wrong.
- **`uncertain`**: The reviewer cannot make a reliable determination from the
  available material, or the question/reference answer is too ambiguous to
  support a defensible label. Explain what evidence is missing.

## Annotation fields

Each response has an `annotation` object:

- `correctness_label`: one of the four labels above, or blank while unreviewed.
- `rationale`: a concise explanation supporting the label. Required once a
  label is recorded.
- `reviewer_id`: the human reviewer identifier. Required once a label is
  recorded.

An unreviewed response must leave all three fields blank. Do not enter a
provisional label without also recording a rationale and reviewer ID.

## Practical review workflow

1. Read the legal question and generated response without using the similarity
   scores to decide the label.
2. Read the IndicLegalQA answer as a reference answer, remembering that it is
   not guaranteed ground truth.
3. Check the response's parties, legal rule, reasoning, dates, and conclusion
   for material accuracy and whether it addresses the question asked.
4. Record one label, a short rationale identifying the decisive evidence, and
   the reviewer ID.
5. For a disputed or ambiguous case, use `uncertain` temporarily and describe
   the missing authority or point of disagreement. A second reviewer should
   independently review it; retain both rationales outside this single-label
   template or resolve the case through an agreed adjudication process.
6. If reviewers disagree, do not choose the majority label solely because it is
   the majority. Compare the rationales, consult authoritative sources, and
   document the adjudicator's decision and remaining uncertainty.

The validation script checks annotation completeness and label vocabulary; it
does not judge whether a legal rationale is substantively correct.
