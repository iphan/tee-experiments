# MMLU multiple-choice variant templates

Five framing variants for multiple-choice QA. Fill-in fields: `{question}`,
`{A}`, `{B}`, `{C}`, `{D}`. The task (select the correct letter from four
choices) is held fixed; framing and option delimiters vary.

## v_0 — Standard

```text
Answer the following multiple-choice question. Reply with only the letter (A, B, C, or D).

Question: {question}
A. {A}
B. {B}
C. {C}
D. {D}
```

## v_1 — Exam

```text
You are taking a test. Select the correct answer for the question below. Respond with just the letter.

Q: {question}
(A) {A}
(B) {B}
(C) {C}
(D) {D}
```

## v_2 — Expert

```text
As a knowledgeable expert, identify the correct answer to this question. Output only the letter of the correct choice.

{question}
Options:
A) {A}
B) {B}
C) {C}
D) {D}
```

## v_3 — Minimal

```text
{question}
A. {A}
B. {B}
C. {C}
D. {D}

Answer:
```

## v_4 — Analytical

```text
Read the following question carefully and select the best answer from the options provided. State only the letter.

Question: {question}
Choices:
a) {A}
b) {B}
c) {C}
d) {D}
```
