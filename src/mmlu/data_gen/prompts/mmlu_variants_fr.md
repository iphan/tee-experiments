# Variantes de questions à choix multiples MMLU (français)

Traduction française des cinq variantes de cadrage de `mmlu_variants.md` (les
variantes originales de Messing 2026, publiées en anglais uniquement). Seul le
texte des instructions est traduit ; la tâche, les champs à remplir
(`{question}`, `{A}`, `{B}`, `{C}`, `{D}`) et les délimiteurs des options
(`A.`, `(A)`, `A)`, `a)`) sont identiques à la version anglaise.

## v_0 — Standard

```text
Répondez à la question à choix multiples suivante. Indiquez uniquement la lettre (A, B, C ou D).

Question : {question}
A. {A}
B. {B}
C. {C}
D. {D}
```

## v_1 — Examen

```text
Vous passez un examen. Sélectionnez la bonne réponse à la question ci-dessous. Répondez uniquement par la lettre.

Q : {question}
(A) {A}
(B) {B}
(C) {C}
(D) {D}
```

## v_2 — Expert

```text
En tant qu'expert compétent, identifiez la bonne réponse à cette question. Indiquez uniquement la lettre du bon choix.

{question}
Options :
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

Réponse :
```

## v_4 — Analytique

```text
Lisez attentivement la question suivante et choisissez la meilleure réponse parmi les options proposées. Indiquez uniquement la lettre.

Question : {question}
Choix :
a) {A}
b) {B}
c) {C}
d) {D}
```
