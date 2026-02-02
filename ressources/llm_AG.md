## Task Definition

You must identify and annotate **mentions of legal authorities** in Canadian legal decisions.
Each **mention (occurrence)** must be annotated, whether **complete or partial**.

### Authority Types (choose exactly one)

* `<legislation>` — statutes, codes, constitutional texts
* `<decision>` — judicial or administrative decisions
* `<secondary_sources>` — doctrine, books, articles, reports, dictionaries
* `<unable_to_classify>` — ambiguous sources (borderline law vs fact)

---

## Core Principles (MANDATORY)

1. **Annotate every occurrence** of an authority, even if abbreviated.
2. **Reuse the same `docid`** for all mentions of the same authority **within the same document**.
3. **Always apply a general tag first**, then specific tags inside it.
4. **Fragments must be ≤ 2 stop-words away** from the title/reference/source to belong to the same mention.

   * Otherwise, create a **new mention** with the same `docid`.

---

## Global Annotation Structure

Each mention follows this structure:

```
<AUTHORITY_TYPE docid="...">
    <title titletype="official|alias">...</title>
    <reference>...</reference>        (legislation / decisions only)
    <authors>...</authors>            (secondary sources only)
    <source>...</source>              (secondary sources only)
    <fragment fragmentid="...">...</fragment>
</AUTHORITY_TYPE>
```

Only include the elements **that actually appear** in the text.

---

## 1. Legislation

### 1.1 General Rules

* Wrap the **entire mention** in `<legislation>`.
* Assign a **manual `docid`** using a **short, stable identifier**.

#### Examples of `docid`

```
IRPA        → Immigration and Refugee Protection Act
Charter    → Canadian Charter of Rights and Freedoms
CCQ        → Civil Code of Quebec
CriminalCode
CBCA
CharteQuebecoise
```

---

### 1.2 Titles

* Tag the **law name** as `<title>`:

  * `titletype="official"` → full legal title
  * `titletype="alias"` → acronyms, short names, “the Act”

#### Example

```
<title titletype="official">Immigration and Refugee Protection Act</title>
<title titletype="alias">IRPA</title>
<title titletype="alias">the Act</title>
```

If **official + alias appear together**, tag both under the same `<legislation>`.

---

### 1.3 References

* Tag statute citations as `<reference>`

#### Example

```
<reference>SC 2001, c 27</reference>
<reference>RSC 1985, c C-46</reference>
```

---

### 1.4 Fragments (CRITICAL)

Each cited part of a statute must be tagged as `<fragment fragmentid="...">`.

#### Standard Fragment IDs (Federal / Common Structure)

| Legal Element | fragmentid format      |
| ------------- | ---------------------- |
| Section       | `sec 1`                |
| Subsection    | `subsec 1(1)`          |
| Paragraph     | `para 1(1)(a)`         |
| Subparagraph  | `subpara 1(1)(a)(i)`   |
| Clause        | `clause 1(1)(a)(i)(A)` |
| Schedule      | `schedule I`           |

📌 **Rule**: Use the **last subdivision** to name the fragment type.

---

#### Quebec Codes (CCQ, etc.)

| Element                  | fragmentid         |
| ------------------------ | ------------------ |
| Book                     | `book I`           |
| Title                    | `title I`          |
| Chapter                  | `chapter II`       |
| Article                  | `sec 53`           |
| Paragraph (non-numbered) | `para 53(1)`       |
| Subparagraph             | `subpara 53(2)(5)` |

---

#### Non-standard fragments

If the structure does not fit standard formats:

* Set `non_standard=true`
* Reproduce the format clearly

##### Example

```
fragmentid="Preamble, para. 3"
fragmentid="Schedules I, II, s 53"
```

---

### 1.5 Special Cases

#### Laws incorporating other laws

* **Annotate each law separately**

#### Constitutional texts

* Each constitutional instrument = **distinct `<legislation>`**

---

## 2. Decisions (Case Law)

### 2.1 General Rules

* Wrap each mention in `<decision>`
* Assign a **short, distinctive `docid`**

#### Examples

```
Appulonappa
Pfizer
Lessard_QCCA
Lessard_CSC
```

📌 If same parties, different courts → **different `docid`s**

---

### 2.2 Titles

```
<title titletype="official">R v Appulonappa, 2015 SCC 59</title>
<title titletype="alias">Appulonappa</title>
```

---

### 2.3 References

Each citation = its own `<reference>`

```
<reference>2015 SCC 59</reference>
<reference>[1986] RJQ 123</reference>
```

---

### 2.4 Fragments (Pages / Paragraphs)

| Fragment type | fragmentid          |
| ------------- | ------------------- |
| Page          | `p 353`             |
| Paragraph     | `para 25`           |
| Interval      | `para 25 - para 35` |
| Multiple      | separate fragments  |

#### Example

```
<fragment fragmentid="para 26"/>
<fragment fragmentid="para 32"/>
<fragment fragmentid="para 46"/>
```

📌 Always repeat `p` or `para` on **both bounds** of an interval.

---

## 3. Secondary Sources (Doctrine)

### 3.1 General Rules

* Wrap each mention in `<secondary_sources>`
* `docid` = short distinctive title (≤ 5 words if possible)

#### Examples

```
La_responsabilite_civile
Post_Confederation_Rights
Exclusion_des_biens
```

---

### 3.2 Titles

* Books → book title
* Articles → article title
* Collective works → contribution title only

```
<title titletype="official">La responsabilité civile</title>
<title titletype="alias">La responsabilité</title>
```

---

### 3.3 Authors

* All authors in **one `<authors>` tag**
* Editors of collective works → **NOT authors**

```
<authors>J.-L. Baudouin; P. Deslauriers; B. Moore</authors>
```

---

### 3.4 Source

Includes **journal / publisher / year / volume / starting page**

```
<source>(1958-59) 3 Themis 225</source>
<source>Éditions Yvon Blais, 1997</source>
```

If CanLII Docs exists → add **second `<source>`**

---

### 3.5 Fragments

| Fragment  | fragmentid         |
| --------- | ------------------ |
| Page      | `p 127`            |
| Paragraph | `no 1958`          |
| Interval  | `p 225 - p 227`    |
| Mixed     | separate fragments |

#### Example

```
<fragment fragmentid="no 2615"/>
<fragment fragmentid="p 1499"/>
```

📌 Starting page of article/book = **source**, not fragment
(unless later cited specifically)

---

## 4. Unable to Classify

Use `<unable_to_classify>` **only if**:

* Unclear whether legal authority or factual evidence
* Borderline interpretative document

Assign a `docid` anyway.
