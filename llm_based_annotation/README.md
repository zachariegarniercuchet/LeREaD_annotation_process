# LLM-Based Legal Authority Annotation Pipeline

## Overview

This module implements a controlled annotation pipeline using Large Language Models (GPT-5) to extract and label legal authorities in judicial decisions. The core challenge is ensuring that LLMs **copy the input text faithfully while adding annotation labels**, without modifying, hallucinating, or omitting content.

## The Copy-and-Annotate Challenge

### Task Definition

We prompt LLMs to annotate text by copying the input verbatim and inserting XML-style labels around legal authorities:

**Input:**
```
The decision in R. v. Oakes, [1986] 1 SCR 103, is well established.
```

**Expected Output:**
```
The decision in <decision>R. v. Oakes, [1986] 1 SCR 103</decision>, is well established.
```

### Why Transformers Excel at Copying

Recent research shows that transformer architectures are exceptionally good at copying tasks. As demonstrated in [*Repeat After Me: Transformers are Better than State Space Models at Copying*](https://arxiv.org/pdf/2402.01032), transformers can reliably reproduce input sequences, making them well-suited for annotation tasks that require faithful text preservation.

However, **"good at copying" ≠ "perfect copying"**. Even with precise prompts, LLMs can:
- **Drop characters or tokens** (e.g., missing punctuation, spaces)
- **Add extraneous text** (hallucinated content, extra explanations)
- **Modify wording** (paraphrasing, "fixing" grammar)
- **Insert malformed labels** (unclosed tags, incorrect nesting)

### Our Solution: Multi-Layer Validation Pipeline

To ensure output quality, we implement a three-stage pipeline:

```
┌─────────────────────────┐
│   1. RAW LLM OUTPUT     │
│   (potentially flawed)  │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  2. POST-PROCESSING     │
│  • Extract markers      │
│  • Tokenize output      │
│  • Normalize tags       │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  3. ERROR CORRECTION    │
│  • Levenshtein-based    │
│  • Token alignment      │
│  • Character recovery   │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  4. VERIFICATION        │
│  ✓ No hallucination     │
│  ✓ Tag consistency      │
│  ✓ Schema compliance    │
└───────────┬─────────────┘
            │
       Pass │  Fail
            ▼    │
        Success  ▼
            ┌──────────┐
            │ FALLBACK │
            │ (retry)  │
            └──────────┘
```

---

## Core Components

### 1. Tokenization Strategy

We use **word-level tokenization with special handling for HTML tags** to enable precise alignment between input and output:

```python
from utils.tokenizer_utils import tokenize_html

# Input text with annotations
text = "See <decision>R. v. Smith</decision> at para. 15."

# Tokenization preserves tags as single tokens
tokens = tokenize_html(text)
# ['See', ' ', '<decision>', 'R', '.', ' ', 'v', '.', ' ', 'Smith', '</decision>', ' ', 'at', ' ', 'para', '.', ' ', '15', '.']
```

**Key insight**: By treating tags as atomic tokens, we can:
- Compare input and output token-by-token
- Detect exactly where the LLM deviated from the source
- Apply surgical corrections without affecting surrounding text

### 2. Error Correction via Levenshtein Alignment

When the LLM drops or adds characters, we use **Levenshtein distance-based alignment** to recover the original text:

```python
from utils.processing_utils import apply_operations

# LLM accidentally dropped a space and period
original = ['R', '.', ' ', 'v', '.', ' ', 'Smith']
llm_output = ['R', ' ', 'v', '.', ' ', 'Smith']  # Missing '.' after 'R'

# Compute minimal edit operations (insertions, deletions, substitutions)
corrected = apply_operations(original, llm_output)
# Result: ['R', '.', ' ', 'v', '.', ' ', 'Smith']
```

**How it works**:
1. Compute Levenshtein distance between original and output tokens
2. Backtrack to find optimal alignment (which tokens correspond)
3. Apply minimal insertions/deletions to make output match input
4. Preserve LLM-inserted labels while correcting text errors

This approach catches common LLM mistakes:
- **Missing punctuation**: `"R. v"` → `"R v"` (corrected to `"R. v"`)
- **Dropped spaces**: `"Smith v.Jones"` → `"Smith v. Jones"`)
- **Extra characters**: `"para. 15"` → `"para . 15"` (corrected)

### 3. Three-Level Verification

After correction, we validate the output against three criteria:

#### 3.1 Hallucination Check

**Goal**: Ensure LLM didn't add or modify non-label text

```python
from utils.verification_utils import check_hallucination

# Extract non-label tokens from both input and output
original_text = ['The', ' ', 'decision', ' ', 'in', ' ', 'Smith']
output_text = ['The', ' ', 'decision', ' ', 'in', ' ', 'Smith']  # Must match exactly

result = check_hallucination(original_text, output_text)
# Returns: VerificationResult(passed=True, error_type=None, details=None)
```

**What we check**:
- Token-by-token equality (ignoring label tags)
- Character-level precision (spaces, punctuation preserved)
- No insertions, deletions, or modifications

**Common failures**:
- LLM paraphrases: `"the Court"` → `"the court"` (capitalization changed)
- LLM adds explanations: `"...<end> Note: This is a citation."`
- LLM "fixes" grammar: `"ain't"` → `"isn't"`

#### 3.2 Consistency Check

**Goal**: Ensure tags are well-formed and properly nested

```python
from utils.verification_utils import check_consistency

tokens = ['<decision>', 'R', '.', ' ', 'v', '.', ' ', 'Smith', '</decision>']

result = check_consistency(tokens)
# Returns: VerificationResult(passed=True, error_type=None, details=None)
```

**What we check**:
- Every opening tag has a matching closing tag
- Tags are properly nested (no overlapping spans)
- No orphaned or malformed tags

**Common failures**:
- Unclosed tags: `<decision>R. v. Smith` (missing `</decision>`)
- Wrong closing tag: `<decision>...</legislation>`
- Overlapping spans: `<decision>R. v. <legislation>Smith</decision></legislation>`

#### 3.3 Label Scheme Validation

**Goal**: Ensure annotations conform to our predefined schema

```python
from utils.verification_utils import check_label_scheme

tokens = ['<auto_label labelname="decision">', 'R', '.', ' ', 'v', '.', ' ', 'Smith', '</auto_label>']
allowed_labels = ["decision", "legislation", "secondary sources"]

result = check_label_scheme(tokens, allowed_labels)
# Returns: VerificationResult(passed=True, error_type=None, details=None)
```

**What we check**:
- Label names are in allowed set
- Attributes conform to schema (e.g., `titletype`, `fragmentid`)
- Tag structure matches expected format

**Label hierarchy** (from `ressources/label_scheme.json`):
- **legislation**: Statutes, regulations, constitutional documents
  - Sublabels: `title`, `reference`, `fragment`
  - Attributes: `docid`, `uri`, `titletype`, `fragmentid`
- **decision**: Case law, judicial decisions
  - Sublabels: `title`, `reference`, `fragment`
  - Attributes: `docid`, `uri`, `fragmentid`
- **secondary sources**: Treatises, articles, dictionaries
  - Sublabels: `title`, `reference`, `fragment`
  - Attributes: `docid`, `fragmentid`

### 4. Fallback Strategy

If verification fails, we **retry with an adjusted prompt** that emphasizes the error type:

```python
from process_chunks import process_single_chunk

result, status, error = process_single_chunk(
    model=gpt_model,
    chunk=text_chunk,
    max_fallback_attempts=2  # Try up to 2 additional times per error type
)

# If first attempt fails hallucination check:
# → Retry with prompt: "CRITICAL: Copy text EXACTLY. Do not modify any words."

# If retry fails consistency check:
# → Retry with prompt: "Ensure every <tag> has a matching </tag>."
```

**Fallback triggers**:
- `hallucination_fail` → Emphasize verbatim copying
- `consistency_fail` → Emphasize balanced tags
- `label_scheme_fail` → Provide explicit label list


---

## Usage

### Basic Pipeline

```python
from models import GPTModel
from process_chunks import process_chunks

# 1. Initialize model
model = GPTModel(model_name="gpt-5", temperature=0.1)

# 2. Prepare text chunks (semantically-aware splitting)
from utils.chunker_utils import chunk_document
chunks = chunk_document(document_html, max_tokens=1500)

# 3. Configure annotation settings
label_config = {
    "use_simplified": True,  # Use <decision> instead of <auto_label labelname="decision">
    "keep_attributes": ["labelname", "docid", "fragmentid"],
    "switch_type": True  # Convert back to full format after processing
}

# 4. Process chunks
processed_chunks = process_chunks(
    model=model,
    token_chunks=chunks,
    process_prompt_path="utils/prompts/extraction_prompt.txt",
    label_config=label_config,
    few_shot_examples=examples,  # 3-5 representative examples
    allowed_labels=["legislation", "decision", "secondary sources"],
    output_dir="./output",
    filename="document_name",
    max_fallback_attempts=2
)

# 5. Reconstruct annotated document
from utils.html_utils import reconstruct_document
annotated_html = reconstruct_document(processed_chunks, original_html)
```

### Monitoring Processing

```python
# Access detailed processing history
history = processed_chunks['history']

print(history.summary())
# Output:
# {
#   "total": 25,
#   "success": 20,
#   "hallucination_fail": 3,
#   "consistency_fail": 2,
#   "label_scheme_fail": 0
# }

# Inspect failures
for entry in history.entries:
    if entry['status'] != 'Success':
        print(f"Chunk {entry['chunk_idx']}: {entry['error_details']}")
```

---

## Module Structure

```
llm_based_annotation/
├── main.py                          # Entry point
├── models.py                        # GPT model wrapper
├── process_chunks.py                # Main orchestration logic
│
├── utils/
│   ├── tokenizer_utils.py           # HTML-aware tokenization
│   ├── processing_utils.py          # Levenshtein alignment & correction
│   ├── verification_utils.py        # Three-level verification
│   ├── post_processing_utils.py     # Output transformation
│   │
│   ├── chunker_utils.py             # Document chunking
│   ├── prompt_utils.py              # Prompt construction
│   ├── few_shot_utils.py            # Example management
│   ├── html_utils.py                # HTML manipulation
│   └── html_cleaner.py              # HTML preprocessing
│
├── main_label_extraction.ipynb              # Workflow: Extract main labels
├── main_sublabel_extraction.ipynb           # Workflow: Extract sublabels
└── main_label_extraction_full_doc.ipynb     # Workflow: Full document processing
```

---

## Key Functions

### Tokenization

```python
from utils.tokenizer_utils import tokenize_html, detokenize

# Tokenize preserving HTML structure
tokens = tokenize_html("<decision>R. v. Smith</decision>")
# ['<decision>', 'R', '.', ' ', 'v', '.', ' ', 'Smith', '</decision>']

# Reconstruct from tokens
text = detokenize(tokens)
# "<decision>R. v. Smith</decision>"
```

### Error Correction

```python
from utils.processing_utils import apply_operations

# Correct LLM mistakes while preserving labels
corrected = apply_operations(
    original_tokens=input_tokens,
    processed_tokens=llm_output_tokens
)
```

### Verification

```python
from utils.verification_utils import verify_processed_chunk

result = verify_processed_chunk(
    original_tokens=input_tokens,
    processed_tokens=corrected_tokens,
    allowed_labels=["decision", "legislation", "secondary sources"]
)

if not result.passed:
    print(f"Verification failed: {result.error_type}")
    print(f"Details: {result.details}")
```

---

## Prompt Engineering

Our prompts emphasize three key instructions:

### 1. Verbatim Copying
```
CRITICAL INSTRUCTION: Copy the input text EXACTLY as provided. 
Do not modify, paraphrase, or "correct" any content.
Preserve all punctuation, spacing, and capitalization PRECISELY.
```

### 2. Label Insertion Only
```
Your ONLY modification should be inserting <label></label> tags around legal authorities.
DO NOT add explanations, notes, or any other text.
```

### 3. Structured Output Markers
```
Begin your output with <start> and end with <end>.
This allows us to extract your annotations reliably.

Example:
<start>
The court in <decision>R. v. Smith</decision> held that...
<end>
```

See [utils/prompts/](utils/prompts/) for full prompt templates.
