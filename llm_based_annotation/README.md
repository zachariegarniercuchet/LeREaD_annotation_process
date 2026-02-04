# Refactored Chunk Processing Architecture

## Overview

The chunk processing pipeline has been reorganized into three distinct concerns:

1. **Post-processing** (`utils/post_processing_utils.py`): Transforms raw LLM output into the desired format
2. **Verification** (`utils/verification_utils.py`): Validates processed chunks without modification
3. **Processing** (`process_chunks_refactored.py`): Orchestrates the entire pipeline with fallback strategies

## Architecture

```
┌─────────────────┐
│  Raw LLM Output │
└────────┬────────┘
         │
         ▼
┌─────────────────────┐
│  POST-PROCESSING    │
│  - Extract <start>  │
│  - Convert formats  │
│  - Normalize tags   │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  ERROR CORRECTION   │
│  - Token alignment  │
│  - Operations safe  │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  VERIFICATION       │
│  1. Hallucination   │
│  2. Consistency     │
│  3. Label Scheme    │
└────────┬────────────┘
         │
    Pass │  Fail
         ▼    │
    ┌─────────┴─────────┐
    │                   │
    ▼                   ▼
 Success          ┌──────────┐
                  │ FALLBACK │
                  └──────────┘
```

## Module Details

### 1. Post-Processing (`utils/post_processing_utils.py`)

**Purpose**: Transform raw LLM output into standard format

**Functions**:
- `extract_start_end_tokens(tokens)`: Extract content between `<start>` and `<end>` markers
- `simplified_to_normal_form(tokens, label_type)`: Convert simplified tags to full format
  - `<decision>` → `<auto_label labelname="decision">`
- `apply_post_processing_transforms(raw_output, use_simplified, label_type)`: Full pipeline

**Example**:
```python
from utils import apply_post_processing_transforms

raw_llm_output = "<start><decision>Smith v. Jones</decision><end>"
tokens = apply_post_processing_transforms(
    raw_output=raw_llm_output,
    use_simplified=True,
    label_type='auto_label'
)
# Result: ['<auto_label labelname="decision">', 'Smith', 'v', '.', 'Jones', '</auto_label>']
```

### 2. Verification (`utils/verification_utils.py`)

**Purpose**: Validate processed chunks against multiple criteria

**Functions**:
- `check_hallucination(original, processed)`: Verify text content unchanged
- `check_consistency(tokens)`: Verify tags properly balanced
- `check_label_scheme(tokens, allowed_labels)`: Verify labels conform to schema
- `verify_processed_chunk(original, processed, allowed_labels)`: Run all checks

**Returns**: `VerificationResult` object with:
- `passed`: Boolean indicating success/failure
- `error_type`: "hallucination", "consistency", or "label_scheme"
- `details`: Human-readable error description
- `tokens`: Problematic tokens for debugging

**Example**:
```python
from utils import verify_processed_chunk

result = verify_processed_chunk(
    original_tokens=cleaned_chunk,
    processed_tokens=llm_output_tokens,
    allowed_labels=["decision", "legislation", "secondary sources"]
)

if result.passed:
    print("✓ Verification passed")
else:
    print(f"✗ {result.error_type}: {result.details}")
```

**Label Scheme Validation**:
Based on `annotation/label schemes.html`, validates:
- **legislation**: Can have `titletype`, `fragmentid` attributes
- **decision**: Can have `titletype`, `fragmentid` attributes  
- **secondary sources**: Can have `titletype`, `fragmentid` attributes
- **unclassified**: No specific attributes

### 3. Refactored Processing (`process_chunks_refactored.py`)

**Purpose**: Main orchestration with error handling and fallback

**Key Features**:
- Automatic fallback on verification failures
- Configurable max retry attempts
- Detailed history tracking
- Comprehensive error reporting

**Main Functions**:
- `process_single_chunk(...)`: Process one chunk with full pipeline
- `process_chunks(...)`: Process multiple chunks with progress tracking
- `ProcessingHistory`: Track and analyze processing results

**Usage**:
```python
from process_chunks_refactored import process_chunks

processed_chunks = process_chunks(
    model=gpt_model,
    token_chunks=chunks_to_process,
    process_prompt_path="path/to/prompt.txt",
    label_config={
        "use_simplified": True,
        "keep_attributes": ["labelname"],
        "switch_type": True
    },
    few_shot_examples=examples,
    allowed_labels=["decision", "legislation", "secondary sources"],
    output_dir="./output",
    filename="document_name",
    max_fallback_attempts=1  # Number of retries per error type
)
```

## Comparison: Old vs New

### Old Architecture (Mixed Concerns)

```python
def process_chunks(...):
    for chunk in chunks:
        # 1. Clean chunk
        cleaned = clean_tokens(...)
        
        # 2. Generate
        output = model.generate(...)
        
        # 3. Post-process (mixed with verification)
        tokens = tokenize(output)
        tokens = tokens[start:end]  # Extract
        tokens = convert_format(...)  # Transform
        
        # 4. Error correction
        corrected = apply_operations(...)
        
        # 5. Verify hallucination
        if not check_hallucination(...):
            # Fallback inline
            output = model.generate(fallback_prompt)
            tokens = tokenize(output)
            # ... repeat steps
        
        # 6. Verify consistency
        if not check_consistency(...):
            # Another fallback inline
            output = model.generate(fallback_prompt)
            # ... repeat steps
```

**Problems**:
- Mixed transformation and validation logic
- Repeated code for fallback attempts
- Hard to add new verification checks
- Difficult to test individual components
- No label scheme validation

### New Architecture (Separation of Concerns)

```python
def process_single_chunk(...):
    # 1. Clean chunk
    cleaned = prepare_input(chunk)
    
    # 2. Generate
    output = model.generate(...)
    
    # 3. Post-process (pure transformations)
    tokens = apply_post_processing_transforms(output, config)
    
    # 4. Error correction
    corrected = apply_operations(tokens)
    
    # 5. Verify (all checks in one place)
    result = verify_processed_chunk(
        original=cleaned,
        processed=corrected,
        allowed_labels=labels
    )
    
    if result.passed:
        return corrected, "Success"
    
    # 6. Fallback (generic retry logic)
    return attempt_fallback(
        error_type=result.error_type,
        max_attempts=max_retries
    )
```

**Benefits**:
- Clear separation: transform → verify → fallback
- Easy to add new verification types
- Reusable verification functions
- Testable components
- Label scheme validation included
- Better error messages

## Migration Guide

### Updating Existing Code

Replace old import:
```python
from utils import process_chunks
```

With new import:
```python
from process_chunks_refactored import process_chunks
```

The function signature is similar, but adds:
- `allowed_labels`: List of valid label names
- `max_fallback_attempts`: Control retry behavior

### Testing Individual Components

```python
# Test post-processing
from utils import apply_post_processing_transforms
tokens = apply_post_processing_transforms("<start>text<end>", True)

# Test verification
from utils import check_hallucination, check_consistency, check_label_scheme
hal_result = check_hallucination(original, processed)
cons_result = check_consistency(processed)
scheme_result = check_label_scheme(processed, ["decision", "legislation"])

# Test full pipeline on single chunk
from process_chunks_refactored import process_single_chunk
result, status, error = process_single_chunk(
    model=model,
    chunk=test_chunk,
    system_prompt=prompt,
    user_prompt_template=template,
    label_config=config,
    allowed_labels=labels
)
```

## History Tracking

The refactored version includes detailed history tracking:

```json
{
  "entries": [
    {
      "status": "Success",
      "chunk_idx": 0,
      "raw_output": "...",
      "error_details": null
    },
    {
      "status": "Hallucination Fail",
      "chunk_idx": 1,
      "raw_output": "...",
      "error_details": "First difference at position 42..."
    }
  ]
}
```

Summary statistics:
```python
history.summary()
# {
#   "total": 10,
#   "success": 8,
#   "hallucination_fail": 1,
#   "consistency_fail": 1,
#   ...
# }
```

## Future Enhancements

1. **Additional Verification Checks**:
   - Cross-reference validation (citation formats)
   - Semantic consistency (label hierarchy rules)
   - Fragment ID validation

2. **Smarter Fallback Strategies**:
   - Different fallback prompts per error type
   - Adaptive retry limits based on error severity
   - Ensemble strategies (multiple model attempts)

3. **Performance Optimizations**:
   - Batch verification for multiple chunks
   - Parallel processing with async/await
   - Caching for repeated patterns

4. **Better Error Reporting**:
   - Visual diff outputs
   - Confidence scores for corrections
   - Suggested manual review points

## Files Modified/Created

### New Files:
- `utils/verification_utils.py`: Verification functions and VerificationResult class
- `process_chunks_refactored.py`: Refactored main processing logic
- `REFACTORING_README.md`: This documentation

### Modified Files:
- `utils/post_processing_utils.py`: Added transformation functions
- `utils/__init__.py`: Updated exports

### Unchanged:
- Original `process_chunks.py` (in `main_test.ipynb`) - kept for reference
- Core utilities: `tokenizer_utils.py`, `html_utils.py`, `processing_utils.py`
