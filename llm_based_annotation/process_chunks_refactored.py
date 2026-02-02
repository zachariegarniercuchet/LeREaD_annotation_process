"""
Refactored chunk processing with separation of concerns:
- Post-processing: transformations (extract tokens, format conversion)
- Verification: validation checks (hallucination, consistency, label scheme)
- Error correction: fallback strategies when verification fails
"""

import os
import json
from tqdm import tqdm
from typing import List, Tuple, Optional

from utils.tokenizer_utils import tokenize, decode
from llm_based_annotation.utils.document_level_post_processing_utils import apply_post_processing_transforms
from utils.verification_utils import verify_processed_chunk, VerificationResult
from utils.processing_utils import distance_lists_auto_label, apply_operations_safe
from utils.prompt_utils import get_prompt_processing, get_prompt_fallback_hallucination, get_prompt_fallback_consistency


class ProcessingHistory:
    """Track processing history for debugging and analysis."""
    
    def __init__(self):
        self.entries = []
    
    def add(self, status: str, chunk_idx: int, raw_output: str, error_details: str = None):
        """Add an entry to the history."""
        self.entries.append({
            "status": status,
            "chunk_idx": chunk_idx,
            "raw_output": raw_output,
            "error_details": error_details
        })
    
    def save(self, output_dir: str, filename: str):
        """Save history to JSON file."""
        if not output_dir or not filename:
            return
        
        json_path = os.path.join(output_dir, f"history_{filename}.json")
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(self.entries, f, ensure_ascii=False, indent=4)
            print(f"   ✓ Processing history saved to: {json_path}")
        except Exception as e:
            print(f"   ✗ Error saving history: {e}")
    
    def summary(self) -> dict:
        """Get summary statistics."""
        statuses = [entry["status"] for entry in self.entries]
        return {
            "total": len(statuses),
            "success": statuses.count("Success"),
            "hallucination_fail": statuses.count("Hallucination Fail"),
            "consistency_fail": statuses.count("Consistency Fail"),
            "label_scheme_fail": statuses.count("Label Scheme Fail"),
            "double_hallucination_fail": statuses.count("Double Hallucination Fail"),
            "double_consistency_fail": statuses.count("Double Consistency Fail")
        }


def process_single_chunk(
    model,
    chunk: list,
    system_prompt: str,
    user_prompt_template: str,
    label_config: dict,
    allowed_labels: Optional[List[str]] = None,
    max_fallback_attempts: int = 1
) -> Tuple[list, str, str]:
    """
    Process a single chunk with post-processing, verification, and fallback.
    
    Pipeline:
    1. Prepare input (clean chunk)
    2. Generate LLM output
    3. Post-process output (extract, transform)
    4. Verify output (hallucination, consistency, label scheme)
    5. If verification fails, apply error correction and retry
    6. If still fails after max attempts, return original chunk
    
    Args:
        model: LLM model instance
        chunk: List of tokens to process
        system_prompt: System prompt for LLM
        user_prompt_template: User prompt template with {text} placeholder
        label_config: Configuration for label transformations
        allowed_labels: List of allowed label names for scheme validation
        max_fallback_attempts: Maximum number of fallback attempts per error type
    
    Returns:
        Tuple of (processed_tokens, status, error_details)
        - processed_tokens: Successfully processed tokens or original chunk if failed
        - status: "Success", "Hallucination Fail", "Consistency Fail", etc.
        - error_details: Description of error if failed, None if success
    """
    from utils.few_shot_utils import _prepare_label_tokens
    
    # ------ 1. PREPARE INPUT ------
    cleaned_chunk = _prepare_label_tokens(
        chunk,
        label_config={
            "keep_attributes": ["labelname"],
            "switch_type": False,
            "use_simplified": False
        }
    )
    text = decode(cleaned_chunk)
    
    # ------ 2. GENERATE LLM OUTPUT ------
    user_prompt = user_prompt_template.format(text=text)
    raw_output = model.generate(
        system_prompt=system_prompt,
        user_prompt=user_prompt
    )
    
    # ------ 3. POST-PROCESS OUTPUT ------
    try:
        processed_tokens = apply_post_processing_transforms(
            raw_output=raw_output,
            use_simplified=label_config.get("use_simplified", False),
            label_type='auto_label'
        )
    except Exception as e:
        return chunk, "Post-processing Error", f"Failed to post-process: {str(e)}"
    
    # ------ 4. APPLY ERROR CORRECTION (BEFORE VERIFICATION) ------
    # This aligns tokens to handle minor discrepancies
    _, operations = distance_lists_auto_label(cleaned_chunk, processed_tokens)
    processed_tokens_corrected = apply_operations_safe(processed_tokens, operations)
    
    # ------ 5. VERIFY OUTPUT ------
    verification = verify_processed_chunk(
        original_tokens=cleaned_chunk,
        processed_tokens=processed_tokens_corrected,
        allowed_labels=allowed_labels,
        check_scheme=True
    )
    
    if verification.passed:
        return processed_tokens_corrected, "Success", None
    
    # ------ 6. HANDLE VERIFICATION FAILURES WITH FALLBACK ------
    print(f"   ⚠ Verification failed: {verification.error_type}")
    print(f"   Details: {verification.details}")
    
    # Determine which fallback to use
    if verification.error_type == "hallucination":
        fallback_system, fallback_user_template = get_prompt_fallback_hallucination()
        failure_type = "Hallucination Fail"
    elif verification.error_type == "consistency":
        fallback_system, fallback_user_template = get_prompt_fallback_consistency()
        failure_type = "Consistency Fail"
    elif verification.error_type == "label_scheme":
        # For label scheme errors, we could use a specialized fallback
        # For now, treat similar to consistency issues
        fallback_system, fallback_user_template = get_prompt_fallback_consistency()
        failure_type = "Label Scheme Fail"
    else:
        return chunk, failure_type, verification.details
    
    # Attempt fallback correction
    for attempt in range(max_fallback_attempts):
        print(f"   → Fallback attempt {attempt + 1}/{max_fallback_attempts}...")
        
        fallback_user_prompt = fallback_user_template.format(
            text_pair=f"Original Text:\n\n<<<ORIGINAL_TEXT>>>{text}<<<END_ORIGINAL_TEXT>>>\n\n"
                     f"Annotated Text with errors:\n\n<<<ANNOTATED_TEXT>>>{raw_output}<<<END_ANNOTATED_TEXT>>>\n\n"
        )
        
        raw_output = model.generate(
            system_prompt=fallback_system,
            user_prompt=fallback_user_prompt
        )
        
        # Post-process fallback output
        try:
            processed_tokens = apply_post_processing_transforms(
                raw_output=raw_output,
                use_simplified=label_config.get("use_simplified", False),
                label_type='auto_label'
            )
        except Exception as e:
            print(f"   ✗ Fallback post-processing failed: {e}")
            continue
        
        # Apply error correction
        _, operations = distance_lists_auto_label(cleaned_chunk, processed_tokens)
        processed_tokens_corrected = apply_operations_safe(processed_tokens, operations)
        
        # Verify again
        verification = verify_processed_chunk(
            original_tokens=cleaned_chunk,
            processed_tokens=processed_tokens_corrected,
            allowed_labels=allowed_labels,
            check_scheme=True
        )
        
        if verification.passed:
            print(f"   ✓ Fallback succeeded on attempt {attempt + 1}")
            return processed_tokens_corrected, f"Success (after {attempt + 1} fallback)", None
    
    # All attempts failed
    print(f"   ✗ All fallback attempts failed")
    return chunk, f"Double {failure_type}", verification.details


def process_chunks(
    model,
    token_chunks: list,
    process_prompt_path: str,
    label_config: dict,
    few_shot_examples: Optional[list] = None,
    allowed_labels: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    filename: Optional[str] = None,
    max_fallback_attempts: int = 1
) -> list:
    """
    Process multiple chunks using AI model with verification and fallback.
    
    This is the main entry point for chunk processing. It coordinates:
    - LLM generation
    - Post-processing transformations
    - Verification checks
    - Error correction and fallbacks
    - History tracking
    
    Args:
        model: LLM model instance
        token_chunks: List of token chunks to process
        process_prompt_path: Path to the main processing prompt file
        label_config: Configuration for label transformations
        few_shot_examples: Optional list of (input, output) examples
        allowed_labels: Optional list of allowed label names
        output_dir: Optional directory to save outputs
        filename: Optional filename prefix for outputs
        max_fallback_attempts: Maximum fallback attempts per error type
    
    Returns:
        List of processed token chunks
    """
    # Load prompts
    system_prompt, user_prompt_template = get_prompt_processing(
        process_prompt_path,
        few_shot_examples=few_shot_examples
    )
    
    # Initialize tracking
    processed_chunks = []
    history = ProcessingHistory()
    
    print(f"   ✓ Processing {len(token_chunks)} chunks with LLM...")
    print(f"   ✓ Using {len(few_shot_examples) if few_shot_examples else 0} few-shot examples")
    if allowed_labels:
        print(f"   ✓ Label scheme validation enabled with {len(allowed_labels)} allowed labels")
    
    # Process each chunk
    for idx, chunk in enumerate(tqdm(token_chunks, desc="Processing chunks")):
        processed_tokens, status, error_details = process_single_chunk(
            model=model,
            chunk=chunk,
            system_prompt=system_prompt,
            user_prompt_template=user_prompt_template,
            label_config=label_config,
            allowed_labels=allowed_labels,
            max_fallback_attempts=max_fallback_attempts
        )
        
        processed_chunks.append(processed_tokens)
        history.add(status, idx, decode(processed_tokens), error_details)
        
        if status != "Success" and not status.startswith("Success (after"):
            print(f"   ⚠ Chunk {idx} failed: {status}")
    
    # Save history and results
    history.save(output_dir, filename)
    
    # Print summary
    summary = history.summary()
    print(f"\n   ✓ Processing completed:")
    print(f"      - Total chunks: {summary['total']}")
    print(f"      - Successful: {summary['success']}")
    print(f"      - Failed: {summary['total'] - summary['success']}")
    
    if output_dir and filename:
        json_path = os.path.join(output_dir, f"processed_chunks_{filename}.json")
        try:
            # Convert token lists to strings for JSON serialization
            chunks_as_strings = [decode(chunk) for chunk in processed_chunks]
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(chunks_as_strings, f, indent=4, ensure_ascii=False)
            print(f"   ✓ Processed chunks saved to: {json_path}")
        except Exception as e:
            print(f"   ✗ Error saving processed chunks: {e}")
    
    return processed_chunks
