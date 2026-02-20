import os
import json
from tqdm import tqdm
from typing import List, Tuple, Optional

from utils_extraction import decode, apply_post_processing_transforms
from utils_extraction.prompt_utils import get_prompt_processing, get_prompt_sublabel_extraction
from utils_extraction.levenshtein_utils import distance_lists_auto_label, apply_operations_safe
from utils_extraction.few_shot_utils import prepare_label_tokens, get_list_of_mention
from utils_extraction.verification_utils import verify_processed_chunk


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




# ==============================================================================================================================
# ============ BELOW ARE MAIN INTERNAL HELPER FUNCTIONS FOR CHUNCK PROCESSING (STAGE 1 OF THE EXTRACTION PROCESS)  =============
# ==============================================================================================================================

def process_single_chunk(
    model,
    chunk: list,
    system_prompt: str,
    user_prompt_template: str,
    label_config: dict,
    allowed_labels: Optional[List[str]] = None,
) -> Tuple[list, str, str]:
    """
    Process a single chunk with post-processing, verification, and fallback.
    
    Pipeline:
    1. Prepare input (clean chunk)
    2. Generate LLM output
    3. Post-process output (extract, transform)
    4. Verify output (hallucination, consistency, label scheme)
    
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
    # ------ 1. PREPARE INPUT ------
    cleaned_chunk = prepare_label_tokens(
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

    #print(f"   → Raw LLM output for chunk: {raw_output}...")  
    
    # ------ 3. POST-PROCESS OUTPUT ------
    try:
        processed_tokens = apply_post_processing_transforms(
            raw_output=raw_output,
            use_simplified=label_config.get("use_simplified", False),
            label_type='auto_label'
        )
    except Exception as e:
        return chunk, "Post-processing Error", f"Failed to post-process: {str(e)}"
    
    #print(f"   → Processed tokens before verification: {decode(processed_tokens)}")
    
    # ------ 4. APPLY ERROR CORRECTION (BEFORE VERIFICATION) ------
    # This aligns tokens to handle minor discrepancies
    _, operations = distance_lists_auto_label(cleaned_chunk, processed_tokens)
    processed_tokens_corrected = apply_operations_safe(processed_tokens, operations)
    
    # ------ 5. VERIFY OUTPUT ------

    #print(f"Processed tokens after correction: {decode(processed_tokens_corrected)}")
    verification = verify_processed_chunk(
        original_tokens=cleaned_chunk,
        processed_tokens=processed_tokens_corrected,
        allowed_labels=allowed_labels,
        check_scheme=True
    )
    
    if verification.passed:
        return processed_tokens_corrected, "Success", None
    
    # ------ 6. HANDLE VERIFICATION FAILURES ------
    print(f"   ⚠ Verification failed: {verification.error_type}")
    print(f"   Details: {verification.details}")
    
    if verification.error_type == "hallucination":
        failure_type = "Hallucination Fail"
    elif verification.error_type == "consistency":
        failure_type = "Consistency Fail"
    elif verification.error_type == "label_scheme":
        failure_type = "Label Scheme Fail"
    
    
    return chunk, failure_type, verification.details




# ============================================================
# =========== MAIN FUNCTION FOR CHUNCK PROCESSING  ===========
# ============================================================


def process_chunks(
    model,
    token_chunks: list,
    process_prompt_path: str,
    label_config: dict,
    few_shot_examples: Optional[list] = None,
    allowed_labels: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    filename: Optional[str] = None,
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
        prompt_path=process_prompt_path,
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








# ==============================================================================================================================
# =========== BELOW ARE MAIN INTERNAL HELPER FUNCTIONS FOR SUBLABELS PROCESSING (STAGE 2 OF THE EXTRACTION PROCESS)  ===========
# ==============================================================================================================================


def _build_processing_segments(tokens, parent_mentions):
    """
    Build a list of segments alternating between:
      - non-processable token spans
      - processable mention spans

    Returns:
        List[dict]: each dict has:
            - "process": bool
            - "tokens": list
            - "meta": optional mention metadata
    """
    segments = []
    cursor = 0

    for html_label, start_idx, end_idx in parent_mentions:
        # Non-processable tokens before the mention
        if cursor < start_idx:
            segments.append({
                "process": False,
                "tokens": tokens[cursor:start_idx]
            })

        # The mention itself (processable)
        segments.append({
            "process": True,
            "tokens": tokens[start_idx:end_idx + 1],
            "meta": {
                "label": html_label,
                "start": start_idx,
                "end": end_idx
            }
        })

        cursor = end_idx + 1

    # Trailing non-processable tokens
    if cursor < len(tokens):
        segments.append({
            "process": False,
            "tokens": tokens[cursor:]
        })

    return segments


def process_single_mention(
    model,
    mention: list,
    system_prompt: str,
    user_prompt_template: str,
    sublabel_config: dict,
    allowed_labels: list = None,
):
    """
    Process a single parent mention to extract sublabels.
    
    Pipeline:
    1. Prepare input (simplified form, keep right attributes)
    2. Decode mention to text
    3. Generate LLM output
    4. Post-process output (TODO: extract, transform)
    5. Verify output (hallucination, consistency, label scheme)
    6. If verification fails, apply fallback
    7. If still fails, return original mention
    
    Args:
        model: LLM model instance
        mention: List of tokens for the parent mention
        system_prompt: System prompt for LLM
        user_prompt_template: User prompt template with {text} placeholder
        sublabel_config: Configuration for sublabel transformations
        allowed_labels: List of allowed sublabel names
        max_fallback_attempts: Maximum number of fallback attempts
    
    Returns:
        Tuple of (processed_tokens, status, error_details)
    """
    # ------ 1. PREPARE INPUT ------
    input_config = sublabel_config.copy()
    input_config["keep_labels"] = None # Keep everything
    input_config["switch_type"] = False # Keep original types for input
    prepared_mention = prepare_label_tokens(mention,
        label_config={
            "switch_type": False,
            "use_simplified": sublabel_config.get("use_simplified", False),
            "keep_attributes": sublabel_config.get("keep_attributes", None)
        }
    )

    
    # ------ 2. DECODE TO TEXT ------
    text = decode(prepared_mention)
    
    # ------ 3. GENERATE LLM OUTPUT ------
    user_prompt = user_prompt_template.format(text=text)
    raw_output = model.generate(
        system_prompt=system_prompt,
        user_prompt=user_prompt
    )
    
    
    # ------ 4. POST-PROCESS OUTPUT ------
    try:
        processed_tokens = apply_post_processing_transforms(
            raw_output=raw_output,
            use_simplified=sublabel_config.get("use_simplified", False), # Did we use simplified form ?
            label_type='auto_label'
        )
    except Exception as e:
        return mention, "Post-processing Error", f"Failed to post-process: {str(e)}"
    
    
    # ------ 5. APPLY ERROR CORRECTION ------
    # This aligns tokens to handle minor discrepancies
    cleaned_input_mention_token = prepare_label_tokens(
        mention,
        label_config={
            "keep_attributes": sublabel_config.get("keep_attributes", None),
            "switch_type": False,
            "use_simplified": False
        }
    ) # Just remove non-kept attributes for alignment
    _, operations = distance_lists_auto_label(cleaned_input_mention_token, processed_tokens)
    processed_tokens_corrected = apply_operations_safe(processed_tokens, operations)
    


    
    # ------ 6. VERIFY OUTPUT ------
    verification = verify_processed_chunk(
        original_tokens=mention,
        processed_tokens=processed_tokens_corrected,
        allowed_labels=allowed_labels,
        check_scheme=True
    )
    
    if verification.passed:
        return processed_tokens_corrected, "Success", None
    
    else :
        return mention, "Verification Failed", verification.details



# ============================================================
# ========== MAIN FUNCTION FOR SUBLABEL PROCESSING  ==========
# ============================================================
def process_labels(
    model,
    tokens: list,
    sublabel_config: dict,
    few_shot_examples: list = None,
    prompt_path: str = None,
    output_dir: str = None,
    filename: str = None,
):
    """
    Process tokens to extract sublabels from parent mentions.
    
    This function extracts sublabels (e.g., title) from already annotated
    parent labels (e.g., decision, legislation, secondary sources).
    
    Args:
        model: LLM model instance
        tokens: List of tokens containing parent labels
        sublabel_config: Configuration dict with:
            - parent: List of parent label names
            - keep_labels: List of sublabel names to extract
            - keep_attributes, switch_type, use_simplified: Transform options
        few_shot_examples: Optional list of (input, output) examples
        prompt_path: Optional path to prompt templates
        output_dir: Optional directory to save outputs
        filename: Optional filename prefix for outputs
        max_fallback_attempts: Maximum fallback attempts per error type
    
    Returns:
        List of processed tokens (flat list, not chunked)
    """
    # ------ 1. GET PROMPT ------
    system_prompt, user_prompt_template = get_prompt_sublabel_extraction(
        prompt_path=prompt_path,
        keep_labels=sublabel_config["new_labels"],
        few_shot_examples=few_shot_examples
    )
    
    # ------ 2. GET LIST OF AUTO_LABEL MENTIONS TO PROCESS ------
    # We're looking for auto_label parents (already extracted from previous step)
    parent_mentions = get_list_of_mention(
        tokens=tokens,
        keep_labels=sublabel_config["parent"],
        label_type="auto_label"  # Process auto_labels from parent extraction
    )
    
    print(f"   ✓ Found {len(parent_mentions)} parent mentions to process")
    
    # ------ 3. INITIALIZE TRACKING ------
    history = ProcessingHistory()
    
    # Create a copy of tokens to modify
    processed_tokens = tokens.copy()
    
    # ------ 4. BUILD SEGMENTS ------
    segments = _build_processing_segments(tokens, parent_mentions)

    print(f"   ✓ Built {len(segments)} token segments "
        f"({sum(s['process'] for s in segments)} to process)")

    # ------ 5. PROCESS EACH PROCESSABLE SEGMENT ------
    for idx, segment in enumerate(tqdm(segments, desc="Processing mentions")):
        if not segment["process"]:
            continue

        mention = segment["tokens"]
        html_label = segment["meta"]["label"]





        processed_mention, status, error_details = process_single_mention(
            model=model,
            mention=mention,
            system_prompt=system_prompt,
            user_prompt_template=user_prompt_template,
            sublabel_config=sublabel_config,
            allowed_labels=sublabel_config["new_labels"] + sublabel_config["already_labeled"] + sublabel_config["parent"],
        )

        # Replace the entire segment safely
        segment["tokens"] = processed_mention

        # Track history
        history.add(
            status,
            idx,
            decode(processed_mention),
            error_details
        )

        if status != "Success" and not status.startswith("Success (after"):
            print(f"   ⚠ Segment {idx} ({html_label.name}) failed: {status} \n Details: {error_details}")

    
    # ------ 6. SAVE HISTORY AND RESULTS ------
    if output_dir and filename:
        history.save(output_dir, f"{filename}_sublabel")
        
        # Save processed tokens
        json_path = os.path.join(output_dir, f"processed_sublabels_{filename}.json")
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(decode(processed_tokens), f, indent=4, ensure_ascii=False)
            print(f"   ✓ Processed tokens saved to: {json_path}")
        except Exception as e:
            print(f"   ✗ Error saving processed tokens: {e}")
    
    # ------ 7. PRINT SUMMARY ------
    summary = history.summary()
    print(f"\n   ✓ Sublabel extraction completed:")
    print(f"      - Total mentions: {summary['total']}")
    print(f"      - Successful: {summary['success']}")
    print(f"      - Failed: {summary['total'] - summary['success']}")



    # ------ 8. FLATTEN SEGMENTS ------
    processed_tokens = [
        token
        for segment in segments
        for token in segment["tokens"]
    ]
    
    return processed_tokens