"""
Transform Reflex annotations to autolabel format and compare with human annotations.

This script provides two main functionalities:
1. Transform Reflex HTML annotations to simple <auto_label labelname="reflex"> tags
2. Compare Reflex spans with human annotations (SPAN-LEVEL, LABEL-AGNOSTIC)

KEY CONCEPT:
------------
Reflex only identifies SPAN LOCATIONS (legal citations) but doesn't classify them
into specific types (decision, legislation, secondary sources). Therefore:

- ALL Reflex annotations are labeled as "reflex"
- Human annotations have specific labels (decision, legislation, secondary sources, etc.)
- Comparison is done at SPAN LEVEL with EXACT text matching
- Context is used to disambiguate identical spans at different locations
- Only compares against human annotations of type: decision, legislation, secondary sources

EXACT MATCHING EXAMPLES:
- ✅ "Charter" matches "Charter" (exact)
- ❌ "Charter" does NOT match "Charter of Rights" (different)
- ❌ "s. 8" does NOT match "s. 8 of Charter" (different)
- If "Charter" appears twice, context determines which instance

LENIENT MATCHING EXAMPLES (for error analysis):
- ✅ "Charter of Rights" ≈ "Charter of Rights and Freedoms" (50%+ similarity)
- ✅ "s. 8" ≈ "s. 8 of Charter" (50%+ overlap)
- These partial matches help identify Reflex boundary errors

USAGE:
------

METHOD 1: Transform and Compare (all-in-one)
    1. Edit the 'filename' variable in main() function
    2. Edit the 'human_file' path to match your human annotation file
    3. Run: python reflex.py

METHOD 2: Transform only
    1. Edit the 'filename' variable in main() function
    2. Comment out the comparison step in main()
    3. Run: python reflex.py

METHOD 3: Compare existing files (programmatic)
    from reflex import compare_reflex_vs_human
    
    # Strict mode (exact matches)
    results_strict = compare_reflex_vs_human(
        reflex_autolabel_file="path/to/file_autolabel.html",
        human_annotation_file="path/to/human_annotated.html",
        match_type='context_overlap',
        context_chars=200,
        strict=True  # Exact text matching
    )
    
    # Lenient mode (to see Reflex errors)
    results_lenient = compare_reflex_vs_human(
        reflex_autolabel_file="path/to/file_autolabel.html",
        human_annotation_file="path/to/human_annotated.html",
        match_type='context_overlap',
        context_chars=200,
        strict=False  # Similarity matching
    )
    
    # Access results
    f1_strict = results_strict['f1']
    f1_lenient = results_lenient['f1']

COMPARISON METRICS:
-------------------
The script runs TWO comparisons:

1. STRICT MODE - Shows actual performance with exact matches:
   - Precision: How many Reflex spans are exact matches
   - Recall: How many human annotations were exactly captured
   - F1 Score: Harmonic mean

2. LENIENT MODE - Shows where Reflex makes errors:
   - Higher precision/recall (includes partial matches)
   - Difference from strict = Reflex extraction errors
   - Helps identify over-extraction and under-extraction patterns

Match types:
- 'context': Exact context match (very strict)
- 'context_overlap': Similar context match (default, handles HTML differences)

MATCHING LOGIC:
---------------
TWO MODES AVAILABLE:

1. STRICT MODE (strict=True) - Exact Text Match:
   - Span texts must be EXACTLY identical (normalized for whitespace/case)
   - Context similarity >= 70% (only to disambiguate identical spans at different locations)
   - Examples:
     ✅ "Charter" = "Charter"
     ❌ "Charter" ≠ "Charter of Rights"
     ❌ "s. 8" ≠ "s. 8 of Charter"
   - Use this to see PRECISE matching performance

2. LENIENT MODE (strict=False) - Similarity Match:
   - Text similarity >= 50% (Jaccard) - allows partial matches
   - Context similarity >= 70%
   - Examples:
     ✅ "Charter of Rights" ≈ "Charter of Rights and Freedoms" (similar)
     ✅ "s. 8" ≈ "s. 8 of Charter" (overlap)
   - Use this to identify COMMON REFLEX ERRORS:
     * Over-extraction: Reflex captured too much text
     * Under-extraction: Reflex captured too little text
     * Boundary issues: Close but not exact matches

The script runs BOTH modes to provide complete analysis.

FILTERING:
----------
Human annotations are filtered to ONLY include:
- decision
- legislation  
- secondary sources

All other label types are excluded from comparison.

OUTPUT:
-------
1. Transformed file: original_filename_autolabel.html
2. STRICT comparison report (exact matches only)
3. LENIENT comparison report (partial matches - shows Reflex errors)
4. Summary comparing both modes

EXAMPLE FILES:
--------------
Reflex input:  data/Document_Échantillon_Initial/ronde_2/reflex/1989CanLII1415CITT.html
Autolabel output: data/Document_Échantillon_Initial/ronde_2/reflex/1989CanLII1415CITT_autolabel.html
Human annotation: data/Documents_Annotés/EG/1999CanLII7320_annotated_EG.html
"""

from bs4 import BeautifulSoup, NavigableString
from pathlib import Path
import sys
import os

# Import IAA functions for span extraction
sys.path.append(os.path.dirname(__file__))
from IAA import extract_spans_from_html


# Target classes and types for Reflex annotations
TARGET_CLASSES = ["reflex2-link", "reflex3-block", "reflex3-caselaw"]
TARGET_TYPES = ["a", "span"]


def find_reflex_elements(soup):
    """
    Find all Reflex annotation elements in the document.
    
    Args:
        soup: BeautifulSoup object
        
    Returns:
        List of Reflex elements
    """
    reflex_elements = soup.find_all(
        TARGET_TYPES,
        class_=lambda x: x in TARGET_CLASSES if x else False
    )
    return reflex_elements


def get_text_content(element):
    """
    Extract all text content from an element, including nested elements.
    
    Args:
        element: BeautifulSoup element
        
    Returns:
        String containing all text content
    """
    return element.get_text()


def transform_reflex_to_autolabel(html_content):
    """
    Transform Reflex annotations to autolabel format.
    
    Args:
        html_content: String containing HTML document
        
    Returns:
        Transformed HTML content as string
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Find all outermost reflex elements (we want to avoid processing nested ones multiple times)
    # We'll process reflex3-block and reflex2-link as they are typically the outermost containers
    reflex_blocks = soup.find_all(
        TARGET_TYPES,
        class_=lambda x: ('reflex3-block' in x or 'reflex2-link' in x) if x else False
    )
    
    # Track elements we've already processed to avoid duplicates
    processed = set()
    
    for element in reflex_blocks:
        # Skip if this element is inside another reflex block we've already processed
        if element in processed:
            continue
            
        # Check if this element is inside another reflex block
        parent_reflex = element.find_parent(
            TARGET_TYPES,
            class_=lambda x: ('reflex3-block' in x or 'reflex2-link' in x) if x else False
        )
        
        if parent_reflex and parent_reflex != element:
            # This is a nested reflex element, skip it
            continue
        
        # Extract text content
        text_content = get_text_content(element)
        
        # Create new autolabel tag
        new_tag = soup.new_tag('auto_label', labelname='reflex')
        new_tag.string = text_content
        
        # Replace the original element
        element.replace_with(new_tag)
        
        # Mark as processed
        processed.add(element)
    
    return str(soup)


def process_file(input_path, output_path=None):
    """
    Process an HTML file and transform Reflex annotations.
    
    Args:
        input_path: Path to input HTML file
        output_path: Path to output HTML file (optional, defaults to input_path with _autolabel suffix)
    """
    input_path = Path(input_path)
    
    # Read input file
    with open(input_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # Transform
    transformed_html = transform_reflex_to_autolabel(html_content)
    
    # Determine output path
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_autolabel{input_path.suffix}"
    else:
        output_path = Path(output_path)
    
    # Write output file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(transformed_html)
    
    print(f"Transformed file saved to: {output_path}")
    
    # Count reflex annotations found
    soup = BeautifulSoup(html_content, 'html.parser')
    reflex_count = len(soup.find_all(
        TARGET_TYPES,
        class_=lambda x: ('reflex3-block' in x or 'reflex2-link' in x) if x else False
    ))
    
    soup_transformed = BeautifulSoup(transformed_html, 'html.parser')
    autolabel_count = len(soup_transformed.find_all('auto_label', labelname='reflex'))
    
    print(f"Found {reflex_count} Reflex annotations")
    print(f"Created {autolabel_count} autolabel elements")


def span_match_ignore_label(span1, span2, match_type='context_overlap', threshold=0.7, strict=True):
    """
    Check if two spans match based on text + context.
    
    STRICT MODE (strict=True):
    - EXACT text match required (normalized for whitespace/case)
    - Context used for disambiguation
    - "Charter of Rights" ≠ "Charter of Rights and Freedoms"
    - "s. 8" ≠ "s. 8 of Charter"
    
    LENIENT MODE (strict=False):
    - Text similarity >= 50% (Jaccard similarity)
    - Context similarity >= 70%
    - Allows partial matches to identify common Reflex errors
    - "Charter of Rights" might match "Charter of Rights and Freedoms"
    
    Args:
        span1: First span
        span2: Second span
        match_type: 'context' for exact context, 'context_overlap' for similar context
        threshold: Similarity threshold for context_overlap (default: 0.7)
        strict: True for exact text match, False for similarity-based match
        
    Returns:
        Boolean indicating if spans match
    """
    if not span1.text or not span2.text:
        return False
    
    # Normalize texts
    text1 = ' '.join(span1.text.split()).lower().strip()
    text2 = ' '.join(span2.text.split()).lower().strip()
    
    if strict:
        # STRICT MODE: EXACT text match
        if text1 != text2:
            return False
        
        # Use context to distinguish identical spans in different locations
        if not span1.context_text or not span2.context_text:
            return True
        
        if match_type == 'context':
            context1 = ' '.join(span1.context_text.split()).lower()
            context2 = ' '.join(span2.context_text.split()).lower()
            return context1 == context2
        
        elif match_type == 'context_overlap':
            words1 = set(span1.context_text.lower().split())
            words2 = set(span2.context_text.lower().split())
            
            if not words1 or not words2:
                return True
            
            intersection = len(words1 & words2)
            union = len(words1 | words2)
            
            if union == 0:
                return True
            
            context_similarity = intersection / union
            return context_similarity >= threshold
    
    else:
        # LENIENT MODE: Similarity-based matching
        if not span1.context_text or not span2.context_text:
            return False
        
        # Calculate text similarity
        text_words1 = set(text1.split())
        text_words2 = set(text2.split())
        
        text_intersection = len(text_words1 & text_words2)
        text_union = len(text_words1 | text_words2)
        
        if text_union == 0:
            return False
        
        text_similarity = text_intersection / text_union
        
        # Require at least 50% text similarity
        if text_similarity < 0.5:
            return False
        
        # Check context similarity
        if match_type == 'context':
            context1 = ' '.join(span1.context_text.split()).lower()
            context2 = ' '.join(span2.context_text.split()).lower()
            return context1 == context2
        
        elif match_type == 'context_overlap':
            words1 = set(span1.context_text.lower().split())
            words2 = set(span2.context_text.lower().split())
            
            if not words1 or not words2:
                return False
            
            intersection = len(words1 & words2)
            union = len(words1 | words2)
            
            if union == 0:
                return False
            
            context_similarity = intersection / union
            return context_similarity >= threshold
    
    return False


def compare_reflex_vs_human(reflex_autolabel_file: str, human_annotation_file: str, 
                           match_type: str = 'context_overlap', context_chars: int = 200,
                           strict: bool = True, show_matches: bool = False, max_matches_to_show: int = 10):
    """
    Compare Reflex auto-annotations with human annotations at span level ONLY.
    
    This function:
    1. Extracts ALL spans from the Reflex-transformed autolabel file
    2. Extracts ONLY decision, legislation, and secondary sources spans from human annotations
    3. Calculates metrics based on span position matching (ignoring label names)
    4. Reports overall precision, recall, F1 score
    
    NOTE: Reflex only outputs span locations (all labeled "reflex"), not specific label types.
    Therefore, we compare based on whether spans overlap at the same document location,
    regardless of what label name the human annotator assigned.
    
    MATCHING CRITERIA (STRICT vs LENIENT):
    
    STRICT MODE (strict=True) - Exact matching:
    - Span text must be EXACTLY the same (normalized for whitespace/case)
    - Context similarity >= 70%: Used to distinguish identical texts at different locations
    - NO partial matches (e.g., "s. 8" will not match "s. 8 of Charter")
    - Use this to see PRECISE matches
    
    LENIENT MODE (strict=False) - Similarity matching:
    - Text similarity >= 50% (allows partial matches)
    - Context similarity >= 70%
    - Partial matches allowed (e.g., "s. 8" might match "s. 8 of Charter")
    - Use this to identify COMMON REFLEX ERRORS (over-extraction, under-extraction)
    
    Args:
        reflex_autolabel_file: Path to the transformed Reflex file (with auto_label tags)
        human_annotation_file: Path to human annotated file (with manual_label tags)
        match_type: 'context' for exact context match (strict),
                   'context_overlap' for similar context match (lenient, default)
        context_chars: Number of characters before/after for context (default: 200)
        strict: True for exact text matching, False for similarity matching
        show_matches: If True, display sample matched/unmatched spans for verification
        max_matches_to_show: Maximum number of matches to display (default: 10)
        
    Returns:
        Dictionary with overall metrics
    """
    mode_name = "STRICT (Exact Match)" if strict else "LENIENT (Similarity Match)"
    
    print(f"\n{'='*80}")
    print(f"COMPARING REFLEX SPANS VS HUMAN SPANS - {mode_name}")
    print(f"{'='*80}")
    print(f"\nReflex file:  {Path(reflex_autolabel_file).name}")
    print(f"Human file:   {Path(human_annotation_file).name}")
    print(f"Match type:   {match_type}")
    print(f"Context:      {context_chars} characters")
    print(f"Strict mode:  {strict}")
    print(f"\nNote: Comparing ONLY decision, legislation, and secondary sources")
    print(f"      Label names are IGNORED (span position matching only)")
    if strict:
        print(f"      EXACT text match required + context for disambiguation")
    else:
        print(f"      LENIENT matching: text similarity >= 50%, context >= 70%")
    print()
    
    # Extract spans from both files using IAA functions
    print("Extracting spans from Reflex autolabel file...")
    reflex_spans = extract_spans_from_html(reflex_autolabel_file, context_chars)
    
    print("Extracting spans from human annotation file...")
    all_human_spans = extract_spans_from_html(human_annotation_file, context_chars)
    
    # Filter human spans to ONLY decision, legislation, secondary sources
    target_labels = ['decision', 'legislation', 'secondary sources']
    human_spans = [s for s in all_human_spans if s.labelname.lower() in target_labels]
    
    print(f"\nReflex spans (all):                      {len(reflex_spans)}")
    print(f"Human spans (all):                       {len(all_human_spans)}")
    print(f"Human spans (decision/legislation/sec):  {len(human_spans)}")
    
    # Show breakdown by label in human annotations (for information only)
    from collections import Counter
    human_label_counts = Counter(s.labelname for s in human_spans)
    if human_label_counts:
        print(f"\nHuman annotation breakdown:")
        for label, count in sorted(human_label_counts.items()):
            print(f"  {label}: {count}")
    
    # Calculate span-level matching (ignoring label names)
    print(f"\n{'='*80}")
    print(f"SPAN-LEVEL MATCHING - {mode_name}")
    print(f"{'='*80}\n")
    
    # Match Reflex spans to human spans (precision calculation)
    matched_reflex = 0
    matched_human_indices = set()
    matched_pairs = []  # For debugging
    unmatched_reflex = []  # For debugging
    
    for reflex_span in reflex_spans:
        found_match = False
        for j, human_span in enumerate(human_spans):
            if j not in matched_human_indices:
                if span_match_ignore_label(reflex_span, human_span, match_type, strict=strict):
                    matched_reflex += 1
                    matched_human_indices.add(j)
                    matched_pairs.append((reflex_span, human_span))
                    found_match = True
                    break
        
        if not found_match:
            unmatched_reflex.append(reflex_span)
    
    matched_human = len(matched_human_indices)
    unmatched_human = [s for i, s in enumerate(human_spans) if i not in matched_human_indices]
    
    # Calculate metrics
    precision = matched_reflex / len(reflex_spans) if reflex_spans else 0
    recall = matched_human / len(human_spans) if human_spans else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Print results
    print(f"Matched spans: {matched_reflex}")
    print(f"  - Reflex spans that matched a human span: {matched_reflex}/{len(reflex_spans)}")
    print(f"  - Human spans that matched a Reflex span: {matched_human}/{len(human_spans)}")
    
    print(f"\n{'='*80}")
    print("METRICS")
    print(f"{'='*80}\n")
    
    print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"  = Matched Reflex spans / Total Reflex spans")
    print(f"  = {matched_reflex} / {len(reflex_spans)}")
    print(f"  = How many Reflex spans correspond to human annotations\n")
    
    print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"  = Matched Human spans / Total Human spans")
    print(f"  = {matched_human} / {len(human_spans)}")
    print(f"  = How many human annotations were found by Reflex\n")
    
    print(f"F1 Score:  {f1:.4f} ({f1*100:.2f}%)")
    print(f"  = Harmonic mean of Precision and Recall\n")
    
    # Show sample matches for verification
    if show_matches and matched_pairs:
        print(f"{'='*80}")
        print(f"SAMPLE MATCHES (for verification)")
        print(f"{'='*80}\n")
        for i, (reflex_span, human_span) in enumerate(matched_pairs[:max_matches_to_show]):
            print(f"Match {i+1}:")
            print(f"  Reflex: '{reflex_span.text[:80]}...'")
            print(f"  Human:  '{human_span.text[:80]}...' (label: {human_span.labelname})")
            print()
        
        if len(matched_pairs) > max_matches_to_show:
            print(f"... and {len(matched_pairs) - max_matches_to_show} more matches\n")
    
    # Show sample unmatched spans
    if show_matches and unmatched_reflex:
        print(f"{'='*80}")
        print(f"SAMPLE UNMATCHED REFLEX SPANS")
        print(f"{'='*80}\n")
        for i, span in enumerate(unmatched_reflex[:max_matches_to_show]):
            print(f"Unmatched {i+1}: '{span.text[:100]}...'")
        
        if len(unmatched_reflex) > max_matches_to_show:
            print(f"... and {len(unmatched_reflex) - max_matches_to_show} more unmatched\n")
    
    if show_matches and unmatched_human:
        print(f"{'='*80}")
        print(f"SAMPLE UNMATCHED HUMAN SPANS")
        print(f"{'='*80}\n")
        for i, span in enumerate(unmatched_human[:max_matches_to_show]):
            print(f"Unmatched {i+1}: '{span.text[:100]}...' (label: {span.labelname})")
        
        if len(unmatched_human) > max_matches_to_show:
            print(f"... and {len(unmatched_human) - max_matches_to_show} more unmatched\n")
    
    print(f"{'='*80}\n")
    
    # Return results for programmatic use
    return {
        'mode': 'strict' if strict else 'lenient',
        'matched_reflex': matched_reflex,
        'matched_human': matched_human,
        'reflex_count': len(reflex_spans),
        'human_count': len(human_spans),
        'human_total_count': len(all_human_spans),
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'human_label_breakdown': dict(human_label_counts) if human_label_counts else {},
        'matched_pairs': matched_pairs,
        'unmatched_reflex': unmatched_reflex,
        'unmatched_human': unmatched_human
    }


def main():
    # Hardcoded input file path
    filename = "1989CanLII1415ONCA" #"2021QCCA1675" #1997CanLII16226_ONCA"  # Change this to the desired file
    input_file = fr"C:\Users\zakga\OneDrive\Documents\code\LeREaD_annotation_process\data\Document_Échantillon_Initial\ronde_2\reflex\{filename}.html"
    output_file = None  # Will auto-generate with _autolabel suffix
    
    # Step 1: Transform Reflex to autolabel
    print("STEP 1: Transforming Reflex annotations to autolabel format")
    print("="*80)
    process_file(input_file, output_file)
    
    # Step 2: Compare with human annotation (optional - uncomment to use)
    # Set the human annotator file path here
    # Note: Update this path to match your human annotation file naming pattern
    anno = "GL"
    human_file = fr"C:\Users\zakga\OneDrive\Documents\code\LeREaD_annotation_process\data\Documents_Annotés\{anno}\{filename}_annotated_{anno}_tech.html"
    
    # Generate the autolabel file path
    input_path = Path(input_file)
    autolabel_file = input_path.parent / f"{input_path.stem}_autolabel{input_path.suffix}"
    
    # Check if human annotation file exists
    if os.path.exists(human_file):
        print("\nSTEP 2: Comparing Reflex autolabel with human annotation")
        print("="*80)
        
        # Run STRICT comparison (exact matches)
        print("\n" + "="*80)
        print("RUNNING STRICT COMPARISON (Exact Text Match)")
        print("="*80)
        results_strict = compare_reflex_vs_human(
            str(autolabel_file),
            human_file,
            match_type='context_overlap',
            context_chars=200,
            strict=True,  # Exact matching
            show_matches=True,
            max_matches_to_show=5
        )
        
        # Run LENIENT comparison (similarity matches)
        print("\n" + "="*80)
        print("RUNNING LENIENT COMPARISON (Similarity Match - to see Reflex errors)")
        print("="*80)
        results_lenient = compare_reflex_vs_human(
            str(autolabel_file),
            human_file,
            match_type='context_overlap',
            context_chars=200,
            strict=False,  # Lenient matching
            show_matches=True,
            max_matches_to_show=5
        )
        
        # Summary comparison
        print("\n" + "="*80)
        print("SUMMARY: STRICT vs LENIENT COMPARISON")
        print("="*80)
        print(f"\nSTRICT MODE (Exact matches only):")
        print(f"  Precision: {results_strict['precision']:.4f} ({results_strict['precision']*100:.2f}%)")
        print(f"  Recall:    {results_strict['recall']:.4f} ({results_strict['recall']*100:.2f}%)")
        print(f"  F1 Score:  {results_strict['f1']:.4f} ({results_strict['f1']*100:.2f}%)")
        print(f"  Matched:   {results_strict['matched_reflex']}/{results_strict['reflex_count']} Reflex, {results_strict['matched_human']}/{results_strict['human_count']} Human")
        
        print(f"\nLENIENT MODE (Partial matches allowed - shows Reflex errors):")
        print(f"  Precision: {results_lenient['precision']:.4f} ({results_lenient['precision']*100:.2f}%)")
        print(f"  Recall:    {results_lenient['recall']:.4f} ({results_lenient['recall']*100:.2f}%)")
        print(f"  F1 Score:  {results_lenient['f1']:.4f} ({results_lenient['f1']*100:.2f}%)")
        print(f"  Matched:   {results_lenient['matched_reflex']}/{results_lenient['reflex_count']} Reflex, {results_lenient['matched_human']}/{results_lenient['human_count']} Human")
        
        print(f"\nDifference (Lenient - Strict):")
        print(f"  Additional matches: {results_lenient['matched_reflex'] - results_strict['matched_reflex']} Reflex spans")
        print(f"  These represent partial/fuzzy matches (Reflex over/under-extraction)")
        print(f"\n{'='*80}\n")
        
    else:
        print(f"\nNote: Human annotation file not found at:")
        print(f"  {human_file}")
        print("Skipping comparison step.")
        print("\nTo enable comparison, update the human_file path in main() function.")


if __name__ == '__main__':
    main()





"""
1997CanLII16226_ONCA
REFLEX RESULTS ANALYSIS:
STRICT MODE (Exact matches only):
  Precision: 0.2081 (20.81%)
  Recall:    0.0744 (7.44%)
  F1 Score:  0.1097 (10.97%)
  Matched:   46/221 Reflex, 46/618 Human

LENIENT MODE (Partial matches allowed - shows Reflex errors):
  Precision: 0.5385 (53.85%)
  Recall:    0.1926 (19.26%)
  F1 Score:  0.2837 (28.37%)
  Matched:   119/221 Reflex, 119/618 Human


  
  LENIENT EXAMPLES: 

  Reflex: 'Charter of Rights...'
  Human:  'Charter of Rights and Freedoms...' (label: legislation)

  Reflex: 's. 8...'
  Human:  's. 8 of Charter...' (label: legislation)

  Reflex: 'Canadian Charter of Rights and Freedoms, ss. 8...'
  Human:  'Canadian Charter of Rights and Freedoms, ss. 8, 24(2)...' (label: legislation)

  Reflex: 's. 1...'
  Human:  's. 1 of Charter...' (label: legislation)

  
  2021QCCA1675

  STRICT MODE (Exact matches only):
  Precision: 0.1667 (16.67%)
  Recall:    0.0606 (6.06%)
  F1 Score:  0.0889 (8.89%)
  Matched:   6/36 Reflex, 6/99 Human

LENIENT MODE (Partial matches allowed - shows Reflex errors):
  Precision: 0.6111 (61.11%)
  Recall:    0.2222 (22.22%)
  F1 Score:  0.3259 (32.59%)
  Matched:   22/36 Reflex, 22/99 Human

Difference (Lenient - Strict):
  Additional matches: 16 Reflex spans
  These represent partial/fuzzy matches (Reflex over/under-extraction)


1989CanLII1415ONCA
STRICT MODE (Exact matches only):
  Precision: 0.2000 (20.00%)
  Recall:    0.0108 (1.08%)
  F1 Score:  0.0204 (2.04%)
  Matched:   1/5 Reflex, 1/93 Human

LENIENT MODE (Partial matches allowed - shows Reflex errors):
  Precision: 0.6000 (60.00%)
  Recall:    0.0323 (3.23%)
  F1 Score:  0.0612 (6.12%)
  Matched:   3/5 Reflex, 3/93 Human

  Reflex: 'R.S.C. 1985, c. E-15...'
  Human:  'R.S.C. 1970, c. E-13; now R.S.C. 1985, c. E-15...' (label: legislation)

  Reflex: '1975 CanLII 194 (SCC), [1977] 1 S.C.R. 456...'
  Human:  '[1977] 1 S.C.R. 456...' (label: decision)
"""
