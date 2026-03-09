"""
Context-Aware Inter-Annotator Agreement (IAA) Evaluation for HTML Annotations

This module provides span-level IAA evaluation using surrounding text context
to accurately match annotations between two annotators, even when HTML structure differs.

Key Innovation: Uses context text around each label to ensure we're comparing
the same document location, not just identical text that may appear multiple times.
"""

from bs4 import BeautifulSoup
from collections import defaultdict
from typing import Dict, List, Tuple
import os
from pathlib import Path
import re
from datetime import datetime


class Span:
    """Represents an annotated span with context for unique identification."""
    
    def __init__(self, text: str, start: int, end: int, labelname: str, 
                 attributes: Dict[str, str], context_text: str = ""):
        self.text = text.strip()
        self.start = start
        self.end = end
        self.labelname = labelname
        self.attributes = attributes
        self.context_text = context_text  # Surrounding text for unique location identification
        
    def __repr__(self):
        return f"Span('{self.text[:30]}...', label={self.labelname})"
    
    def context_match(self, other: 'Span') -> bool:
        """
        Check if two spans match based on their surrounding context.
        This ensures we're comparing the SAME location in the document.
        """
        if self.labelname != other.labelname:
            return False
        
        if not self.context_text or not other.context_text:
            return False
        
        # Normalize and compare contexts
        context1 = ' '.join(self.context_text.split()).lower()
        context2 = ' '.join(other.context_text.split()).lower()
        return context1 == context2
    
    def context_overlap(self, other: 'Span', threshold: float = 0.7) -> bool:
        """
        Check if two spans have similar surrounding context (allows partial matches).
        Useful for cases where annotators might have slightly different boundaries.
        """
        if self.labelname != other.labelname:
            return False
        
        if not self.context_text or not other.context_text:
            return False
        
        # Get word sets for both contexts
        words1 = set(self.context_text.lower().split())
        words2 = set(other.context_text.lower().split())
        
        if not words1 or not words2:
            return False
        
        # Calculate Jaccard similarity
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        if union == 0:
            return False
        
        similarity = intersection / union
        return similarity >= threshold


def get_context_around_element(element, plain_text: str, context_chars: int = 200) -> Tuple[str, int, int]:
    """
    Extract context text around an element to uniquely identify its location.
    
    The context includes text BEFORE and AFTER the labeled span, making it
    long enough to be unique in the document even if the label text itself repeats.
    
    Args:
        element: BeautifulSoup element (manual_label)
        plain_text: Plain text of the entire document
        context_chars: Number of characters to include before and after (default: 200)
        
    Returns:
        Tuple of (context_text, start_position, end_position)
    """
    # Get the element's text (normalized)
    elem_text = element.get_text()
    normalized_elem_text = ' '.join(elem_text.split())
    
    if not normalized_elem_text:
        return "", 0, 0
    
    # Normalize the plain text
    normalized_plain = ' '.join(plain_text.split())
    
    # Find the element text in the plain text
    start_idx = normalized_plain.find(normalized_elem_text)
    
    if start_idx == -1:
        # Try with first 50 characters if full text not found
        search_text = normalized_elem_text[:50] if len(normalized_elem_text) > 50 else normalized_elem_text
        start_idx = normalized_plain.find(search_text)
        if start_idx == -1:
            # Fallback: use the text itself as context
            return normalized_elem_text, 0, len(normalized_elem_text)
    
    end_idx = start_idx + len(normalized_elem_text)
    
    # Extract context: text before + element text + text after
    # This makes the context unique even if the label text appears multiple times
    context_start = max(0, start_idx - context_chars)
    context_end = min(len(normalized_plain), end_idx + context_chars)
    
    context_text = normalized_plain[context_start:context_end]
    
    return context_text, start_idx, end_idx


def extract_spans_from_html(html_file: str, context_chars: int = 200) -> List[Span]:
    """
    Extract all manual_label spans from an HTML file with surrounding context.
    
    Process:
    1. Extract plain text from the document body (no HTML tags)
    2. For each manual_label, capture text before + label text + text after
    3. Create spans with this context for accurate cross-document matching
    
    Args:
        html_file: Path to the HTML file
        context_chars: Number of characters before/after for context (default: 200)
        
    Returns:
        List of Span objects with context information
    """
    with open(html_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    soup = BeautifulSoup(content, 'html.parser')
    body = soup.find('body')
    
    if not body:
        return []
    
    # Get plain text from body (all HTML stripped)
    plain_text = body.get_text()
    
    # Find all manual_label elements
    labels = body.find_all('manual_label') + body.find_all('auto_label')
    
    spans = []
    
    for label in labels:
        labelname = label.get('labelname', '')
        
        # Get all attributes except style, labelname
        attributes = {k: v for k, v in label.attrs.items() 
                     if k not in ['style', 'labelname']}
        
        # Get the label's text content (normalized)
        text = label.get_text()
        normalized_text = ' '.join(text.split())
        
        if not normalized_text:
            continue
        
        # Extract context around this label - THIS IS THE KEY INNOVATION
        context_text, start_idx, end_idx = get_context_around_element(label, plain_text, context_chars)
        
        span = Span(normalized_text, start_idx, end_idx, labelname, attributes, context_text)
        spans.append(span)
    
    return spans


def calculate_iaa_metrics(spans1: List[Span], spans2: List[Span], 
                          match_type: str = 'context') -> Dict:
    """
    Calculate IAA metrics using context-aware matching.
    
    Args:
        spans1: Spans from annotator 1 (with context)
        spans2: Spans from annotator 2 (with context)
        match_type: 'context' for exact context match (strict),
                   'context_overlap' for similar context match (lenient)
        
    Returns:
        Dictionary with metrics: precision, recall, f1, matched counts
    """
    if match_type == 'context':
        # Exact context match: same surrounding text and label
        matched = 0
        matched_indices = set()
        
        for s1 in spans1:
            for j, s2 in enumerate(spans2):
                if j not in matched_indices and s1.context_match(s2):
                    matched += 1
                    matched_indices.add(j)
                    break
        
        precision = matched / len(spans1) if spans1 else 0
        recall = matched / len(spans2) if spans2 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'matched': matched,
            'annotator1_count': len(spans1),
            'annotator2_count': len(spans2),
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    elif match_type == 'context_overlap':
        # Context-based overlap: similar surrounding text
        matched1 = set()
        matched2 = set()
        
        for i, s1 in enumerate(spans1):
            for j, s2 in enumerate(spans2):
                if s1.context_overlap(s2):
                    matched1.add(i)
                    matched2.add(j)
        
        precision = len(matched1) / len(spans1) if spans1 else 0
        recall = len(matched2) / len(spans2) if spans2 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'matched_annotator1': len(matched1),
            'matched_annotator2': len(matched2),
            'annotator1_count': len(spans1),
            'annotator2_count': len(spans2),
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    else:
        raise ValueError(f"Unknown match_type: {match_type}. Use 'context' or 'context_overlap'")


def calculate_per_label_iaa(spans1: List[Span], spans2: List[Span], 
                            match_type: str = 'context') -> Dict[str, Dict]:
    """
    Calculate IAA metrics per label type.
    
    Args:
        spans1: Spans from annotator 1
        spans2: Spans from annotator 2
        match_type: 'context' or 'context_overlap'
        
    Returns:
        Dictionary mapping label names to their metrics
    """
    # Group spans by label
    labels1 = defaultdict(list)
    labels2 = defaultdict(list)
    
    for span in spans1:
        labels1[span.labelname].append(span)
    
    for span in spans2:
        labels2[span.labelname].append(span)
    
    # Get all unique labels
    all_labels = set(labels1.keys()) | set(labels2.keys())
    
    # Calculate metrics for each label
    results = {}
    for label in sorted(all_labels):
        l1_spans = labels1.get(label, [])
        l2_spans = labels2.get(label, [])
        results[label] = calculate_iaa_metrics(l1_spans, l2_spans, match_type)
    
    return results


def calculate_attribute_iaa(spans1: List[Span], spans2: List[Span], 
                           match_type: str = 'context') -> Tuple[Dict, Dict, Dict, Dict]:
    """
    Calculate attribute-level IAA for matched spans (Level 2).
    
    For spans that match at the context level (Level 1), evaluate whether
    the annotators assigned the same attribute values.
    
    Args:
        spans1: Spans from annotator 1
        spans2: Spans from annotator 2
        match_type: 'context' or 'context_overlap'
        
    Returns:
        Tuple of (overall_metrics, per_label_metrics, overall_attribute_metrics, parent_attribute_metrics)
    """
    # Find matched pairs
    matched_pairs = []
    matched_indices_2 = set()
    
    if match_type == 'context':
        for s1 in spans1:
            for j, s2 in enumerate(spans2):
                if j not in matched_indices_2 and s1.context_match(s2):
                    matched_pairs.append((s1, s2))
                    matched_indices_2.add(j)
                    break
    elif match_type == 'context_overlap':
        # For overlap, we need to find best matches
        matched_indices_1 = set()
        matched_indices_2 = set()
        
        for i, s1 in enumerate(spans1):
            if i in matched_indices_1:
                continue
            for j, s2 in enumerate(spans2):
                if j not in matched_indices_2 and s1.context_overlap(s2):
                    matched_pairs.append((s1, s2))
                    matched_indices_1.add(i)
                    matched_indices_2.add(j)
                    break
    
    if not matched_pairs:
        return {
            'total_matched_spans': 0,
            'total_attributes': 0,
            'matching_attributes': 0,
            'precision': 0,
            'recall': 0,
            'f1': 0
        }, {}, {}, {}
    
    # Calculate attribute agreement
    total_attributes = 0
    matching_attributes = 0
    per_label_stats = defaultdict(lambda: {
        'total': 0, 
        'matching': 0, 
        'spans': 0,
        'attributes': defaultdict(lambda: {'total': 0, 'matching': 0})
    })
    
    # Track overall per-attribute stats (across all labels)
    overall_attribute_stats = defaultdict(lambda: {'total': 0, 'matching': 0})
    
    # Track per-category attribute stats (legislation, decision, secondary sources)
    category_attribute_stats = defaultdict(lambda: defaultdict(lambda: {'total': 0, 'matching': 0}))
    
    for s1, s2 in matched_pairs:
        # Determine the category: check labelname first, then parent attribute
        labelname_lower = s1.labelname.lower()
        
        if labelname_lower in ['legislation', 'decision', 'secondary sources']:
            category = labelname_lower
        else:
            # Look at parent attribute
            parent_attr = s1.attributes.get('parent', '')
            if parent_attr:
                # Extract top-level parent (first element if comma-separated)
                top_parent = parent_attr.split(',')[0].strip().lower()
                if top_parent in ['legislation', 'decision', 'secondary sources']:
                    category = top_parent
                else:
                    category = 'other'
            else:
                category = 'other'
        
        # Get attributes excluding style parent, labelname, verified
        attrs1 = {k: v for k, v in s1.attributes.items() 
                 if k not in ['style', 'parent', 'labelname', 'verified']}
        attrs2 = {k: v for k, v in s2.attributes.items() 
                 if k not in ['style', 'parent', 'labelname', 'verified']}
        
        # Get all unique attribute keys
        all_keys = set(attrs1.keys()) | set(attrs2.keys())
        
        for key in all_keys:
            total_attributes += 1
            per_label_stats[s1.labelname]['total'] += 1
            per_label_stats[s1.labelname]['attributes'][key]['total'] += 1
            overall_attribute_stats[key]['total'] += 1
            category_attribute_stats[category][key]['total'] += 1
            
            val1 = attrs1.get(key, None)
            val2 = attrs2.get(key, None)
            
            if val1 == val2:
                matching_attributes += 1
                per_label_stats[s1.labelname]['matching'] += 1
                per_label_stats[s1.labelname]['attributes'][key]['matching'] += 1
                overall_attribute_stats[key]['matching'] += 1
                category_attribute_stats[category][key]['matching'] += 1
        
        per_label_stats[s1.labelname]['spans'] += 1
    
    # Calculate overall metrics
    agreement = matching_attributes / total_attributes if total_attributes > 0 else 0
    
    overall = {
        'total_matched_spans': len(matched_pairs),
        'total_attributes': total_attributes,
        'matching_attributes': matching_attributes,
        'agreement': agreement,
        'precision': agreement,
        'recall': agreement,
        'f1': agreement
    }
    
    # Calculate per-label metrics
    per_label = {}
    for label, stats in per_label_stats.items():
        agreement_rate = stats['matching'] / stats['total'] if stats['total'] > 0 else 0
        
        # Calculate per-attribute agreement rates
        attributes_detail = {}
        for attr_name, attr_stats in stats['attributes'].items():
            attr_agreement = attr_stats['matching'] / attr_stats['total'] if attr_stats['total'] > 0 else 0
            attributes_detail[attr_name] = {
                'total': attr_stats['total'],
                'matching': attr_stats['matching'],
                'agreement': attr_agreement
            }
        
        per_label[label] = {
            'matched_spans': stats['spans'],
            'total_attributes': stats['total'],
            'matching_attributes': stats['matching'],
            'agreement': agreement_rate,
            'f1': agreement_rate,
            'attributes': attributes_detail
        }
    
    # Calculate overall per-attribute metrics (across all labels)
    overall_attributes = {}
    for attr_name, attr_stats in overall_attribute_stats.items():
        attr_agreement = attr_stats['matching'] / attr_stats['total'] if attr_stats['total'] > 0 else 0
        overall_attributes[attr_name] = {
            'total': attr_stats['total'],
            'matching': attr_stats['matching'],
            'agreement': attr_agreement
        }
    
    # Calculate per-category per-attribute metrics
    category_attributes = {}
    for category_name, attrs in category_attribute_stats.items():
        category_attributes[category_name] = {}
        for attr_name, attr_stats in attrs.items():
            attr_agreement = attr_stats['matching'] / attr_stats['total'] if attr_stats['total'] > 0 else 0
            category_attributes[category_name][attr_name] = {
                'total': attr_stats['total'],
                'matching': attr_stats['matching'],
                'agreement': attr_agreement
            }
    
    return overall, per_label, overall_attributes, category_attributes


def print_attribute_iaa_results(overall: Dict, per_label: Dict[str, Dict], 
                               overall_attributes: Dict[str, Dict],
                               category_attributes: Dict[str, Dict[str, Dict]],
                               match_type: str, file1: str, file2: str):
    """
    Print Level 2 attribute IAA results in a beautiful, readable format.
    """
    # Header
    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + " " * 12 + "LEVEL 2: ATTRIBUTE-LEVEL INTER-ANNOTATOR AGREEMENT" + " " * 16 + "║")
    print("╚" + "═" * 78 + "╝")
    
    # Files being compared
    print(f"\n📄 Document 1: {os.path.basename(file1)}")
    print(f"📄 Document 2: {os.path.basename(file2)}")
    
    # Method description
    print(f"\n🔍 Analysis Level: ATTRIBUTE AGREEMENT")
    print(f"   → Evaluating attribute values for matched spans (Level 1 matches)")
    print(f"   → Excluded attributes: style, parent, labelname")
    print(f"   → Included: docid, uri, titletype, fragmentid, etc.")
    
    # Overall results
    print("\n" + "─" * 80)
    print("📊 OVERALL ATTRIBUTE AGREEMENT")
    print("─" * 80)
    
    print(f"\n  Matched Spans (from Level 1):  {overall['total_matched_spans']:>4}")
    print(f"  Total Attributes Compared:     {overall['total_attributes']:>4}")
    print(f"  Matching Attribute Values:     {overall['matching_attributes']:>4}")
    
    print(f"\n  {'Metric':<20} {'Value':>10}   {'Percentage':>10}")
    print(f"  {'-'*20} {'-'*10}   {'-'*10}")
    print(f"  {'Agreement Rate':<20} {overall['agreement']:>10.4f}   {overall['agreement']*100:>9.2f}%")
    print(f"  {'F1 Score':<20} {overall['f1']:>10.4f}   {overall['f1']*100:>9.2f}%")
    
    # Category-based attribute breakdown table
    if overall_attributes:
        print("\n" + "─" * 120)
        print("🌐 ATTRIBUTE AGREEMENT BY CATEGORY")
        print("─" * 120)
        
        # Table header
        print(f"\n{'Attribute':<20} {'Legislation':<22} {'Decision':<22} {'Secondary Sources':<22} {'Total':>8} {'Match':>8} {'Agreement':>10}")
        print("─" * 120)
        
        for attr_name in sorted(overall_attributes.keys()):
            overall_data = overall_attributes[attr_name]
            
            # Get data for each category
            leg_data = category_attributes.get('legislation', {}).get(attr_name, None)
            dec_data = category_attributes.get('decision', {}).get(attr_name, None)
            sec_data = category_attributes.get('secondary sources', {}).get(attr_name, None)
            
            # Format category cells as "match/total (XX%)"
            leg_str = f"{leg_data['matching']}/{leg_data['total']} ({leg_data['agreement']*100:.1f}%)" if leg_data else "-"
            dec_str = f"{dec_data['matching']}/{dec_data['total']} ({dec_data['agreement']*100:.1f}%)" if dec_data else "-"
            sec_str = f"{sec_data['matching']}/{sec_data['total']} ({sec_data['agreement']*100:.1f}%)" if sec_data else "-"
            
            print(f"{attr_name:<20} {leg_str:<22} {dec_str:<22} {sec_str:<22} "
                  f"{overall_data['total']:>8} {overall_data['matching']:>8} {overall_data['agreement']*100:>9.2f}%")
    
    print("\n" + "═" * 120 + "\n")


def print_iaa_results(overall: Dict, per_label: Dict[str, Dict], 
                     match_type: str, file1: str, file2: str, context_chars: int):
    """
    Print Level 1 span IAA results in a beautiful, readable format.
    """
    # Header
    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + " " * 10 + "LEVEL 1: SPAN-LEVEL INTER-ANNOTATOR AGREEMENT (IAA)" + " " * 17 + "║")
    print("╚" + "═" * 78 + "╝")
    
    # Files being compared
    print(f"\n📄 Document 1: {os.path.basename(file1)}")
    print(f"📄 Document 2: {os.path.basename(file2)}")
    
    # Method description
    print(f"\n🔍 Matching Strategy: {match_type.upper()}")
    if match_type == 'context':
        print(f"   → Exact context matching ({context_chars} chars before/after each label)")
        print(f"   → Ensures labels are at the SAME document location")
    elif match_type == 'context_overlap':
        print(f"   → Similarity-based context matching (70% threshold)")
        print(f"   → Allows slight boundary differences between annotators")
    
    # Overall results
    print("\n" + "─" * 80)
    print("📊 OVERALL AGREEMENT")
    print("─" * 80)
    
    if match_type == 'context':
        print(f"\n  Total Labels (Annotator 1):  {overall['annotator1_count']:>4}")
        print(f"  Total Labels (Annotator 2):  {overall['annotator2_count']:>4}")
        print(f"  Matched Labels:              {overall['matched']:>4}")
    else:
        print(f"\n  Total Labels (Annotator 1):  {overall['annotator1_count']:>4}")
        print(f"  Total Labels (Annotator 2):  {overall['annotator2_count']:>4}")
        print(f"  Matched (Annotator 1):       {overall['matched_annotator1']:>4}")
        print(f"  Matched (Annotator 2):       {overall['matched_annotator2']:>4}")
    
    print(f"\n  {'Metric':<15} {'Value':>10}   {'Percentage':>10}")
    print(f"  {'-'*15} {'-'*10}   {'-'*10}")
    print(f"  {'Precision':<15} {overall['precision']:>10.4f}   {overall['precision']*100:>9.2f}%")
    print(f"  {'Recall':<15} {overall['recall']:>10.4f}   {overall['recall']*100:>9.2f}%")
    print(f"  {'F1 Score':<15} {overall['f1']:>10.4f}   {overall['f1']*100:>9.2f}%")
    
    # Per-label results
    print("\n" + "─" * 80)
    print("📋 PER-LABEL BREAKDOWN")
    print("─" * 80)
    
    print(f"\n{'Label':<30} {'A1':>6} {'A2':>6} {'Match':>8} {'F1 Score':>10}")
    print("─" * 80)
    
    for label in sorted(per_label.keys()):
        metrics = per_label[label]
        if match_type == 'context':
            match_str = f"{metrics['matched']}"
        else:
            match_str = f"{metrics['matched_annotator1']}/{metrics['matched_annotator2']}"
        
        # Truncate long label names
        display_label = label if len(label) <= 30 else label[:27] + "..."
        
        print(f"{display_label:<30} {metrics['annotator1_count']:>6} {metrics['annotator2_count']:>6} "
              f"{match_str:>8} {metrics['f1']*100:>9.2f}%")
    
    print("\n" + "═" * 80 + "\n")


def evaluate_iaa(file1: str, file2: str, match_type: str = 'context', context_chars: int = 200):
    """
    Evaluate Inter-Annotator Agreement using context-aware matching.
    
    Args:
        file1: Path to first annotated HTML file
        file2: Path to second annotated HTML file
        match_type: 'context' (strict, RECOMMENDED) or 'context_overlap' (lenient)
        context_chars: Characters before/after label for context (default: 200)
    
    Returns:
        Tuple of (overall_metrics, per_label_metrics)
    """
    print(f"\n🔄 Extracting spans from {os.path.basename(file1)}...")
    spans1 = extract_spans_from_html(file1, context_chars)
    print(f"   ✓ Found {len(spans1)} labeled spans")
    
    print(f"\n🔄 Extracting spans from {os.path.basename(file2)}...")
    spans2 = extract_spans_from_html(file2, context_chars)
    print(f"   ✓ Found {len(spans2)} labeled spans")
    
    # Calculate metrics
    print(f"\n⚙️  Computing {match_type} agreement metrics...")
    overall = calculate_iaa_metrics(spans1, spans2, match_type)
    per_label = calculate_per_label_iaa(spans1, spans2, match_type)
    
    # Print results
    print_iaa_results(overall, per_label, match_type, file1, file2, context_chars)
    
    return overall, per_label


def longest_common_substring(s1: str, s2: str) -> str:
    """
    Find the longest common substring between two strings.
    Used for matching filenames.
    """
    m = [[0] * (1 + len(s2)) for _ in range(1 + len(s1))]
    longest, x_longest = 0, 0
    for x in range(1, 1 + len(s1)):
        for y in range(1, 1 + len(s2)):
            if s1[x - 1] == s2[y - 1]:
                m[x][y] = m[x - 1][y - 1] + 1
                if m[x][y] > longest:
                    longest = m[x][y]
                    x_longest = x
            else:
                m[x][y] = 0
    return s1[x_longest - longest: x_longest]


def find_matching_llm_file(human_file: str, llm_folder: str) -> str:
    """
    Find the best matching LLM file for a given human annotated file.
    Uses longest common substring matching on filenames.
    
    Args:
        human_file: Path to human annotated HTML file
        llm_folder: Path to folder containing LLM annotated files
        
    Returns:
        Path to best matching LLM file, or None if no good match found
    """
    human_basename = Path(human_file).stem.lower()
    
    best_match = None
    best_match_length = 0
    
    # Search recursively in llm_folder for HTML files
    llm_path = Path(llm_folder)
    if not llm_path.exists():
        return None
    
    for llm_file in llm_path.rglob("*.html"):
        llm_basename = llm_file.stem.lower()
        
        # Find longest common substring
        common = longest_common_substring(human_basename, llm_basename)
        
        if len(common) > best_match_length:
            best_match_length = len(common)
            best_match = str(llm_file)
    
    # Only return match if it's substantial (at least 10 characters)
    if best_match_length >= 10:
        return best_match
    return None


def batch_evaluate_folder(human_folder: str, llm_folder: str, 
                          evaluation_level: str = "both", 
                          match_type: str = "context",
                          context_chars: int = 200):
    """
    Perform batch evaluation of all files in a folder, comparing human annotations with LLM annotations.
    
    Args:
        human_folder: Path to folder containing human annotated files
        llm_folder: Path to folder containing LLM annotated files
        evaluation_level: "level1", "level2", or "both"
        match_type: "context" or "context_overlap"
        context_chars: Context characters for matching
    """
    # Create output folder for results
    output_folder = Path(llm_folder)
    output_file = output_folder / "evaluation_results.txt"
    
    # Get all human annotated HTML files
    human_path = Path(human_folder)
    human_files = sorted(list(human_path.glob("*.html")))
    
    if not human_files:
        print(f"❌ No HTML files found in {human_folder}")
        return
    
    print("\n" + "╔" + "═" * 98 + "╗")
    print("║" + " " * 30 + "BATCH EVALUATION MODE" + " " * 47 + "║")
    print("╚" + "═" * 98 + "╝")
    print(f"\n📁 Human Annotations: {human_folder}")
    print(f"📁 LLM Annotations:   {llm_folder}")
    print(f"📄 Output Log:        {output_file}")
    print(f"\n🔍 Found {len(human_files)} human annotated files")
    
    # Open log file
    with open(output_file, 'w', encoding='utf-8') as log:
        log.write("=" * 100 + "\n")
        log.write(f"BATCH EVALUATION RESULTS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.write("=" * 100 + "\n")
        log.write(f"Human Annotations Folder: {human_folder}\n")
        log.write(f"LLM Annotations Folder:   {llm_folder}\n")
        log.write(f"Evaluation Level:         {evaluation_level}\n")
        log.write(f"Match Type:               {match_type}\n")
        log.write(f"Context Characters:       {context_chars}\n")
        log.write("=" * 100 + "\n\n")
        
        # Store results for summary
        all_results = []
        
        # Process each human file
        for i, human_file in enumerate(human_files, 1):
            human_file_str = str(human_file)
            human_basename = human_file.name
            
            print(f"\n{'─' * 100}")
            print(f"[{i}/{len(human_files)}] Processing: {human_basename}")
            print(f"{'─' * 100}")
            
            log.write(f"\n{'=' * 100}\n")
            log.write(f"FILE {i}/{len(human_files)}: {human_basename}\n")
            log.write(f"{'=' * 100}\n")
            
            # Find matching LLM file
            llm_file = find_matching_llm_file(human_file_str, llm_folder)
            
            if not llm_file:
                msg = f"❌ No matching LLM file found for {human_basename}"
                print(msg)
                log.write(msg + "\n")
                continue
            
            llm_basename = Path(llm_file).name
            print(f"✓ Matched with: {llm_basename}")
            log.write(f"Matched LLM file: {llm_file}\n\n")
            
            try:
                # Extract spans
                spans1 = extract_spans_from_html(human_file_str, context_chars)
                spans2 = extract_spans_from_html(llm_file, context_chars)
                
                print(f"  Human spans: {len(spans1)}, LLM spans: {len(spans2)}")
                log.write(f"Human spans: {len(spans1)}, LLM spans: {len(spans2)}\n\n")
                
                # Evaluate based on level
                result = {
                    'human_file': human_basename,
                    'llm_file': llm_basename,
                    'human_spans': len(spans1),
                    'llm_spans': len(spans2)
                }
                
                # Level 1: Span matching
                if evaluation_level in ["level1", "both"]:
                    overall_l1 = calculate_iaa_metrics(spans1, spans2, match_type)
                    per_label_l1 = calculate_per_label_iaa(spans1, spans2, match_type)
                    
                    result['l1_precision'] = overall_l1['precision']
                    result['l1_recall'] = overall_l1['recall']
                    result['l1_f1'] = overall_l1['f1']
                    result['per_label_l1'] = per_label_l1  # Store per-label results
                    
                    log.write("LEVEL 1: SPAN MATCHING\n")
                    log.write(f"  Precision: {overall_l1['precision']*100:.2f}%\n")
                    log.write(f"  Recall:    {overall_l1['recall']*100:.2f}%\n")
                    log.write(f"  F1 Score:  {overall_l1['f1']*100:.2f}%\n\n")
                    
                    # Add per-label breakdown table
                    log.write("─" * 80 + "\n")
                    log.write("📋 PER-LABEL BREAKDOWN\n")
                    log.write("─" * 80 + "\n\n")
                    log.write(f"{'Label':<30} {'A1':>6} {'A2':>6} {'Match':>8} {'F1 Score':>10}\n")
                    log.write("─" * 80 + "\n")
                    
                    for label in sorted(per_label_l1.keys()):
                        metrics = per_label_l1[label]
                        if match_type == 'context':
                            match_str = f"{metrics['matched']}"
                        else:
                            match_str = f"{metrics['matched_annotator1']}/{metrics['matched_annotator2']}"
                        
                        display_label = label if len(label) <= 30 else label[:27] + "..."
                        log.write(f"{display_label:<30} {metrics['annotator1_count']:>6} {metrics['annotator2_count']:>6} "
                                 f"{match_str:>8} {metrics['f1']*100:>9.2f}%\n")
                    
                    log.write("\n")
                    print(f"  L1 F1: {overall_l1['f1']*100:.2f}%", end="")
                
                # Level 2: Attribute matching
                if evaluation_level in ["level2", "both"]:
                    overall_l2, per_label_l2, _, _ = calculate_attribute_iaa(spans1, spans2, match_type)
                    result['l2_agreement'] = overall_l2['agreement']
                    result['l2_f1'] = overall_l2['f1']
                    result['l2_matched_spans'] = overall_l2['total_matched_spans']
                    result['per_label_l2'] = per_label_l2  # Store per-label results
                    
                    log.write("LEVEL 2: ATTRIBUTE MATCHING\n")
                    log.write(f"  Matched Spans:     {overall_l2['total_matched_spans']}\n")
                    log.write(f"  Total Attributes:  {overall_l2['total_attributes']}\n")
                    log.write(f"  Matching Attrs:    {overall_l2['matching_attributes']}\n")
                    log.write(f"  Agreement:         {overall_l2['agreement']*100:.2f}%\n")
                    log.write(f"  F1 Score:          {overall_l2['f1']*100:.2f}%\n")
                    
                    if evaluation_level == "both":
                        print(f", L2 F1: {overall_l2['f1']*100:.2f}%")
                    else:
                        print(f"  L2 F1: {overall_l2['f1']*100:.2f}%")
                
                if evaluation_level == "level1":
                    print()  # Newline
                
                all_results.append(result)
                log.write("\n")
                
            except Exception as e:
                error_msg = f"❌ Error processing {human_basename}: {str(e)}"
                print(error_msg)
                log.write(error_msg + "\n\n")
                continue
        
        # Generate summary table
        if all_results:
            print("\n\n" + "╔" + "═" * 98 + "╗")
            print("║" + " " * 40 + "SUMMARY TABLE" + " " * 45 + "║")
            print("╚" + "═" * 98 + "╝\n")
            
            log.write("\n" + "=" * 100 + "\n")
            log.write("SUMMARY TABLE\n")
            log.write("=" * 100 + "\n\n")
            
            # Table header
            if evaluation_level == "both":
                header = f"{'File':<40} {'Human':>8} {'LLM':>8} {'L1 F1':>10} {'L2 F1':>10}"
                print(header)
                print("─" * 100)
                log.write(header + "\n")
                log.write("─" * 100 + "\n")
                
                for result in all_results:
                    row = f"{result['human_file'][:40]:<40} {result['human_spans']:>8} {result['llm_spans']:>8} {result.get('l1_f1', 0)*100:>9.2f}% {result.get('l2_f1', 0)*100:>9.2f}%"
                    print(row)
                    log.write(row + "\n")
                
                # Calculate means
                mean_l1_f1 = sum(r.get('l1_f1', 0) for r in all_results) / len(all_results)
                mean_l2_f1 = sum(r.get('l2_f1', 0) for r in all_results) / len(all_results)
                
                print("─" * 100)
                mean_row = f"{'MEAN':<40} {'':>8} {'':>8} {mean_l1_f1*100:>9.2f}% {mean_l2_f1*100:>9.2f}%"
                print(mean_row)
                
                log.write("─" * 100 + "\n")
                log.write(mean_row + "\n")
                
            elif evaluation_level == "level1":
                header = f"{'File':<50} {'Human':>8} {'LLM':>8} {'Precision':>12} {'Recall':>12} {'F1':>10}"
                print(header)
                print("─" * 100)
                log.write(header + "\n")
                log.write("─" * 100 + "\n")
                
                for result in all_results:
                    row = f"{result['human_file'][:50]:<50} {result['human_spans']:>8} {result['llm_spans']:>8} {result.get('l1_precision', 0)*100:>11.2f}% {result.get('l1_recall', 0)*100:>11.2f}% {result.get('l1_f1', 0)*100:>9.2f}%"
                    print(row)
                    log.write(row + "\n")
                
                # Calculate means
                mean_precision = sum(r.get('l1_precision', 0) for r in all_results) / len(all_results)
                mean_recall = sum(r.get('l1_recall', 0) for r in all_results) / len(all_results)
                mean_f1 = sum(r.get('l1_f1', 0) for r in all_results) / len(all_results)
                
                print("─" * 100)
                mean_row = f"{'MEAN':<50} {'':>8} {'':>8} {mean_precision*100:>11.2f}% {mean_recall*100:>11.2f}% {mean_f1*100:>9.2f}%"
                print(mean_row)
                
                log.write("─" * 100 + "\n")
                log.write(mean_row + "\n")
                
            else:  # level2
                header = f"{'File':<50} {'Matched':>10} {'Agreement':>12} {'F1':>10}"
                print(header)
                print("─" * 100)
                log.write(header + "\n")
                log.write("─" * 100 + "\n")
                
                for result in all_results:
                    row = f"{result['human_file'][:50]:<50} {result.get('l2_matched_spans', 0):>10} {result.get('l2_agreement', 0)*100:>11.2f}% {result.get('l2_f1', 0)*100:>9.2f}%"
                    print(row)
                    log.write(row + "\n")
                
                # Calculate means
                mean_agreement = sum(r.get('l2_agreement', 0) for r in all_results) / len(all_results)
                mean_f1 = sum(r.get('l2_f1', 0) for r in all_results) / len(all_results)
                
                print("─" * 100)
                mean_row = f"{'MEAN':<50} {'':>10} {mean_agreement*100:>11.2f}% {mean_f1*100:>9.2f}%"
                print(mean_row)
                
                log.write("─" * 100 + "\n")
                log.write(mean_row + "\n")
            
            # Add mean per-label breakdown for Level 1
            if evaluation_level in ["level1", "both"]:
                # Collect all unique labels
                all_labels = set()
                for result in all_results:
                    if 'per_label_l1' in result:
                        all_labels.update(result['per_label_l1'].keys())
                
                if all_labels:
                    print("\n" + "─" * 100)
                    print("📊 MEAN PER-LABEL BREAKDOWN (LEVEL 1)")
                    print("─" * 100)
                    
                    log.write("\n" + "=" * 100 + "\n")
                    log.write("MEAN PER-LABEL BREAKDOWN (LEVEL 1)\n")
                    log.write("=" * 100 + "\n\n")
                    
                    # Calculate mean for each label
                    label_stats = {}
                    for label in sorted(all_labels):
                        total_a1 = 0
                        total_a2 = 0
                        total_matched = 0
                        total_f1 = 0
                        count = 0
                        
                        for result in all_results:
                            if 'per_label_l1' in result and label in result['per_label_l1']:
                                metrics = result['per_label_l1'][label]
                                total_a1 += metrics['annotator1_count']
                                total_a2 += metrics['annotator2_count']
                                if match_type == 'context':
                                    total_matched += metrics['matched']
                                else:
                                    total_matched += metrics['matched_annotator1']
                                total_f1 += metrics['f1']
                                count += 1
                        
                        if count > 0:
                            label_stats[label] = {
                                'a1': total_a1,
                                'a2': total_a2,
                                'matched': total_matched,
                                'mean_f1': total_f1 / count,
                                'count': count
                            }
                    
                    # Print table
                    header = f"{'Label':<30} {'Total A1':>10} {'Total A2':>10} {'Matched':>10} {'Mean F1':>12}"
                    print(header)
                    print("─" * 100)
                    log.write(header + "\n")
                    log.write("─" * 100 + "\n")
                    
                    for label in sorted(label_stats.keys()):
                        stats = label_stats[label]
                        display_label = label if len(label) <= 30 else label[:27] + "..."
                        row = f"{display_label:<30} {stats['a1']:>10} {stats['a2']:>10} {stats['matched']:>10} {stats['mean_f1']*100:>11.2f}%"
                        print(row)
                        log.write(row + "\n")
                    
                    print("─" * 100)
                    log.write("─" * 100 + "\n")
            
            print("\n" + "═" * 100)
            print(f"✓ Results saved to: {output_file}")
            print("═" * 100 + "\n")
            
            log.write("\n" + "=" * 100 + "\n")
            log.write(f"Evaluation completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log.write("=" * 100 + "\n")


if __name__ == "__main__":
    # ============================================================================
    # CONFIGURATION: Choose evaluation mode
    # ============================================================================
    
    # MODE SELECTION
    # "single" = Evaluate a single pair of files
    # "batch"  = Evaluate all files in folders (human vs LLM)
    MODE = "batch"  # Options: "single", "batch"
    
    # ============================================================================
    # BATCH MODE CONFIGURATION
    # ============================================================================
    if MODE == "batch":
        # Folder paths
        project_root = r"C:\Users\zakga\OneDrive\Documents\code\LeREaD_annotation_process"
        human_folder = fr"{project_root}\data\final\Annotated"
        llm_folder = fr"{project_root}\data\Documents_Annotés\llm\p2_c500_fsselected-0_mgpt-5.2"
        
        # Evaluation settings
        evaluation_level = "both"  # Options: "level1", "level2", "both"
        match_type = "context"     # Options: "context", "context_overlap"
        context_chars = 200
        
        # Run batch evaluation
        batch_evaluate_folder(human_folder, llm_folder, evaluation_level, match_type, context_chars)
    
    # ============================================================================
    # SINGLE MODE CONFIGURATION
    # ============================================================================
    elif MODE == "single":
        # Configuration
        anno1 = "EG"  # Options: "GL", "EG", "VP"
        anno2 = "llm"  # Options: "GL", "EG", "VP", "llm"

        # File paths
        project_root = r"C:\Users\zakga\OneDrive\Documents\code\LeREaD_annotation_process"
        filename = '1997CanLII16226_ONCA' #"1989CanLII1415ONCA" #"2021QCCA1675" #"1997CanLII16226_ONCA"
        
        file1 = fr"{project_root}\data\Documents_Annotés\{anno1}\{filename}_annotated_{anno1}_tech.html"
        file2 = fr"{project_root}\data\Documents_Annotés\{anno2}\{filename}\v_prompt_2_500_selected_30_gpt5.2_chunk2_subdef2\{filename}_llm_v1.0.htmL"
        
        # Choose evaluation level
        # "level1" = Span matching (context-based)
        # "level2" = Attribute matching (for matched spans)
        # "both" = Both levels
        evaluation_level = "both"  # Options: "level1", "level2", "both"
        
        # Choose matching type for Level 1
        # "context" = Strict (exact context match, RECOMMENDED)
        # "context_overlap" = Lenient (70% context similarity)
        match_type = "context"  # Options: "context", "context_overlap"
        
        # Extract spans
        print("\n" + "═" * 80)
        print("🔄 EXTRACTING ANNOTATIONS...")
        print("═" * 80)
        spans1 = extract_spans_from_html(file1, context_chars=200)
        spans2 = extract_spans_from_html(file2, context_chars=200)
        print(f"\n✓ Annotator 1 ({anno1}): {len(spans1)} spans")
        print(f"✓ Annotator 2 ({anno2}): {len(spans2)} spans")
        
        # LEVEL 1: SPAN MATCHING
        if evaluation_level in ["level1", "both"]:
            print("\n\n" + "🎯 " + "=" * 76)
            print(f"LEVEL 1: SPAN MATCHING ({match_type.upper()})")
            print("=" * 78)
            
            overall_l1 = calculate_iaa_metrics(spans1, spans2, match_type)
            per_label_l1 = calculate_per_label_iaa(spans1, spans2, match_type)
            
            print_iaa_results(overall_l1, per_label_l1, match_type, file1, file2, 200)
        
        # LEVEL 2: ATTRIBUTE MATCHING
        if evaluation_level in ["level2", "both"]:
            print("\n\n" + "🎯 " + "=" * 76)
            print(f"LEVEL 2: ATTRIBUTE MATCHING (for Level 1 matches)")
            print("=" * 78)
            
            overall_l2, per_label_l2, overall_attrs_l2, category_attrs_l2 = calculate_attribute_iaa(spans1, spans2, match_type)
            
            print_attribute_iaa_results(overall_l2, per_label_l2, overall_attrs_l2, category_attrs_l2, match_type, file1, file2)
        
        # Summary
        if evaluation_level == "both":
            print("\n" + "╔" + "═" * 78 + "╗")
            print("║" + " " * 30 + "SUMMARY" + " " * 41 + "║")
            print("╚" + "═" * 78 + "╝")
            print(f"\n  Level 1 (Span Matching):       F1 = {overall_l1['f1']*100:>6.2f}%")
            print(f"  Level 2 (Attribute Agreement): F1 = {overall_l2['f1']*100:>6.2f}%")
            print("\n" + "═" * 80 + "\n")
    
    else:
        print(f"❌ Unknown MODE: {MODE}. Use 'single' or 'batch'")
