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


if __name__ == "__main__":
    # Configuration
    #anno1 = "EG"  # Options: "GL", "EG", "VP"
    #anno2 = "VP"  # Options: "GL", "EG", "VP"
    #file_name = f"2019SCC65_annotated" #"2016QCCS1184_annotated" #"1999CanLII7320_annotated" #"2001BCSC1342_annotated" 2019SCC65_annotated
    #file1 = rf"C:\Users\zakga\OneDrive\Documents\code\labelstudio\annotation\data\Documents_Annotés\{anno1}\{file_name}_{anno1}_v1.html"
    #file2 = rf"C:\Users\zakga\OneDrive\Documents\code\labelstudio\annotation\data\Documents_Annotés\{anno2}\{file_name}_{anno2}_v1.html"

    anno1 = "EG"  # Options: "GL", "EG", "VP"
    anno2 = "llm"  # Options: "GL", "EG", "VP"

    # File paths
    project_root = r"C:\Users\zakga\OneDrive\Documents\code\LeREaD_annotation_process"
    filename = '1997CanLII16226_ONCA' #"1989CanLII1415ONCA" #"2021QCCA1675" #"1997CanLII16226_ONCA"
    round = "ronde_2"
    anno = "llm"
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
