"""
Extract annotation patterns from few-shot examples and add them to the JSON output.

This script:
1. Reads a JSON file of few-shot examples
2. Parses the 'output' field to extract annotation patterns
3. Adds a 'pattern' field to each example showing the sequence of parent labels and sublabels
4. Saves the modified JSON back to the same file

Pattern format: [(parent_label, sublabel1, sublabel2, ...), ...]
Each tuple represents one top-level annotation with its children in order of appearance.
"""



import json
import re
from pathlib import Path


def extract_annotation_pattern(output_text):
    if not output_text:
        return []
    
    patterns = []
    
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(output_text, 'html.parser')  # html.parser handles malformed XML better
        
        parent_labelnames = ['legislation', 'decision', 'secondary sources']
        
        for label in soup.find_all('manual_label'):
            labelname = label.get('labelname', '')
            
            if labelname not in parent_labelnames:
                continue
            
            # Get ALL nested manual_label descendants in document order
            sublabels = [
                child.get('labelname', '')
                for child in label.find_all('manual_label')  # recursive=True by default
            ]
            
            pattern_list = [labelname] + sublabels
            patterns.append(pattern_list)
    
    except Exception as e:
        print(f"BeautifulSoup parsing failed: {e}. Falling back to regex.")
        patterns = _extract_pattern_regex(output_text)
    
    return patterns


def _extract_pattern_regex(output_text):
    """Fallback regex-based pattern extraction."""
    patterns = []
    parent_labelnames = {'legislation', 'decision', 'secondary sources'}
    
    # Find each parent-level tag and its full content
    parent_pattern = r'<manual_label\s+labelname="(legislation|decision|secondary\s+sources)">(.*?)</manual_label>'
    
    for parent_match in re.finditer(parent_pattern, output_text, re.DOTALL):
        parent_label = parent_match.group(1)
        content = parent_match.group(2)
        
        # Get ALL nested manual_label tags in order of appearance
        child_pattern = r'<manual_label\s+labelname="([^"]+)"'
        sublabels = [
            m.group(1)
            for m in re.finditer(child_pattern, content)
            if m.group(1) not in parent_labelnames
        ]
        
        patterns.append([parent_label] + sublabels)
    
    return patterns


def process_examples_file(input_path, output_path=None):
    """
    Read examples from JSON file, extract patterns, add them to each example,
    and save to file (same file by default).
    
    Args:
        input_path: Path to input JSON file
        output_path: Path to output JSON file (defaults to input_path)
    """
    if output_path is None:
        output_path = input_path
    
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    # Load JSON
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Ensure data is a list
    if not isinstance(data, list):
        print(f"Warning: Expected list, got {type(data).__name__}")
        data = [data]
    
    # Process each example
    processed_count = 0
    for example in data:
        if isinstance(example, dict) and 'example' in example and isinstance(example['example'], dict):
            if 'output' in example['example']:
                output_text = example['example']['output']
                pattern = extract_annotation_pattern(output_text)
                example['pattern'] = pattern
                processed_count += 1
    
    # Save modified data
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Processed {processed_count} examples")
    print(f"✓ Output saved to: {output_path}")


if __name__ == '__main__':
    input_file = r'C:\Users\zakga\OneDrive\Documents\code\LeREaD_annotation_process\few_shot_selection_tool\second_selected\examples_selected_45_with_sources_fixed_spacing_manual_label.json'
    output_file = r'C:\Users\zakga\OneDrive\Documents\code\LeREaD_annotation_process\few_shot_selection_tool\second_selected\examples_selected_45_with_sources_fixed_spacing_manual_label_with_patterns.json'

    input_file = r'C:\Users\zakga\OneDrive\Documents\code\LeREaD_annotation_process\few_shot_selection_tool\few_shot_set_train.json'
    output_file = r'C:\Users\zakga\OneDrive\Documents\code\LeREaD_annotation_process\few_shot_selection_tool\few_shot_set_train.json'

    
    print(f"Processing: {input_file}")
    process_examples_file(input_file, output_file)
    print("Done!")




