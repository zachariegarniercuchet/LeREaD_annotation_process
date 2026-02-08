"""
Quick script to fix parent attributes in auto_label tags based on nesting context.
"""
import sys
sys.path.append('llm_based_annotation')

from utils.document_level_post_processing_utils import add_style_and_parent_to_auto_labels

# Read the HTML file
input_file = r"data\Documents_Annotés\llm\1997CanLII16226_ONCA\1997CanLII16226_ONCA_llm_v2.html"

with open(input_file, 'r', encoding='utf-8') as f:
    html_content = f.read()

print(f"Processing {input_file}...")

# Apply the function to fix parents based on nesting
fixed_html = add_style_and_parent_to_auto_labels(html_content)

# Write back to the same file
with open(input_file, 'w', encoding='utf-8') as f:
    f.write(fixed_html)

print(f"✓ Fixed parent attributes in {input_file}")
