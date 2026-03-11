import os
import glob
import re

# Test with actual file
html_file = r"C:\Users\zakga\OneDrive\Documents\code\LeREaD_annotation_process\data\Documents_Annotés\llm\p2_c500_fsselected-30_mgpt-5.2\2019SCC65_v1.3.html"

print(f"Reading file: {html_file}")
print(f"File exists: {os.path.exists(html_file)}\n")

with open(html_file, 'r', encoding='utf-8') as f:
    html_content = f.read()

print(f"File size: {len(html_content)} characters\n")

# The pattern
pattern = r'(<auto_label[^>]*>)([""''])(.+?)([""''])(</auto_label>)'

print(f"Pattern: {pattern}\n")
print("="*80)

# Find all matches
matches = list(re.finditer(pattern, html_content))
print(f"\nTotal matches found: {len(matches)}")

# Filter for title + secondary sources
relevant_matches = []
for match in matches:
    opening_tag = match.group(1)
    if 'labelname="title"' in opening_tag and 'parent="secondary sources"' in opening_tag:
        relevant_matches.append(match)

print(f"Matches with labelname='title' and parent='secondary sources': {len(relevant_matches)}")

if relevant_matches:
    print("\nFirst 5 relevant matches:")
    for i, match in enumerate(relevant_matches[:5], 1):
        print(f"\n{i}. Match at position {match.start()}-{match.end()}:")
        print(f"   Opening quote: {match.group(2)!r}")
        print(f"   Content (first 50 chars): {match.group(3)[:50]}...")
        print(f"   Closing quote: {match.group(4)!r}")
