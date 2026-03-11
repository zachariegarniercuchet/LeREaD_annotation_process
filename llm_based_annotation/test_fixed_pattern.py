import re

html_file = r"C:\Users\zakga\OneDrive\Documents\code\LeREaD_annotation_process\data\Documents_Annotés\llm\p2_c500_fsselected-30_mgpt-5.2\2019SCC65_v1.3.html"

with open(html_file, 'r', encoding='utf-8') as f:
    html_content = f.read()

# Updated pattern with Unicode escapes
pattern = r'(<auto_label[^>]*>)([\u0022\u201C\u201D\u0027\u2018\u2019])(.+?)([\u0022\u201C\u201D\u0027\u2018\u2019])(</auto_label>)'

matches = list(re.finditer(pattern, html_content))
print(f"Total matches found: {len(matches)}")

# Filter for title + secondary sources
relevant_matches = []
for match in matches:
    opening_tag = match.group(1)
    if 'labelname="title"' in opening_tag and 'parent="secondary sources"' in opening_tag:
        relevant_matches.append(match)

print(f"Matches with labelname='title' and parent='secondary sources': {len(relevant_matches)}")

if relevant_matches:
    print("\nFirst 3 relevant matches:")
    for i, match in enumerate(relevant_matches[:3], 1):
        opening_quote = match.group(2)
        content = match.group(3)[:60]
        closing_quote = match.group(4)
        print(f"\n{i}. {opening_quote}{content}...{closing_quote}")
