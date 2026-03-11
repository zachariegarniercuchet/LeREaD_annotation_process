import re

# Let's check what characters are actually in the pattern string
pattern = r'(<auto_label[^>]*>)([""''])(.+?)([""''])(</auto_label>)'

print("Characters in the pattern quote groups:")
# Extract the quote character class from the pattern
import re as re_module
quote_chars = "\"\"''"

for i, char in enumerate(quote_chars):
    print(f"  Char {i}: '{char}' - Unicode: U+{ord(char):04X} - {repr(char)}")

print("\n" + "="*80)

# Now let's build the pattern with explicit Unicode escapes to ensure it works
pattern_explicit = r'(<auto_label[^>]*>)([\u0022\u201C\u201D\u0027\u2018\u2019])(.+?)([\u0022\u201C\u201D\u0027\u2018\u2019])(</auto_label>)'

print(f"\nPattern with explicit Unicode: {pattern_explicit}")

# Test with a sample
test_html = '<auto_label labelname="title" parent="secondary sources">"Test"</auto_label>'
print(f"\nTest HTML: {test_html}")
print(f"Quotes in test: {repr(test_html[63])} and {repr(test_html[68])}")

matches = list(re.finditer(pattern_explicit, test_html))
print(f"\nMatches with explicit pattern: {len(matches)}")
