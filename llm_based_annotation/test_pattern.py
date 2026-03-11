import re

# Your exact example
test_html = '<auto_label labelname="title" parent="secondary sources" style="background-color: rgb(147, 196, 125); color: black;" verified="false" titletype="official">"Triumph of Reasonableness: But How Much Does It Really Matter?"</auto_label>'

print("Test HTML:")
print(test_html)
print("\n" + "="*80 + "\n")

# The pattern from the code
pattern = r'(<auto_label[^>]*>)([""''])(.+?)([""''])(</auto_label>)'

print(f"Pattern: {pattern}")
print("\n" + "="*80 + "\n")

# Try to find matches
matches = list(re.finditer(pattern, test_html))

print(f"Number of matches found: {len(matches)}")

if matches:
    for i, match in enumerate(matches, 1):
        print(f"\nMatch {i}:")
        print(f"  Full match: {match.group(0)}")
        print(f"  Group 1 (opening tag): {match.group(1)}")
        print(f"  Group 2 (opening quote): {match.group(2)!r}")
        print(f"  Group 3 (content): {match.group(3)}")
        print(f"  Group 4 (closing quote): {match.group(4)!r}")
        print(f"  Group 5 (closing tag): {match.group(5)}")
else:
    print("No matches found!")
    print("\nLet's check the quote characters in the HTML:")
    for i, char in enumerate(test_html):
        if char in '""\'"\'':
            print(f"  Position {i}: '{char}' (Unicode: U+{ord(char):04X})")
