
# 1. Load and tokenize document, clean tokens (preserving manual_label tags)
import sys
import json
from pathlib import Path
from typing import List, Dict
from bs4 import BeautifulSoup
import re

sys.path.append(str(Path(__file__).parent.parent / 'annotation' / 'llm_based_annotation' / 'utils'))
from tokenizer_utils import tokenize
from html_cleaner import clean_tokens
from html_utils import is_manual_label_tag, is_auto_label_tag, remove_bookmarks

def load_and_tokenize(html_file: str):
    # Try different encodings
    for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
        try:
            with open(html_file, 'r', encoding=encoding) as f:
                html_content = f.read()
                break
        except UnicodeDecodeError:
            continue
    else:
        with open(html_file, 'r', encoding='utf-8', errors='ignore') as f:
            html_content = f.read()
    soup = BeautifulSoup(html_content, 'html.parser')
    body = soup.find('body')
    if body is not None:
        html_content = str(body)
    tokens = tokenize(html_content)
    cleaned = clean_tokens(tokens, normalize=True, keep_manual_label=True, keep_auto_label=True, keep_bookmarks=True)
    return tokens, remove_bookmarks(cleaned)

# 2. Build structured label-aware representation
def parse_attributes(tag):
    # Parse attributes from a tag string, ignoring style, labelname, parent
    attrs = {}
    for m in re.finditer(r'(\w+)\s*=\s*"([^"]*)"', tag):
        k, v = m.group(1), m.group(2)
        if k.lower() not in {'style', 'labelname', 'parent', 'name'}:
            attrs[k] = v
    return attrs

def build_labelaware_doc(cleaned_tokens):
    labels = []
    sub_pos = 0
    i = 0
    while i < len(cleaned_tokens):
        token = cleaned_tokens[i]
        if is_manual_label_tag(token) == 1:
            # Opening tag
            start_token_idx = i
            start_pos = i - sub_pos
            tag = token
            name = ''
            m = re.search(r'labelname\s*=\s*"([^"]+)"', tag)
            if m:
                name = m.group(1)
            parent = ''
            m = re.search(r'parent\s*=\s*"([^"]+)"', tag)
            if m:
                parent = m.group(1)
            attributes = {}
            for attr_match in re.finditer(r'(\w+)\s*=\s*"([^"]*)"', tag):
                k, v = attr_match.group(1), attr_match.group(2)
                if k.lower() not in {'style', 'labelname', 'parent'}:
                    attributes[k] = v
            # Find closing tag for this label
            j = i + 1
            depth = 1
            while j < len(cleaned_tokens):
                if is_manual_label_tag(cleaned_tokens[j]) == 1:
                    depth += 1
                elif is_manual_label_tag(cleaned_tokens[j]) == 2:
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            end_token_idx = j
            # Collect tokens in span (excluding label and auto_label tokens)
            tokens_in_span = []
            for k in range(i + 1, j):
                if not (is_manual_label_tag(cleaned_tokens[k]) or is_auto_label_tag(cleaned_tokens[k])):
                    tokens_in_span.append(cleaned_tokens[k])
            end_pos = (j - 1) - sub_pos  # end position is last token in span minus sub_pos
            labels.append({
                'name': name,
                'parent': parent,
                'start_token_idx': start_token_idx,
                'end_token_idx': end_token_idx,
                'start_pos': start_pos,
                'end_pos': end_pos,
                'tokens': tokens_in_span,
                'attributes': attributes
            })
            # For every opening or closing label, increment sub_pos
            sub_pos += 1  # for this opening tag
            # Also count closing tag
            if j < len(cleaned_tokens) and is_manual_label_tag(cleaned_tokens[j]) == 2:
                sub_pos += 1
            i += 1
        elif is_manual_label_tag(token) == 2:
            sub_pos += 1
            i += 1
        else:
            i += 1
    return labels


# --- Span-level F1 with IoU ---
def iou(spanA, spanB):
    startA, endA = spanA
    startB, endB = spanB
    inter = max(0, min(endA, endB) - max(startA, startB))
    union = max(endA, endB) - min(startA, startB)
    return inter / union if union > 0 else 0.0

def match_spans(ref_spans, sys_spans, iou_thresh=0.5):
    matches = []
    used_sys = set()
    for i, r in enumerate(ref_spans):
        best_j = -1
        best_iou = 0.0
        for j, s in enumerate(sys_spans):
            if s['name'] != r['name'] or j in used_sys:
                continue
            score = iou((r['start_pos'], r['end_pos']), (s['start_pos'], s['end_pos']))
            if score >= iou_thresh and score > best_iou:
                best_iou = score
                best_j = j
        if best_j >= 0:
            matches.append((i, best_j, best_iou))
            used_sys.add(best_j)
    return matches

def attribute_f1(matches, spansA, spansB):
    # matches: list of (idxA, idxB, iou)
    # spansA, spansB: list of label dicts (with 'attributes')
    total_tp = 0
    total_fp = 0
    total_fn = 0
    per_label = {}
    for (idxA, idxB, _) in matches:
        attrsA = spansA[idxA].get('attributes', {})
        attrsB = spansB[idxB].get('attributes', {})
        keys = set(attrsA.keys()) | set(attrsB.keys())
        tp = sum(attrsA.get(k) == attrsB.get(k) for k in keys)
        fp = sum((k in attrsB and attrsA.get(k) != attrsB.get(k)) for k in keys)
        fn = sum((k in attrsA and attrsA.get(k) != attrsB.get(k)) for k in keys)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        for k in keys:
            if k not in per_label:
                per_label[k] = {'tp': 0, 'fp': 0, 'fn': 0}
            if attrsA.get(k) == attrsB.get(k):
                per_label[k]['tp'] += 1
            else:
                if k in attrsB:
                    per_label[k]['fp'] += 1
                if k in attrsA:
                    per_label[k]['fn'] += 1
    micro_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * micro_prec * micro_rec / (micro_prec + micro_rec) if (micro_prec + micro_rec) > 0 else 0.0
    macro_prec = sum((v['tp'] / (v['tp'] + v['fp']) if (v['tp'] + v['fp']) > 0 else 0.0) for v in per_label.values()) / len(per_label) if per_label else 0.0
    macro_rec = sum((v['tp'] / (v['tp'] + v['fn']) if (v['tp'] + v['fn']) > 0 else 0.0) for v in per_label.values()) / len(per_label) if per_label else 0.0
    macro_f1 = (
        sum(
            (
                2 * (v['tp'] / (v['tp'] + v['fp']) if (v['tp'] + v['fp']) > 0 else 0.0) * (v['tp'] / (v['tp'] + v['fn']) if (v['tp'] + v['fn']) > 0 else 0.0)
                / (
                    (v['tp'] / (v['tp'] + v['fp']) if (v['tp'] + v['fp']) > 0 else 0.0)
                    + (v['tp'] / (v['tp'] + v['fn']) if (v['tp'] + v['fn']) > 0 else 0.0)
                )
                if ((v['tp'] + v['fp']) > 0 and (v['tp'] + v['fn']) > 0 and ((v['tp'] / (v['tp'] + v['fp'])) + (v['tp'] / (v['tp'] + v['fn']))) > 0)
                else 0.0
            )
            for v in per_label.values()
        ) / len(per_label)
        if per_label else 0.0
    )
    return {
        'micro': {'precision': round(micro_prec, 3), 'recall': round(micro_rec, 3), 'f1': round(micro_f1, 3)},
        'macro': {'precision': round(macro_prec, 3), 'recall': round(macro_rec, 3), 'f1': round(macro_f1, 3)},
        'per_attribute': per_label
    }

def compute_span_f1(docs, files=None, iou_thresh=0.5):
    if len(docs) < 2:
        return {}
    if files is None:
        files = [str(i) for i in range(len(docs))]
    results = {}
    sublabel_results = {}
    attribute_results = {}
    for i in range(len(docs)):
        for j in range(i+1, len(docs)):
            docA = docs[i]
            docB = docs[j]
            label_names = set(l['name'] for l in docA) | set(l['name'] for l in docB)
            pair_result = {}
            sublabel_pair_result = {}
            attribute_pair_result = {}
            for label in label_names:
                spansA = [l for l in docA if l['name'] == label]
                spansB = [l for l in docB if l['name'] == label]
                matches = match_spans(spansA, spansB, iou_thresh)
                tp = len(matches)
                fp = len(spansB) - tp
                fn = len(spansA) - tp
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                pair_result[label] = {
                    'precision': round(prec, 3),
                    'recall': round(rec, 3),
                    'f1': round(f1, 3),
                    'tp': tp, 'fp': fp, 'fn': fn,
                    'matches': matches
                }
                # Attribute agreement for parent
                attribute_pair_result[label] = attribute_f1(matches, spansA, spansB)
                # Level 2: sublabel agreement inside matched parent spans
                sublabel_agreement = {}
                sublabel_attribute_agreement = {}
                for (idxA, idxB, iou_val) in matches:
                    parentA = spansA[idxA]
                    parentB = spansB[idxB]
                    # Group sublabels by name
                    subA = parentA.get('sublabels', [])
                    subB = parentB.get('sublabels', [])
                    sublabel_types = set(l['name'] for l in subA) | set(l['name'] for l in subB)
                    for sublabel in sublabel_types:
                        subspansA = [l for l in subA if l['name'] == sublabel]
                        subspansB = [l for l in subB if l['name'] == sublabel]
                        submatches = match_spans(subspansA, subspansB, iou_thresh)
                        sub_tp = len(submatches)
                        sub_fp = len(subspansB) - sub_tp
                        sub_fn = len(subspansA) - sub_tp
                        sub_prec = sub_tp / (sub_tp + sub_fp) if (sub_tp + sub_fp) > 0 else 0.0
                        sub_rec = sub_tp / (sub_tp + sub_fn) if (sub_tp + sub_fn) > 0 else 0.0
                        sub_f1 = 2 * sub_prec * sub_rec / (sub_prec + sub_rec) if (sub_prec + sub_rec) > 0 else 0.0
                        sublabel_agreement[sublabel] = {
                            'precision': round(sub_prec, 3),
                            'recall': round(sub_rec, 3),
                            'f1': round(sub_f1, 3),
                            'tp': sub_tp, 'fp': sub_fp, 'fn': sub_fn,
                            'matches': submatches
                        }
                        # Attribute agreement for sublabel
                        sublabel_attribute_agreement[sublabel] = attribute_f1(submatches, subspansA, subspansB)
                if sublabel_agreement:
                    sublabel_pair_result[label] = sublabel_agreement
                if sublabel_attribute_agreement:
                    attribute_pair_result[f'{label}__sublabels'] = sublabel_attribute_agreement
            results[f'{files[i]} vs {files[j]}'] = pair_result
            if sublabel_pair_result:
                sublabel_results[f'{files[i]} vs {files[j]}'] = sublabel_pair_result
            if attribute_pair_result:
                attribute_results[f'{files[i]} vs {files[j]}'] = attribute_pair_result
    return {'parent_f1': results, 'sublabel_f1': sublabel_results, 'attribute_f1': attribute_results}

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Build label-aware document structure and compute span-level F1 agreement')
    parser.add_argument('files', nargs='+', help='HTML files to analyze (1 or more)')
    parser.add_argument('--output', '-o', help='Output JSON file (default: stdout)')
    parser.add_argument('--iou', type=float, default=None, help='IoU threshold for span match (default 0.5)')
    parser.add_argument('--min-overlap', type=float, default=None, help='Alias for --iou (for compatibility)')
    args = parser.parse_args()
    # Use whichever is provided, default to 0.5
    iou_thresh = args.iou if args.iou is not None else (args.min_overlap if args.min_overlap is not None else 0.5)
    docs = []
    files = args.files
    for f in files:
        tokens, cleaned = load_and_tokenize(f)
        labels = build_labelaware_doc(cleaned)
        docs.append(labels)
    f1_results = compute_span_f1(docs, files=files, iou_thresh=iou_thresh)
    output = {'docs': docs, 'span_f1': f1_results}
    output_json = json.dumps(output, indent=2, ensure_ascii=False)
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as out:
            out.write(output_json)
        print(f"Results written to {args.output}")
    else:
        import sys
        if sys.platform == 'win32':
            import codecs
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        with open("output.json", 'w', encoding='utf-8') as stdout:
            json.dump(output, stdout, indent=2, ensure_ascii=False)
        print("Results written to output.json")
        #print(output_json)

if __name__ == "__main__":
    main()
