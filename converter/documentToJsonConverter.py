import os
import re
import json
import argparse
from typing import List, Dict, Any, Optional




ROMAN_MAP = {
    'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5,
    'VI': 6, 'VII': 7, 'VIII': 8, 'IX': 9, 'X': 10,
    'XI': 11, 'XII': 12, 'XIII': 13, 'XIV': 14, 'XV': 15
}


def roman_to_int(r: str) -> Optional[int]:
    r = r.strip().upper()
    return ROMAN_MAP.get(r)


def parse_law_text(lines: List[str]) -> Dict[str, Any]:
    # Keep metadata fields but leave them empty per user request.
    law: Dict[str, Any] = {
        "type": "",
        "issuer": "",
        "title": "",
        "source_url": "",
        "promulgation_date": "",
        "effective_date": "",
        "status": "",
        "structure": []
    }

    current_chapter = None
    current_article = None
    collecting_article_text: List[str] = []

    # helper to flush article into chapter
    def flush_article():
        nonlocal current_article, collecting_article_text
        if current_article is None:
            return
        # join text collected
        text = "\n".join([l for l in collecting_article_text]).strip()
        current_article.setdefault('text', '')
        if text and (not current_article.get('clauses')):
            # if there are clauses, keep article text empty per user's notes
            current_article['text'] = text
        elif not current_article.get('clauses'):
            current_article['text'] = text
        collecting_article_text = []
        # append to chapter
        if current_chapter is not None:
            current_chapter.setdefault('articles', []).append(current_article)
        current_article = None

    # Note: per user request, do NOT auto-fill metadata (issuer/title/dates).
    # This parser will only extract structural elements (Chương, Điều, khoản).

    # Main parse pass
    for raw in lines:
        line = raw.rstrip('\n')
        s = line.strip()
        if not s:
            # blank line -> keep collecting but may indicate paragraph break
            if collecting_article_text:
                collecting_article_text.append('')
            continue

        # Chapter detection: lines like "Chương I" or "Chương I" then title next
        m_ch = re.match(r'Chương\s+([IVXLCDM]+)\b', s, re.IGNORECASE)
        if m_ch:
            # flush any pending article
            flush_article()
            roman = m_ch.group(1)
            num = roman_to_int(roman) or roman
            current_chapter = {"type": "chapter", "number": num, "title": "", "articles": []}
            law['structure'].append(current_chapter)
            continue

        # Chapter title: uppercase and short, when we have last chapter without title
        if current_chapter is not None and (not current_chapter.get('title')):
            # heuristic: uppercase and not starting with 'Điều' and not 'Chương'
            if s and len(s) < 120 and s.upper() == s and not s.startswith('Điều'):
                # preserve original casing (user prefers original text)
                current_chapter['title'] = s.strip()
                continue

        # Article detection: "Điều 3. Tính chất, nguyên lý giáo dục"
        m_art = re.match(r'Điều\s+(\d+)\.\s*(.*)', s)
        if m_art:
            # flush previous
            flush_article()
            a_num = int(m_art.group(1))
            a_title = m_art.group(2).strip()
            # user wants the title to include the "Điều N." prefix for compactness
            full_title = f"Điều {a_num}. {a_title}" if a_title else f"Điều {a_num}."
            current_article = {"number": a_num, "title": full_title, "text": "", "clauses": []}
            # start collecting following lines as article text until clause or next article
            collecting_article_text = []
            continue

        # Clause detection: lines starting with "1." or "1. " at line start
        m_clause = re.match(r'^(\d+)\.\s*(.*)', s)
        if m_clause and current_article is not None:
            # flush collecting text into article.text if any and article had no clauses yet
            text_before = '\n'.join(collecting_article_text).strip()
            if text_before and not current_article['clauses']:
                # keep article text separate
                current_article['text'] = text_before
            collecting_article_text = []
            cnum = int(m_clause.group(1))
            ctext = m_clause.group(2).strip()
            current_article['clauses'].append({"number": cnum, "text": ctext})
            continue

        # Sometimes clauses continue on following lines (wrapped paragraphs)
        # If we currently are inside a clause (last clause exists) and line is not a new article/chapter,
        # consider it continuation of last clause (if it doesn't look like a heading)
        if current_article is not None and current_article.get('clauses'):
            # continuation of last clause
            last = current_article['clauses'][-1]
            # append with space
            last['text'] = (last.get('text', '') + ' ' + s).strip()
            continue

        # If we're inside an article but haven't seen clauses yet, collect paragraph lines
        if current_article is not None:
            collecting_article_text.append(s)
            continue

        # We intentionally do not auto-fill top-level metadata fields here
        # (issuer/title/dates) — only extract structural elements.

    # end for
    # flush last article
    flush_article()

    return law


def merge_metadata(target: Dict[str, Any], meta: Dict[str, Any]) -> None:
    # Merge top-level simple metadata fields if provided
    for k in ('issuer', 'title', 'source_url', 'promulgation_date', 'effective_date', 'status'):
        if meta.get(k):
            target[k] = meta[k]


def main():
    parser = argparse.ArgumentParser(description='Convert law TXT to structured JSON')
    # allow multiple input files; if omitted, open a file picker dialog (multiple select)
    parser.add_argument('input', nargs='*', help='Input TXT file(s). If omitted, a file dialog will open to select one or more files.')
    parser.add_argument('output', nargs='?', help='Output JSON file or directory (optional). If omitted, uses converter/<input_base>.json')
    parser.add_argument('--metadata', '-m', help='Optional metadata JSON to merge')
    args = parser.parse_args()

    input_args = args.input or []

    # If no input provided on CLI, open file dialog to select multiple files
    if not input_args:
        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            filetypes = [('Text files', '*.txt'), ('All files', '*.*')]
            files = filedialog.askopenfilenames(title='Select input TXT file(s)', filetypes=filetypes)
            root.destroy()
            input_args = list(files)
        except Exception as e:
            print(f"Could not open file dialog: {e}")
            parser.error('No input file provided and file dialog failed.')

    if not input_args:
        parser.error('No input file selected.')

    # decide output behavior: if args.output is provided and multiple inputs,
    # treat output as a directory; otherwise derive per-input output path
    output_arg = args.output

    # optional metadata content (applies per-file if provided)
    meta_content = None
    if args.metadata:
        try:
            with open(args.metadata, 'r', encoding='utf-8') as mf:
                meta_json = json.load(mf)
                # allow metadata to be nested under content->law or be a flat object
                meta_content = meta_json.get('content', {}).get('law', meta_json)
        except Exception as e:
            print(f"Warning: could not read metadata file: {e}")

    # process each input file separately
    input_paths = input_args
    for input_path in input_paths:
        with open(input_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        law_obj = parse_law_text(lines)

        # Normalize texts: collapse internal newlines/whitespace so that each
        # article.text and each clause.text is a single-line paragraph (breaks
        # only occur between articles in the structure array).
        def collapse(s: str) -> str:
            return re.sub(r"\s+", " ", s).strip()

        for ch in law_obj.get('structure', []):
            for art in ch.get('articles', []):
                if art.get('text'):
                    art['text'] = collapse(art['text'])
                for cl in art.get('clauses', []):
                    if cl.get('text'):
                        cl['text'] = collapse(cl['text'])

        # if metadata provided, merge into this law_obj
        if meta_content:
            merge_metadata(law_obj, meta_content)

        # determine output path for this input
        if output_arg:
            # if multiple inputs, treat provided output as directory (create if needed)
            if len(input_paths) > 1:
                out_dir = output_arg
                if out_dir and not os.path.exists(out_dir):
                    os.makedirs(out_dir, exist_ok=True)
                base = os.path.splitext(os.path.basename(input_path))[0]
                output_path = os.path.join(out_dir, base + '.json')
            else:
                # single input: if output_arg is a directory, write inside it, otherwise use as file path
                if os.path.isdir(output_arg) or output_arg.endswith(os.path.sep):
                    out_dir = output_arg
                    if out_dir and not os.path.exists(out_dir):
                        os.makedirs(out_dir, exist_ok=True)
                    base = os.path.splitext(os.path.basename(input_path))[0]
                    output_path = os.path.join(out_dir, base + '.json')
                else:
                    output_path = output_arg
        else:
            base = os.path.splitext(os.path.basename(input_path))[0]
            output_path = os.path.join(os.path.dirname(__file__), base + '.json')

        # ensure output directory exists
        out_dir = os.path.dirname(output_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        # Write top-level JSON directly (no outer "law" wrapper).
        # Produce structure where each article object is compact (single line) —
        # no internal line breaks inside an article.
        with open(output_path, 'w', encoding='utf-8') as of:
            def j(v):
                return json.dumps(v, ensure_ascii=False)

            of.write('{' + '\n')
            # write simple top-level fields
            top_fields = ['type', 'issuer', 'title', 'source_url', 'promulgation_date', 'effective_date', 'status']
            for fld in top_fields:
                of.write(f'  "{fld}": {j(law_obj.get(fld, ""))},\n')

            # structure
            of.write('  "structure": [\n')
            structs = law_obj.get('structure', [])
            for ci, ch in enumerate(structs):
                of.write('    {\n')
                of.write(f'      "type": {j(ch.get("type"))},\n')
                of.write(f'      "number": {j(ch.get("number"))},\n')
                of.write(f'      "title": {j(ch.get("title", ""))},\n')
                of.write('      "articles": [\n')
                arts = ch.get('articles', [])
                for ai, art in enumerate(arts):
                    # compact article JSON on one line
                    art_compact = json.dumps(art, ensure_ascii=False, separators=(',', ': '))
                    of.write('        ' + art_compact)
                    of.write(',\n' if ai != len(arts) - 1 else '\n')
                of.write('      ]\n')
                of.write('    }')
                of.write(',\n' if ci != len(structs) - 1 else '\n')
            of.write('  ]\n')
            of.write('}\n')

        print(f"Wrote {output_path}")

#Choose txt document -> convert to json in this folder
if __name__ == '__main__':
    main()
