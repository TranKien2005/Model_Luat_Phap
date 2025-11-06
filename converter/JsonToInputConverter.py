import os
import json
import argparse
import re
from typing import List, Dict, Any, Optional


def normalize_doc(obj: Any) -> Dict[str, Any]:
    """Return a document dict that represents a legal doc.
    Handle common wrappers like {"content": {"law": ...}} or {"law": ...}.
    If the object already looks like a document, return it as-is.
    """
    if not obj:
        return {}
    if isinstance(obj, dict):
        if 'content' in obj and isinstance(obj['content'], dict):
            # try content->law or content->related_documents
            if 'law' in obj['content'] and isinstance(obj['content']['law'], dict):
                return obj['content']['law']
        if 'law' in obj and isinstance(obj['law'], dict):
            return obj['law']
        # If it already has structure or type fields, assume it's a doc
        if 'structure' in obj or 'type' in obj:
            return obj
    # fallback
    return {"structure": []}


def load_json_file(path: str) -> Any:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def collect_from_paths(paths: List[str]) -> List[Dict[str, Any]]:
    docs = []
    for p in paths or []:
        if not p:
            continue
        if not os.path.exists(p):
            print(f"Warning: file not found: {p}")
            continue
        try:
            obj = load_json_file(p)
            doc = normalize_doc(obj)
            # carry some top-level fields if available in wrapper
            if isinstance(obj, dict) and not doc.get('title'):
                # try some common locations
                title = obj.get('title') or obj.get('name')
                if title:
                    doc['title'] = title
            docs.append(doc)
        except Exception as e:
            print(f"Warning: could not read {p}: {e}")
    return docs


def interactive_dialog() -> Dict[str, List[str]]:
    """Open sequential file dialogs to pick files. Returns a dict with keys 'law', 'decrees', 'resolutions', 'circulars'.
    If tkinter is not available or user cancels, empty lists/None are returned.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox
    except Exception as e:
        print(f"Tkinter not available: {e}")
        return {"law": [], "decrees": [], "resolutions": [], "circulars": []}

    root = tk.Tk()
    root.withdraw()

    # Select main law (single)
    law_path = filedialog.askopenfilename(title='Select main law JSON (single file)', filetypes=[('JSON files','*.json'),('All files','*.*')])
    if not law_path:
        # user cancelled main law selection -> return empty
        root.destroy()
        return {"law": [], "decrees": [], "resolutions": [], "circulars": []}

    # Select multiple decrees
    decrees = filedialog.askopenfilenames(title='Select decree JSON files (multi-select, optional)', filetypes=[('JSON files','*.json'),('All files','*.*')])
    # Select multiple resolutions
    resolutions = filedialog.askopenfilenames(title='Select resolution JSON files (multi-select, optional)', filetypes=[('JSON files','*.json'),('All files','*.*')])
    # Select multiple circulars
    circulars = filedialog.askopenfilenames(title='Select circular JSON files (multi-select, optional)', filetypes=[('JSON files','*.json'),('All files','*.*')])

    root.destroy()

    return {
        'law': [law_path] if law_path else [],
        'decrees': list(decrees or []),
        'resolutions': list(resolutions or []),
        'circulars': list(circulars or [])
    }


def build_output(law_doc: Dict[str, Any], related_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Build metadata with 'id' first (taken from law_doc if present), then other fields.
    # This ensures the output JSON writes the 'id' field at the top of the metadata block.
    metadata = {
        "law_id": "",
        "version_id": "",
        "status": "",
        "last_updated": ""
    }

    out = {
        "metadata": metadata,
        "content": {
            "law": law_doc or {"structure": []},
            "related_documents": related_docs
        }
    }
    return out


def normalize_texts_in_doc(doc: Dict[str, Any]) -> None:
    """Collapse whitespace/newlines inside article and clause texts so each
    article/clause becomes a single-line string.
    Modifies the dict in-place.
    """
    if not isinstance(doc, dict):
        return

    def collapse(s: Optional[str]) -> str:
        if not s:
            return ""
        return re.sub(r"\s+", " ", s).strip()

    for ch in doc.get('structure', []) or []:
        if not isinstance(ch, dict):
            continue
        for art in ch.get('articles', []) or []:
            if not isinstance(art, dict):
                continue
            if art.get('text'):
                art['text'] = collapse(art.get('text'))
            for cl in art.get('clauses', []) or []:
                if not isinstance(cl, dict):
                    continue
                if cl.get('text'):
                    cl['text'] = collapse(cl.get('text'))


def main():
    parser = argparse.ArgumentParser(description='Combine JSON legal docs into a sample_input-style JSON')
    parser.add_argument('--law', help='Path to main law JSON (if omitted, GUI will ask)')
    parser.add_argument('--decrees', nargs='*', help='Paths to decree JSON files (optional)')
    parser.add_argument('--resolutions', nargs='*', help='Paths to resolution JSON files (optional)')
    parser.add_argument('--circulars', nargs='*', help='Paths to circular JSON files (optional)')
    parser.add_argument('--output', '-o', help='Output JSON path (if omitted, GUI will ask or default to inputs/combined_input.json)')
    args = parser.parse_args()

    selections = {"law": [], "decrees": [], "resolutions": [], "circulars": []}

    if args.law or args.decrees or args.resolutions or args.circulars:
        # collect from CLI args
        if args.law:
            selections['law'] = [args.law]
        if args.decrees:
            selections['decrees'] = args.decrees
        if args.resolutions:
            selections['resolutions'] = args.resolutions
        if args.circulars:
            selections['circulars'] = args.circulars
    else:
        # interactive GUI mode
        selections = interactive_dialog()
        if not selections['law']:
            print('No main law selected; aborting.')
            return

    law_docs = collect_from_paths(selections.get('law', []))
    if not law_docs:
        print('Could not load main law JSON; aborting.')
        return
    main_law = law_docs[0]

    # Ensure texts inside articles/clauses are single-line (collapse newlines/extra spaces)
    normalize_texts_in_doc(main_law)

    related = []
    # collect and tag each group with type if available (we don't force type but prefer to keep)
    for p in collect_from_paths(selections.get('decrees', [])):
        if p:
            # ensure type field
            if not p.get('type'):
                p['type'] = 'decree'
            related.append(p)
    for p in collect_from_paths(selections.get('resolutions', [])):
        if p:
            if not p.get('type'):
                p['type'] = 'resolution'
            related.append(p)
    for p in collect_from_paths(selections.get('circulars', [])):
        if p:
            if not p.get('type'):
                p['type'] = 'circular'
            related.append(p)

    # Normalize related docs as well so their article texts are single-line
    for rd in related:
        normalize_texts_in_doc(rd)

    out_obj = build_output(main_law, related)

    output_path = args.output
    if not output_path:
        # default location
        default = os.path.join(os.path.dirname(__file__), '..', 'inputs', 'combined_input.json')
        default = os.path.normpath(default)
        try:
            # ask save dialog if GUI available
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            save = filedialog.asksaveasfilename(defaultextension='.json', initialfile='combined_input.json', initialdir=os.path.dirname(default), filetypes=[('JSON files','*.json'),('All files','*.*')], title='Save combined JSON as')
            root.destroy()
            if save:
                output_path = save
            else:
                output_path = default
        except Exception:
            output_path = default

    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Write output while keeping article objects compact (one article per line)
    with open(output_path, 'w', encoding='utf-8') as f:
        def j(v):
            return json.dumps(v, ensure_ascii=False)

        f.write('{' + '\n')
        # metadata (write with explicit key so output starts with "metadata")
        meta_raw = json.dumps(out_obj.get('metadata', {}), ensure_ascii=False, indent=2)
        meta_block = meta_raw.replace('\n', '\n  ')
        f.write('  "metadata": ' + meta_block + ',\n')

        # content start
        f.write('  "content": {\n')

        # law
        law = out_obj.get('content', {}).get('law', {}) or {}
        f.write('    "law": {\n')

        # write simple law fields except structure
        simple_keys = [k for k in law.keys() if k != 'structure']
        # keep a stable ordering for common fields if present; ensure 'id' is first
        preferred = ['id', 'type', 'issuer', 'title', 'source_url', 'promulgation_date', 'effective_date', 'status']
        ordered = [k for k in preferred if k in simple_keys] + [k for k in simple_keys if k not in preferred]
        for key in ordered:
            f.write(f'      "{key}": {j(law.get(key, ""))},\n')

        # structure
        f.write('      "structure": [\n')
        structs = law.get('structure', []) or []
        for ci, ch in enumerate(structs):
            f.write('        {\n')
            f.write(f'          "type": {j(ch.get("type"))},\n')
            f.write(f'          "number": {j(ch.get("number"))},\n')
            f.write(f'          "title": {j(ch.get("title", ""))},\n')
            f.write('          "articles": [\n')
            arts = ch.get('articles', []) or []
            for ai, art in enumerate(arts):
                # compact article JSON on one line
                art_compact = json.dumps(art, ensure_ascii=False, separators=(',', ': '))
                f.write('            ' + art_compact)
                f.write(',\n' if ai != len(arts) - 1 else '\n')
            f.write('          ]\n')
            f.write('        }')
            f.write(',\n' if ci != len(structs) - 1 else '\n')
        f.write('      ]\n')
        f.write('    },\n')

        # related_documents: write each related doc with compact articles (one line per article)
        related = out_obj.get('content', {}).get('related_documents', []) or []
        f.write('    "related_documents": [\n')
        for ri, rd in enumerate(related):
            f.write('      {\n')
            # write simple fields except structure
            simple_keys = [k for k in (rd.keys() if isinstance(rd, dict) else []) if k != 'structure']
            preferred = ['id', 'type', 'issuer', 'title', 'source_url', 'promulgation_date', 'effective_date', 'status']
            ordered = [k for k in preferred if k in simple_keys] + [k for k in simple_keys if k not in preferred]
            for key in ordered:
                f.write(f'        "{key}": {j(rd.get(key, ""))},\n')

            # structure for this related doc
            f.write('        "structure": [\n')
            r_structs = (rd.get('structure') if isinstance(rd, dict) else []) or []
            for ci, ch in enumerate(r_structs):
                f.write('          {\n')
                f.write(f'            "type": {j(ch.get("type"))},\n')
                f.write(f'            "number": {j(ch.get("number"))},\n')
                f.write(f'            "title": {j(ch.get("title", ""))},\n')
                f.write('            "articles": [\n')
                r_arts = ch.get('articles', []) or []
                for ai, art in enumerate(r_arts):
                    art_compact = json.dumps(art, ensure_ascii=False, separators=(',', ': '))
                    f.write('              ' + art_compact)
                    f.write(',\n' if ai != len(r_arts) - 1 else '\n')
                f.write('            ]\n')
                f.write('          }')
                f.write(',\n' if ci != len(r_structs) - 1 else '\n')
            f.write('        ]\n')
            f.write('      }')
            f.write(',\n' if ri != len(related) - 1 else '\n')
        f.write('    ]\n')

        # content end
        f.write('  }\n')
        f.write('}\n')

    print(f'Wrote combined file: {output_path}')

#First choose main Law, then choose each type of related documents -> combine to input json input form
if __name__ == '__main__':
    main()
