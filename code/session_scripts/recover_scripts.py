import json
import ntpath
import sys

want = sys.argv[1:]
seen = {}
for line in open(r'C:/Users/TylerM/.claude/projects/D--DNA-to-Display/196eda90-cf5d-4266-a917-5c0f45e4b1a0.jsonl', encoding='utf-8'):
    try:
        d = json.loads(line)
    except Exception:
        continue
    m = d.get('message', {})
    content = m.get('content') if isinstance(m, dict) else None
    if not isinstance(content, list):
        continue
    for c in content:
        if isinstance(c, dict) and c.get('type') == 'tool_use' and c.get('name') in ('Write', 'Edit'):
            fp = c['input'].get('file_path', '')
            if ntpath.basename(fp) in want:
                seen.setdefault(fp, []).append((c['name'], c['input']))
for fp, ops in seen.items():
    name = 'old_' + ntpath.basename(fp)
    print(fp, [o[0] for o in ops])
    for op, inp in ops:
        if op == 'Write':
            s = inp['content']
        else:
            s = s.replace(inp['old_string'], inp['new_string'])
    open(name, 'w', encoding='utf-8').write(s)
