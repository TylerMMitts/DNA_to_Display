# Keep only the hunks of a git diff that contain any of the given markers.
import re
import sys

src, dst, *markers = sys.argv[1:]
text = open(src, encoding='utf-8', newline='').read()
head, *hunks = re.split(r'(?m)^(?=@@ )', text)
keep = [h for h in hunks if any(m in h for m in markers)]
print(f'{len(keep)} of {len(hunks)} hunks kept')
for h in keep:
    print('  ', h.splitlines()[0])
open(dst, 'w', encoding='utf-8', newline='').write(head + ''.join(keep))
