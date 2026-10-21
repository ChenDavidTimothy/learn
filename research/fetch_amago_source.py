#!/usr/bin/env python3
"""Fetch a pinned MIT-licensed official comparator; never overwrite a local edit."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import urllib.request

COMMIT = '54c25ab6da9371614c47352f569a56c0fe938d3b'
BASE = 'https://api.github.com/repos/UT-Austin-RPL/amago'
DEST = Path(__file__).resolve().parent/'third_party'/'amago'


def main():
    tree = json.load(urllib.request.urlopen(f'{BASE}/git/trees/{COMMIT}?recursive=1'))
    entries = [x for x in tree['tree'] if x['type'] == 'blob' and (
        x['path'].startswith('amago/') and x['path'].endswith('.py') or
        x['path'] in ('LICENSE', 'pyproject.toml', 'examples/01_basic_gym.py', 'examples/02_gymnax.py'))]

    def fetch(entry):
        url = f"https://raw.githubusercontent.com/UT-Austin-RPL/amago/{COMMIT}/{entry['path']}"
        content = urllib.request.urlopen(url).read()
        assert hashlib.sha1(f'blob {len(content)}\0'.encode()+content).hexdigest() == entry['sha']
        path = DEST/entry['path']
        if path.exists():
            assert path.read_bytes() == content, f'preserve modified local source: {path}'
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        return dict(path=entry['path'], git_blob=entry['sha'], bytes=len(content))

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        files = list(pool.map(fetch, entries))
    manifest = dict(upstream=BASE.removeprefix('https://api.github.com/repos/'),
                    commit=COMMIT, license='MIT; original LICENSE preserved', files=files)
    (DEST/'UPSTREAM.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(dict(commit=COMMIT, files=len(files), source_bytes=sum(x['bytes'] for x in files))))


if __name__ == '__main__':
    main()
