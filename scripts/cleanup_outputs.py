"""Remove old completed request directories; dry-run unless --apply is supplied."""
import argparse
from pathlib import Path
import re
import shutil
import time


def expired_outputs(root, age_hours=24, keep=100):
    if age_hours < 0 or keep < 0:
        raise ValueError('Age and retained count must be non-negative')
    completed = []
    for marker in Path(root).rglob('.completed'):
        directory = marker.parent
        if not re.fullmatch('[0-9a-f]{32}', directory.name):
            continue
        if any(p.is_symlink() for p in (marker, directory, *directory.parents)):
            continue
        completed.append((marker.stat().st_mtime, directory))
    completed.sort(reverse=True)
    now = time.time()
    return [p for index,(modified,p) in enumerate(completed)
            if index >= keep and now - modified >= age_hours*3600]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='outputs/predictions')
    parser.add_argument('--age-hours', type=float, default=24)
    parser.add_argument('--keep', type=int, default=100)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    for directory in expired_outputs(args.root,args.age_hours,args.keep):
        print(('Remove ' if args.apply else 'Would remove ')+str(directory))
        if args.apply:
            shutil.rmtree(directory)

if __name__ == '__main__':
    main()
