#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Copy anh co file label YOLO RONG de kiem tra bang mat (Python >= 3.9).

Khong can cai thu vien, khong can YAML.
Chay: python copy_empty_label_images.py
Hoac: python copy_empty_label_images.py --dataset "D:/Dataset" --output "D:/Review"

Nhan folder chua train/valid/test (hoac val), folder chua images/labels,
hoac truc tiep folder images. Ho tro thu muc con trong images va labels.
Chi copy anh co file .txt cung ten RONG, hoac chi chua khoang trang/xuong dong.
UTF-8 BOM duoc bo qua khi xac dinh rong. Anh thieu .txt KHONG duoc copy.
Anh co noi dung bat ky trong .txt KHONG duoc copy, ke ca nhan sai dinh dang.
Giu nguyen file anh va dataset goc; ket qua chi copy anh, khong copy label.
Bao cao CSV ghi duong dan anh/label goc de tim lai khi can sua annotation.
Thu muc ket qua phai chua ton tai va nam ngoai dataset goc.
"""

import argparse
from collections import Counter
import csv
from pathlib import Path
import shutil
import sys
import tempfile

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
SPLITS = ('train', 'valid', 'val', 'test')


def clean_path(text):
    value = str(text).strip().strip('"').strip("'")
    if not value:
        raise ValueError('Duong dan khong duoc de trong.')
    return Path(value).expanduser().resolve()


def normalize_root(value):
    root = clean_path(value)
    # Allow the images path the user may paste from File Explorer.
    return root.parent if root.name.lower() == 'images' else root


def find_layout(root):
    if not root.is_dir():
        raise ValueError(f'Khong tim thay dataset: {root}')
    split_dirs = [(split, root / split / 'images', root / split / 'labels')
                  for split in SPLITS if (root / split / 'images').is_dir()]
    if (root / 'images').is_dir():
        if split_dirs:
            raise ValueError('Co ca images truc tiep va images trong split; '
                             'hay chon mot dataset cu the.')
        return [('', root / 'images', root / 'labels')]
    if split_dirs:
        for split in SPLITS:
            if (root / split).is_dir() and not (root / split / 'images').is_dir():
                raise ValueError(f'Thieu folder images trong {root / split}')
        return split_dirs
    raise ValueError('Can folder chua images/labels hoac train/valid/test/images.')


def copy_empty_labels(dataset, output):
    root, output = normalize_root(dataset), clean_path(output)
    if output == root or root in output.parents:
        raise ValueError('Folder ket qua phai nam NGOAI dataset goc.')
    if output.exists():
        raise ValueError(f'Khong ghi de folder da ton tai: {output}')
    layout = find_layout(root)
    selected, totals = [], {}
    for split, image_dir, label_dir in layout:
        counts = Counter(total=0, empty=0, nonempty=0, missing=0)
        totals[split or 'dataset'] = counts
        print(f"Dang quet {split or 'dataset'}...", flush=True)
        for image in sorted(image_dir.rglob('*')):
            if not image.is_file() or image.suffix.lower() not in EXTENSIONS:
                continue
            relative = image.relative_to(image_dir)
            label = label_dir / relative.with_suffix('.txt')
            counts['total'] += 1
            if not label.exists():
                counts['missing'] += 1
                continue
            if not label.is_file():
                raise ValueError(f'Duong dan label khong phai file: {label}')
            # Match the whitespace/BOM handling of train_both.py.
            if label.read_text(encoding='utf-8-sig').strip():
                counts['nonempty'] += 1
                continue
            counts['empty'] += 1
            selected.append((split, image, label, relative))
    if not sum(counts['total'] for counts in totals.values()):
        raise ValueError('Khong tim thay anh duoc ho tro.')
    print(f'Se sao chep {len(selected)} anh co label rong.', flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.empty_label_review_', dir=output.parent))
    try:
        for split, _, _ in layout:
            (staging / split / 'images').mkdir(parents=True)
        with (staging / 'empty_label_images.csv').open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['split', 'original_image', 'original_label', 'copied_image'])
            for number, (split, image, label, relative) in enumerate(selected, 1):
                dest_relative = Path(split) / 'images' / relative
                dest = staging / dest_relative
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(image, dest)
                writer.writerow([split or 'dataset', str(image), str(label), str(output / dest_relative)])
                if number % 1000 == 0:
                    print(f'Da copy {number}/{len(selected)} anh...', flush=True)
        report = [f'Dataset: {root}', f'Output: {output}',
                  'Chi lay anh co file label rong. Anh thieu label khong duoc copy.']
        for split, counts in totals.items():
            report.append(f"{split}: {counts['total']} anh | copy label rong={counts['empty']} | "
                          f"bo qua label co noi dung={counts['nonempty']} | "
                          f"bo qua thieu label={counts['missing']}")
        (staging / 'summary.txt').write_text('\n'.join(report) + '\n', encoding='utf-8')
        output.mkdir()  # Never overwrite a path created by a concurrent run.
        try:
            for child in staging.iterdir():
                shutil.move(str(child), str(output / child.name))
        except BaseException:
            shutil.rmtree(output)
            raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    print('\n' + '\n'.join(report))
    print('HOAN TAT. Anh goc va label goc khong thay doi.')
    print('Xem empty_label_images.csv de tim lai duong dan goc neu can sua nhan.')
    return totals


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', help='Folder dataset hoac images; bo trong de nhap khi chay.')
    parser.add_argument('--output', help='Folder moi; mac dinh <dataset>_empty_labels.')
    args = parser.parse_args()
    try:
        value = args.dataset if args.dataset is not None else input('Nhap duong dan DATASET: ')
        root = normalize_root(value)
        default_output = root.with_name(root.name + '_empty_labels')
        output_value = args.output
        if output_value is None and args.dataset is None:
            output_value = input(f'Folder ket qua (Enter = {default_output}): ').strip()
        output = clean_path(output_value) if output_value else default_output
        copy_empty_labels(root, output)
    except (OSError, ValueError, EOFError) as error:
        print(f'LOI: {error}', file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print('\nDa dung. Dataset goc khong thay doi.', file=sys.stderr)
        return 130
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
