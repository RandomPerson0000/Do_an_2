#!/usr/bin/env python3
"""Copy a Roboflow bbox dataset, remove other, remap fire=0/smoke=1.

Requires: python -m pip install pyyaml
Run without arguments for Vietnamese interactive prompts.
Expected layout: ROOT/{train,valid (or val),test}/{images,labels}/...
Source files are never modified. Destination must be absent or empty.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import shutil
import tempfile

try:
    import yaml
except ImportError:
    raise SystemExit('Cai thu vien: python -m pip install pyyaml')

DEFAULT_SOURCE = r'D:\Do_An\\Fire_Smoke_develop\\datasets\\3classes'
DEFAULT_DEST = r'D:\Do_An\\Fire_Smoke_develop\\datasets\\Data_tune'
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}


def parse_label(path, names):
    rows = []
    for line_no, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f'Dong {line_no}: can bbox 5 cot, khong nhan polygon')
        try:
            class_id = int(fields[0])
            coords = [float(x) for x in fields[1:]]
        except ValueError:
            raise ValueError(f'Dong {line_no}: so khong hop le')
        if class_id not in names:
            raise ValueError(f'Dong {line_no}: class {class_id} khong co trong YAML')
        if not all(math.isfinite(x) and 0 <= x <= 1 for x in coords):
            raise ValueError(f'Dong {line_no}: toa do phai trong [0,1]')
        if coords[2] <= 0 or coords[3] <= 0:
            raise ValueError(f'Dong {line_no}: bbox rong')
        rows.append((names[class_id], fields[1:]))
    return rows


def convert(source, destination, yaml_path, mode):
    source, destination = source.resolve(), destination.resolve()
    if not source.is_dir():
        raise ValueError(f'Khong co thu muc nguon: {source}')
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError('Nguon va dich phai tach biet, khong long vao nhau')
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError(f'Dich da co du lieu: {destination}. Chon thu muc moi/rong de tranh ghi de.')
    config = yaml.safe_load(yaml_path.read_text(encoding='utf-8-sig'))
    raw_names = config['names']
    names = dict(enumerate(raw_names)) if isinstance(raw_names, list) else {int(k): v for k, v in raw_names.items()}
    names = {k: str(v).strip().lower() for k, v in names.items()}
    if len(names) != 3 or set(names.values()) != {'fire', 'other', 'smoke'}:
        raise ValueError(f'Can dung 3 class fire/other/smoke. YAML hien tai: {names}')
    if (source / 'val' / 'images').is_dir() and (source / 'valid' / 'images').is_dir():
        raise ValueError('Nguon co ca val va valid. Can chon lai cau truc de tranh trung split.')
    splits = [s for s in ('train', 'valid', 'val', 'test') if (source / s / 'images').is_dir()]
    if not splits:
        raise ValueError('Khong tim thay train/images, valid/images, val/images hoac test/images trong nguon')

    records, manifest, stats = [], [], {}
    for split in splits:
        image_root = source / split / 'images'
        counts = dict(scanned=0, selected=0, negatives=0, positives=0,
                      fire_boxes=0, smoke_boxes=0, removed_other_boxes=0, skipped=0, invalid=0)
        seen = set()
        for image in sorted(image_root.rglob('*')):
            if not image.is_file() or image.suffix.lower() not in EXTENSIONS:
                continue
            counts['scanned'] += 1
            relative = image.relative_to(image_root)
            key = relative.with_suffix('').as_posix().casefold()
            if key in seen:
                raise ValueError(f'Trung ten goc anh trong split {split}: {relative}')
            seen.add(key)
            label = source / split / 'labels' / relative.with_suffix('.txt')
            if not label.is_file():
                counts['invalid'] += 1
                manifest.append([split, relative.as_posix(), 'invalid', 'Thieu file label; khong tu coi la negative'])
                continue
            try:
                rows = parse_label(label, names)
            except (OSError, UnicodeError, ValueError) as exc:
                counts['invalid'] += 1
                manifest.append([split, relative.as_posix(), 'invalid', str(exc)])
                continue
            has_other = any(name == 'other' for name, _ in rows)
            targets = [(name, coords) for name, coords in rows if name != 'other']
            selected = mode == 'all' or (has_other and (mode == 'other' or not targets))
            if not selected:
                counts['skipped'] += 1
                manifest.append([split, relative.as_posix(), 'skipped', 'Khong thuoc che do da chon'])
                continue
            lines = [' '.join([str({'fire': 0, 'smoke': 1}[name]), *coords]) for name, coords in targets]
            content = '\n'.join(lines) + ('\n' if lines else '')
            records.append((image, split, relative, content))
            counts['selected'] += 1
            counts['positives' if targets else 'negatives'] += 1
            counts['fire_boxes'] += sum(name == 'fire' for name, _ in targets)
            counts['smoke_boxes'] += sum(name == 'smoke' for name, _ in targets)
            counts['removed_other_boxes'] += sum(name == 'other' for name, _ in rows)
            manifest.append([split, relative.as_posix(), 'positive' if targets else 'negative', ''])
        stats[split] = counts
    if not records:
        raise ValueError('Khong co anh hop le duoc chon. Kiem tra nguon/label/che do loc. ' + json.dumps(stats))

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=destination.name + '_tmp_', dir=destination.parent))
    try:
        for image, split, relative, content in records:
            out_image = staging / split / 'images' / relative
            out_label = staging / split / 'labels' / relative.with_suffix('.txt')
            out_image.parent.mkdir(parents=True, exist_ok=True)
            out_label.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image, out_image)
            out_label.write_text(content, encoding='utf-8')
        output_yaml = {'path': destination.as_posix(), 'nc': 2, 'names': {0: 'fire', 1: 'smoke'}}
        for split, counts in stats.items():
            if counts['selected']:
                output_yaml['val' if split in ('val', 'valid') else split] = f'{split}/images'
        (staging / 'data.yaml').write_text(yaml.safe_dump(output_yaml, sort_keys=False, allow_unicode=True), encoding='utf-8')
        report = {'source': str(source), 'source_yaml': str(yaml_path.resolve()), 'destination': str(destination),
                  'mode': mode, 'original_classes': names, 'output_classes': {0: 'fire', 1: 'smoke'}, 'splits': stats,
                  'note': 'Negatives are based on existing labels only. Visually check for unlabelled fire/smoke. Keep split separation when merging.'}
        (staging / 'filter_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        with (staging / 'filter_manifest.csv').open('w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['split', 'image', 'status', 'reason'])
            writer.writerows(manifest)
        if destination.exists():
            destination.rmdir()  # Only succeeds while still empty.
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f'Hoan tat: {destination}\nYAML: {destination / "data.yaml"}')
    print('Negative chi duoc xac dinh theo label. Hay kiem tra anh de loai nhan thieu lua/khoi.')
    if 'train' not in output_yaml or 'val' not in output_yaml:
        print('Chu y: bo loc khong co du train/val; can ghep vao dataset train/val day du truoc khi train.')
    return stats


def prompt_path(label, default):
    return Path((input(f'{label} [Enter = {default}]: ').strip() or default).strip('"'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--dest', type=Path)
    parser.add_argument('--yaml', type=Path, help='Mac dinh: SOURCE/data.yaml; chi doc anh tu cac split trong SOURCE')
    parser.add_argument('--mode', choices=['other', 'negative', 'all'])
    args = parser.parse_args()
    source = args.source or prompt_path('Thu muc dataset nguon', DEFAULT_SOURCE)
    destination = args.dest or prompt_path('Thu muc ket qua', DEFAULT_DEST)
    mode = args.mode
    if mode is None:
        print('1: Lay anh co other, giu fire/smoke neu co (mac dinh)')
        print('2: Chi lay anh chi co other -> negative thuan')
        print('3: Lay tat ca anh co label hop le, bo other (gom label rong)')
        choice = input('Chon 1/2/3 [1]: ').strip() or '1'
        if choice not in {'1', '2', '3'}:
            raise ValueError('Chi chon 1, 2 hoac 3')
        mode = {'1': 'other', '2': 'negative', '3': 'all'}[choice]
    convert(source, destination, args.yaml or source / 'data.yaml', mode)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as error:
        raise SystemExit(f'LOI: {error}')
