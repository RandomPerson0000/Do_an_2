#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lay anh khong nhan va anh co lua + nguon sang khac, GIU LABEL GOC.

Python >= 3.9. Cai thu vien: python -m pip install pyyaml pillow
Chay: python extract_negative_images.py
Script hoi dataset, YAML va folder ket qua.

- Anh thieu label / label rong: tu dong lay.
- Anh co annotation: hien cua so de chon GIU (lua + nguon sang khac) / BO QUA.
- Anh duoc giu se copy ca label goc neu co; khong xoa bbox, khong doi class ID.
- Label khong the cho biet co nguon sang CHUA gan nhan; can nguoi dung xem anh.
- --labeled-mode all lay TOAN BO anh co nhan (ca anh chi co lua), khong loc theo noi dung.
- Khong thay doi du lieu goc, khong ghi de folder ket qua da ton tai.
- Anh co lua + nguon sang la anh POSITIVE co nen de nham, khong phai negative rong.
- Anh thieu label van can kiem tra xem co lua chua gan nhan hay khong.

Ho tro dataset/train/images + labels, valid/val, test; hoac dataset/images + labels.
Co the nhap truc tiep folder images. YAML chi dung xac nhan class ID, khong doc
cac duong dan train/val/test tu YAML. Giu nguyen split va thu muc con khi copy.
"""

import argparse
from collections import Counter
import math
from pathlib import Path
import shutil
import sys
import tempfile

IMAGE_EXTENSIONS = {'.bmp', '.dng', '.jpeg', '.jpg', '.mpo', '.png',
                    '.tif', '.tiff', '.webp', '.pfm', '.heic'}


def clean_path(value):
    return Path(str(value).strip().strip('"').strip("'")).expanduser().resolve()


def read_fire_id(yaml_path):
    try:
        import yaml
    except ImportError:
        raise ValueError('Hay cai thu vien: python -m pip install pyyaml') from None
    try:
        config = yaml.safe_load(yaml_path.read_text(encoding='utf-8-sig'))
    except yaml.YAMLError as error:
        raise ValueError(f'YAML khong hop le: {error}') from None
    if not isinstance(config, dict):
        raise ValueError('YAML phai co muc names.')
    names = config.get('names')
    if isinstance(names, list):
        names = dict(enumerate(names))
    if not isinstance(names, dict) or not names:
        raise ValueError("YAML thieu names, vi du names: ['fire']")
    if any(type(key) is not int or key < 0 for key in names):
        raise ValueError('ID trong names phai la so nguyen khong am.')
    if set(names) != set(range(len(names))):
        raise ValueError('ID trong names phai lien tiep tu 0.')
    if 'nc' in config and config['nc'] != len(names):
        raise ValueError('nc khong khop so lop trong names.')
    matches = [key for key, name in names.items() if str(name).strip().lower() == 'fire']
    if len(matches) != 1:
        raise ValueError("YAML phai co dung mot lop ten 'fire'. Kiem tra names trong YAML.")
    return matches[0], names


def get_layout(root):
    splits = [(split, root / split / 'images', root / split / 'labels')
              for split in ('train', 'valid', 'val', 'test')
              if (root / split / 'images').is_dir()]
    flat = (root / 'images').is_dir()
    if flat and splits:
        raise ValueError('Dataset co ca images truc tiep va images trong split. '
                         'Hay chon dung mot thu muc dataset de tranh bo sot.')
    if flat:
        return [('', root / 'images', root / 'labels')]
    if splits:
        # Do not silently skip a split whose images folder is missing/misspelled.
        for split in ('train', 'valid', 'val', 'test'):
            if (root / split).is_dir() and not (root / split / 'images').is_dir():
                raise ValueError(f'Thieu thu muc images: {root / split}')
        return splits
    raise ValueError('Khong tim thay images. Chon thu muc chua images/labels '
                     'hoac chua train, valid, test voi images/labels ben trong.')


def label_ids(label, known_ids):
    ids = set()
    for number, line in enumerate(label.read_text(encoding='utf-8-sig').splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        location = f'{label}, dong {number}'
        if len(fields) != 5:
            raise ValueError(f'{location}: can 5 cot YOLO detection.')
        if not fields[0].isascii() or not fields[0].isdigit():
            raise ValueError(f'{location}: class ID khong hop le.')
        class_id = int(fields[0])
        if class_id not in known_ids:
            raise ValueError(f'{location}: ID {class_id} khong co trong YAML da chon.')
        try:
            coords = [float(value) for value in fields[1:]]
        except ValueError:
            raise ValueError(f'{location}: toa do khong phai so.') from None
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in coords):
            raise ValueError(f'{location}: toa do phai huu han va nam trong [0, 1].')
        if coords[2] <= 0 or coords[3] <= 0:
            raise ValueError(f'{location}: width va height phai lon hon 0.')
        ids.add(class_id)
    return ids


def make_preview(record, names, size):
    from PIL import Image, ImageDraw
    with Image.open(record['image']) as original:
        picture = original.convert('RGB')
    width, height = picture.size
    picture.thumbnail(size)
    draw = ImageDraw.Draw(picture)
    display_w, display_h = picture.size
    for line in record['label'].read_text(encoding='utf-8-sig').splitlines():
        fields = line.split()
        if not fields:
            continue
        class_id = int(fields[0])
        x, y, w, h = map(float, fields[1:])
        box = ((x-w/2)*display_w, (y-h/2)*display_h,
               (x+w/2)*display_w, (y+h/2)*display_h)
        draw.rectangle(box, outline='red', width=3)
        title = f'{class_id}: {names[class_id]}'
        # Class names such as fire are normally ASCII; fallback avoids font errors.
        title = title.encode('ascii', 'replace').decode('ascii')
        draw.text((max(0, box[0]), max(0, box[1]-13)), title, fill='red')
    return picture


def review_labeled(records, names):
    """Return kept records after explicit visual selection; closing cancels extraction."""
    if not records:
        return []
    try:
        import tkinter as tk
        from tkinter import messagebox
        from PIL import ImageTk
    except ImportError:
        raise ValueError('Can Tkinter va Pillow de xem anh. Cai: python -m pip install pillow') from None
    try:
        window = tk.Tk()
    except tk.TclError:
        raise ValueError('Khong mo duoc cua so. Hay chay tren desktop Windows co giao dien. '
                         'Chi dung --labeled-mode all neu muon lay TOAN BO anh co nhan.') from None
    window.title('Chon anh lua + nguon sang khac - giu nguyen label')
    width = max(500, min(1100, window.winfo_screenwidth()-60))
    height = max(400, min(820, window.winfo_screenheight()-90))
    window.geometry(f'{width}x{height}')
    state = {'index': 0, 'done': False, 'error': None}
    decisions = [None] * len(records)
    tk.Label(window, text='GIU neu anh co lua va nguon sang khac; BO QUA neu khong dung.\n'
             'Khung do = annotation goc. Label duoc copy nguyen ven.', font=('Arial', 11)).pack(pady=8)
    status = tk.Label(window, wraplength=width-30)
    status.pack()
    controls = tk.Frame(window)
    controls.pack(side='bottom', pady=12)
    panel = tk.Label(window)
    panel.pack(expand=True)

    def display():
        index = state['index']
        if index == len(records):
            if messagebox.askyesno('Hoan tat', f'Giu {sum(x is True for x in decisions)} / '
                                   f'{len(records)} anh co nhan. Bat dau sao chep?', parent=window):
                state['done'] = True
                window.destroy()
                return
            state['index'] = len(records)-1
            index = state['index']
        record = records[index]
        status.config(text=f"{index+1}/{len(records)} | {record['image']}\n"
                      f"Da chon giu: {sum(x is True for x in decisions)}")
        try:
            preview = make_preview(record, names, (width-30, height-200))
            photo = ImageTk.PhotoImage(preview, master=window)
            panel.config(image=photo)
            panel.image = photo
        except Exception as error:
            state['error'] = f"Khong doc duoc anh {record['image']}: {error}"
            window.destroy()

    def choose(keep):
        decisions[state['index']] = keep
        state['index'] += 1
        display()

    def back():
        if state['index'] > 0:
            state['index'] -= 1
            display()

    def cancel():
        if messagebox.askyesno('Huy', 'Huy lan trich xuat nay? Chua tao ket qua.', parent=window):
            window.destroy()

    tk.Button(controls, text='Quay lai', command=back).pack(side='left', padx=6)
    tk.Button(controls, text='GIU + label (G)', command=lambda: choose(True),
              bg='#d8f0d0').pack(side='left', padx=6)
    tk.Button(controls, text='BO QUA (B)', command=lambda: choose(False)).pack(side='left', padx=6)
    tk.Button(controls, text='Huy', command=cancel).pack(side='left', padx=6)
    window.bind('g', lambda event: choose(True))
    window.bind('b', lambda event: choose(False))
    window.bind('<Left>', lambda event: back())
    window.protocol('WM_DELETE_WINDOW', cancel)
    display()
    if not state['error']:
        window.mainloop()
    if state['error']:
        raise ValueError(state['error'])
    if not state['done']:
        raise ValueError('Da huy buoc chon anh. Khong tao folder ket qua.')
    return [record for record, keep in zip(records, decisions) if keep]


def extract(root, yaml_path, output, labeled_mode='review', reviewer=None):
    root, yaml_path, output = map(clean_path, (root, yaml_path, output))
    if not root.is_dir():
        raise ValueError(f'Khong tim thay dataset: {root}')
    if root.name.lower() == 'images':
        root = root.parent
        print(f'Da nhan folder images; dung dataset cha: {root}')
    if output == root or root in output.parents:
        raise ValueError('Folder ket qua phai nam NGOAI dataset goc.')
    if output.exists():
        raise ValueError(f'Khong ghi de folder da ton tai: {output}')
    if labeled_mode not in ('review', 'all'):
        raise ValueError('labeled_mode phai la review hoac all.')
    fire_id, names = read_fire_id(yaml_path)
    layout = get_layout(root)
    print(f'Fire ID: {fire_id} | Dataset: {root}', flush=True)
    automatic, labeled, summaries = [], [], {}
    for split, images, labels in layout:
        print(f"Dang quet {split or 'dataset'}...", flush=True)
        counts = Counter()
        summaries[split] = counts
        for image in sorted(images.rglob('*')):
            if not image.is_file() or image.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            relative = image.relative_to(images)
            label = labels / relative.with_suffix('.txt')
            ids = label_ids(label, names) if label.exists() else set()
            counts['total'] += 1
            reason = 'labeled' if ids else ('empty' if label.exists() else 'missing')
            counts[reason] += 1
            record = {'image': image, 'label': label if label.exists() else None,
                      'split': split, 'relative': relative, 'reason': reason}
            (labeled if ids else automatic).append(record)
    if not automatic and not labeled:
        raise ValueError('Khong tim thay anh duoc ho tro.')
    print(f'Tu dong lay {len(automatic)} anh khong nhan; {len(labeled)} anh co annotation.', flush=True)
    if labeled_mode == 'all':
        print('Che do ALL: lay TOAN BO anh co nhan, ke ca anh chi co lua.', flush=True)
        kept_labeled = labeled
    else:
        kept_labeled = (reviewer or review_labeled)(labeled, names)
    selected = automatic + kept_labeled
    for record in kept_labeled:
        summaries[record['split']]['kept_labeled'] += 1
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.fire_background_', dir=output.parent))
    try:
        for split, _, _ in layout:
            for folder in ('images', 'labels'):
                (staging / split / folder).mkdir(parents=True)
        for number, record in enumerate(selected, 1):
            destination = staging / record['split'] / 'images' / record['relative']
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(record['image'], destination)
            if record['label'] is not None:
                label_out = staging / record['split'] / 'labels' / record['relative'].with_suffix('.txt')
                label_out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(record['label'], label_out)
            if number % 1000 == 0:
                print(f'Da copy {number}/{len(selected)}...', flush=True)
        output.mkdir()
        try:
            for child in staging.iterdir():
                shutil.move(str(child), str(output / child.name))
        except BaseException:
            shutil.rmtree(output)
            raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    for split, counts in summaries.items():
        print(f"{split or 'dataset'}: thieu label={counts['missing']}, rong={counts['empty']}, "
              f"giu co label={counts['kept_labeled']}/{counts['labeled']}")
    print(f'HOAN TAT: {output}')
    print('Label duoc giu nguyen byte, class ID va toa do. Dataset goc khong thay doi.')
    print('Anh thieu label can xem lai; anh co lua duoc giu la POSITIVE, khong phai negative rong.')
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', help='Duong dan dataset; neu thieu se hoi nguoi dung.')
    parser.add_argument('--yaml', dest='yaml_path', help='Duong dan YAML; neu thieu se hoi nguoi dung.')
    parser.add_argument('--output', help='Folder moi; mac dinh <dataset>_background_and_mixed.')
    parser.add_argument('--labeled-mode', choices=('review', 'all'), default='review',
                        help='review: chon anh bang mat; all: lay TOAN BO anh co label.')
    args = parser.parse_args()
    interactive = args.dataset is None or args.yaml_path is None
    try:
        dataset_text = args.dataset if args.dataset is not None else input('Nhap duong dan DATASET cu: ').strip()
        yaml_text = args.yaml_path if args.yaml_path is not None else input('Nhap duong dan file YAML: ').strip()
        if not dataset_text.strip() or not yaml_text.strip():
            raise ValueError('Phai nhap ca duong dan dataset va YAML.')
        root = clean_path(dataset_text)
        if root.name.lower() == 'images':
            root = root.parent
        default_output = root.with_name(root.name + '_background_and_mixed')
        output_text = args.output
        if output_text is None and interactive:
            output_text = input(f'Folder ket qua (Enter = {default_output}): ').strip()
        output = clean_path(output_text) if output_text else default_output
        extract(root, clean_path(yaml_text), output, args.labeled_mode)
    except (OSError, ValueError, EOFError) as error:
        print(f'LOI: {error}', file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print('\nDa dung. Dataset goc khong thay doi.', file=sys.stderr)
        return 130
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
