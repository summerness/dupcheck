"""Generate synthetic dataset for duplicate detection experiments.

Creates two folders under project `data/synth_db` and `data/synth_new` and a
labels CSV `data/synth_labels.csv` listing ground-truth matches.

Usage:
  python tools/generate_synthetic.py --out_dir ./data --count 5

This reproduces the same patterns used in the interactive session.
"""
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageEnhance
import random
import csv


def generate(out_dir: Path, count: int = 5):
    db = out_dir / 'synth_db'
    new = out_dir / 'synth_new'
    db.mkdir(parents=True, exist_ok=True)
    new.mkdir(parents=True, exist_ok=True)

    labels = []
    for i in range(1, count+1):
        img = Image.new('RGB',(400,300),(200+i*5,180+i*3,160+i*2))
        draw = ImageDraw.Draw(img)
        for x in range(50,350,6):
            for y in range(60,240,6):
                if (x*y+i) % 13 < 4:
                    draw.point((x,y),(0,0,0))
        base = db / f'base_{i}.jpg'
        img.save(base)

        # exact copy
        img.save(new / f'new_{i}_copy.jpg')
        labels.append((f'new_{i}_copy.jpg', base.name))

        # cropped
        crop = img.crop((80,70,320,230))
        crop.save(new / f'new_{i}_crop.jpg')
        labels.append((f'new_{i}_crop.jpg', base.name))

        # rotated
        rot = img.rotate(15, expand=True, fillcolor=(200,200,200))
        rot.save(new / f'new_{i}_rot.jpg')
        labels.append((f'new_{i}_rot.jpg', base.name))

        # brightness
        bright = ImageEnhance.Brightness(img).enhance(1.3)
        bright.save(new / f'new_{i}_bright.jpg')
        labels.append((f'new_{i}_bright.jpg', base.name))

        # compressed
        img.save(new / f'new_{i}_jpeg30.jpg', quality=30)
        labels.append((f'new_{i}_jpeg30.jpg', base.name))

        # ps overlay (draw rectangle)
        ps = img.copy()
        d = ImageDraw.Draw(ps)
        d.rectangle((120,90,220,160), fill=(255,255,255))
        ps.save(new / f'new_{i}_ps.jpg')
        labels.append((f'new_{i}_ps.jpg', base.name))

        # flipped
        flip = img.transpose(Image.FLIP_LEFT_RIGHT)
        flip.save(new / f'new_{i}_flip.jpg')
        labels.append((f'new_{i}_flip.jpg', base.name))

    # add some unique images
    for j in range(1, count+1):
        u = Image.new('RGB',(300,200),(random.randint(0,255),random.randint(0,255),random.randint(0,255)))
        u.save(new / f'new_unique_{j}.jpg')
        labels.append((f'new_unique_{j}.jpg',''))

    # write labels.csv
    labp = out_dir / 'synth_labels.csv'
    with open(labp, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['new_image','matched_image','label'])
        for newn, dbn in labels:
            lab = 'unique' if dbn=='' else 'partial_duplicate'
            w.writerow([newn, dbn, lab])

    print('Synthetic dataset created:')
    print(' DB:', db)
    print(' NEW:', new)
    print(' Labels:', labp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', default='./data')
    parser.add_argument('--count', type=int, default=5)
    args = parser.parse_args()
    generate(Path(args.out_dir), count=args.count)
