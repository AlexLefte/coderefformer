"""
populate_samples.py
Populează folderele static/images/sample_X/ pentru demo-ul CodeRefFormer.

Utilizare:
    python populate_samples.py \
        --lq   /cale/catre/lq \
        --hq   /cale/catre/hq \
        --ref  /cale/catre/ref \
        --method coderefformer_v1:/cale/catre/predictii_v1 \
        --method coderefformer_v2:/cale/catre/predictii_v2 \
        --method coderefformer_v3:/cale/catre/predictii_v3 \
        --method instantrestore:/cale/catre/instantrestore \
        --method refldm:/cale/catre/refldm \
        --images img001.png img002.jpg img003.png \
        --output static/images

    Sau cu lista dintr-un fișier text (un nume de imagine per linie):
        --images-file selected.txt

Dependențe: Pillow (pip install Pillow)
"""

import argparse
import sys
from pathlib import Path

try:
    from PIL import Image
    HAS_PILLOW = True
except ImportError:
    HAS_PILLOW = False



def find_image(folder: Path, stem: str) -> Path | None:
    """Caută recursiv orice fișier cu stem-ul dat în folder și subfolderele sale."""
    matches = [p for p in folder.rglob(f"{stem}*") if p.is_file() and p.stem == stem]
    return matches[0] if matches else None


def save_as_png(src: Path, dst: Path):
    """Copiază/convertește imaginea la dst cu extensia .png."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() == '.png':
        import shutil
        shutil.copy2(src, dst)
    else:
        if not HAS_PILLOW:
            raise RuntimeError(
                f"Pillow necesar pentru conversia {src.suffix} → PNG. "
                "Instalează cu: pip install Pillow"
            )
        img = Image.open(src).convert('RGB')
        img.save(dst, 'PNG')


def parse_args():
    p = argparse.ArgumentParser(
        description='Populează static/images/sample_X/ pentru demo CodeRefFormer.'
    )
    p.add_argument('--lq',  required=True, help='Folder cu imaginile LQ (degradate)')
    p.add_argument('--hq',  required=True, help='Folder cu imaginile HQ (originale)')
    p.add_argument('--ref', required=True, help='Folder cu imaginile de referință')
    p.add_argument(
        '--method', metavar='NUME:CALE', action='append', default=[],
        help='Metodă și folderul ei de predicții (ex: coderefformer_v1:/cale). '
             'Poate fi specificat de mai multe ori.'
    )
    p.add_argument(
        '--images', nargs='+', metavar='IMAGINE',
        help='Lista de imagini de selectat (nume de fișier, cu sau fără extensie)'
    )
    p.add_argument(
        '--images-file', metavar='FIȘIER',
        help='Fișier text: "imagine ref" per linie (ref opțional; dacă lipsește, '
             'se folosește același stem ca imaginea)'
    )
    p.add_argument(
        '--output', default='static/images',
        help='Folder de ieșire (implicit: static/images)'
    )
    p.add_argument(
        '--start-index', type=int, default=1,
        help='Index de start pentru sample_X (implicit: 1)'
    )
    p.add_argument(
        '--clean', action='store_true',
        help='Șterge folderele sample_X existente înainte de a le popula'
    )
    p.add_argument(
        '--dry-run', action='store_true',
        help='Afișează ce ar face fără să copieze sau să șteargă fișiere'
    )
    return p.parse_args()


def main():
    args = parse_args()

    # ── Colectează lista de imagini: lista de (img_stem, ref_stem) ───────────
    # --images acceptă doar imagini (ref = același stem)
    pairs = [(Path(n).stem, None) for n in (args.images or [])]

    if args.images_file:
        txt = Path(args.images_file).read_text(encoding='utf-8')
        for ln in txt.splitlines():
            ln = ln.strip()
            if not ln or ln.startswith('#'):
                continue
            parts = ln.split()
            img_stem = Path(parts[0]).stem
            ref_stem = Path(parts[1]).stem if len(parts) >= 2 else None
            pairs.append((img_stem, ref_stem))

    if not pairs:
        print("Eroare: nicio imagine specificată (--images sau --images-file).", file=sys.stderr)
        sys.exit(1)

    # ── Parsează metodele ────────────────────────────────────────────────────
    methods = {}
    for entry in args.method:
        if ':' not in entry:
            print(f"Eroare: format invalid pentru --method '{entry}'. Folosește NUME:CALE.", file=sys.stderr)
            sys.exit(1)
        name, path = entry.split(':', 1)
        methods[name.strip()] = Path(path.strip())

    sources = {
        'lq':  Path(args.lq),
        'hq':  Path(args.hq),
        'ref': Path(args.ref),
        **methods,
    }

    # ── Validează folderele sursă ────────────────────────────────────────────
    for label, folder in sources.items():
        if not folder.is_dir():
            print(f"Eroare: folderul pentru '{label}' nu există: {folder}", file=sys.stderr)
            sys.exit(1)

    output_base = Path(args.output)
    errors = 0

    print(f"{'[DRY RUN] ' if args.dry_run else ''}Procesez {len(pairs)} imagini → {output_base}\n")

    for idx, (img_stem, ref_stem) in enumerate(pairs, start=args.start_index):
        sample_dir = output_base / f"sample_{idx}"

        if args.clean and sample_dir.exists():
            print(f"  🗑  Șterg {sample_dir.name}/")
            if not args.dry_run:
                import shutil as _shutil
                _shutil.rmtree(sample_dir)
        # ref_stem fallback la același stem ca imaginea dacă nu e specificat
        effective_ref = ref_stem if ref_stem else img_stem

        print(f"  sample_{idx}/ ← {img_stem}  (ref: {effective_ref})")

        for label, folder in sources.items():
            stem = effective_ref if label == 'ref' else img_stem
            src = find_image(folder, stem)
            dst = sample_dir / f"{label}.png"

            if src is None:
                print(f"    ⚠  [{label}] nu a fost găsită imaginea '{stem}' în {folder}")
                errors += 1
                continue

            print(f"    ✓  {label}.png  ←  {src.name}")
            if not args.dry_run:
                try:
                    save_as_png(src, dst)
                except Exception as e:
                    print(f"    ✗  Eroare la copierea {src}: {e}")
                    errors += 1

        print()

    # ── Sumar ────────────────────────────────────────────────────────────────
    total = len(pairs) * len(sources)
    ok    = total - errors
    print(f"{'[DRY RUN] ' if args.dry_run else ''}Gata: {ok}/{total} fișiere {'ar fi procesate' if args.dry_run else 'procesate'}.", end='')
    if errors:
        print(f" ({errors} erori)")
    else:
        print(" Totul OK.")

    if errors:
        sys.exit(1)


if __name__ == '__main__':
    main()
