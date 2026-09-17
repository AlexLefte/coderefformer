"""
populate_video_samples.py
Populează folderele static/videos/compare/sample_X/ pentru demo-ul CodeRefFormer.

Structura sursă:
  LQ / GT (cu subdirectorul 'videos/'):
    lq_full_mid/{subject_id}/videos/{video_id}/frame_001.png ...
    gt_full/{subject_id}/videos/{video_id}/frame_001.png ...

  Metode (fără subdirectorul 'videos/'):
    method_folder/{subject_id}/{video_id}/frame_001.png ...

  Ref (opțional, imagine statică per subiect):
    ref_folder/{ref_subject_id}.*   sau   ref_folder/{ref_subject_id}/...

Fișier videos.txt — câte o linie:
    subject_id  video_id  [ref_subject_id]

Exemplu:
    0001  clip_001  0042
    0001  clip_003
    0002  clip_001  0010

Dependențe: ffmpeg în PATH
"""

import argparse
import subprocess
import sys
import shutil
from pathlib import Path


# ─── UTILITARE ───────────────────────────────────────────────────────────────

def find_frames_dir(base: Path, subject_id: str, video_id: str, has_videos_subdir: bool) -> Path | None:
    """Găsește folderul cu frame-urile unui clip."""
    if has_videos_subdir:
        p = base / subject_id / 'videos' / video_id
    else:
        p = base / subject_id / video_id
    return p if p.is_dir() else None


def find_ref_image(gt_base: Path, subject_id: str, ref_name: str | None) -> Path | None:
    """Găsește imaginea de referință din gt_base/{subject_id}/images/.
    Dacă ref_name e specificat, caută exact acel fișier (stem, orice extensie).
    Altfel ia prima imagine sortată din folder."""
    images_dir = gt_base / subject_id / 'images'
    if not images_dir.is_dir():
        return None
    if ref_name:
        stem = Path(ref_name).stem
        matches = [p for p in images_dir.iterdir() if p.is_file() and p.stem == stem]
        return matches[0] if matches else None
    candidates = sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp'}
    )
    return candidates[0] if candidates else None


def frames_to_mp4(frames_dir: Path, output: Path, fps: int, dry_run: bool) -> bool:
    """Convertește frame-urile PNG/JPG dintr-un folder la MP4 via ffmpeg."""
    frames = sorted(frames_dir.glob('*.png')) or sorted(frames_dir.glob('*.jpg'))
    if not frames:
        print(f"    ⚠  Nu s-au găsit frame-uri în {frames_dir}")
        return False

    print(f"    ✓  {output.name}  ←  {frames_dir.parent.name}/{frames_dir.name}  ({len(frames)} frame-uri)")

    if dry_run:
        return True

    output.parent.mkdir(parents=True, exist_ok=True)

    # Scriem un fișier concat temporar cu calea absolută a fiecărui frame
    concat_path = output.parent / f"_tmp_concat_{output.stem}.txt"
    concat_path.write_text(
        '\n'.join(f"file '{f.resolve().as_posix()}'" for f in frames),
        encoding='utf-8'
    )

    cmd = [
        'ffmpeg', '-y',
        '-r', str(fps),
        '-f', 'concat', '-safe', '0',
        '-i', str(concat_path),
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-crf', '18',
        '-r', str(fps),
        str(output)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    concat_path.unlink(missing_ok=True)

    if result.returncode != 0:
        print(f"    ✗  ffmpeg error:\n{result.stderr[-500:]}")
        return False
    return True


# ─── ARGPARSE ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Populează static/videos/compare/sample_X/ pentru demo CodeRefFormer.'
    )

    # Surse LQ / GT — au subdirectorul videos/ intercalat
    p.add_argument('--lq',  required=True,
                   help='Folder rădăcină LQ  (ex: lq_full_mid/)')
    p.add_argument('--gt',  required=True,
                   help='Folder rădăcină GT/HQ  (ex: gt_full/)')

    # Metodele — fără subdirectorul videos/
    p.add_argument(
        '--method', metavar='NUME:CALE', action='append', default=[],
        help='Metodă și folderul ei (ex: coderefformer_va:/cale). Poate fi repetat.'
    )


    # Lista de clipuri
    p.add_argument(
        '--videos-file', metavar='FIȘIER',
        help='Fișier text: "subject_id video_id [ref_subject_id]" per linie'
    )
    p.add_argument(
        '--videos', nargs='+', metavar='SUBJECT_ID/VIDEO_ID',
        help='Clipuri direct ca argumente (format: subject_id/video_id)'
    )

    p.add_argument('--fps', type=int, default=25,
                   help='Frame rate pentru MP4 (implicit: 25)')
    p.add_argument('--output', default='static/videos/compare',
                   help='Folder de ieșire (implicit: static/videos/compare)')
    p.add_argument('--start-index', type=int, default=1,
                   help='Index de start pentru sample_X (implicit: 1)')
    p.add_argument('--clean', action='store_true',
                   help='Șterge folderele sample_X existente înainte de a le popula')
    p.add_argument('--dry-run', action='store_true',
                   help='Afișează ce ar face fără să proceseze sau să șteargă fișiere')
    return p.parse_args()


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Colectează lista de clipuri: [(subject_id, video_id, ref_subject_id|None)] ──
    triples = []

    for entry in (args.videos or []):
        parts = entry.split('/')
        if len(parts) != 2:
            print(f"Eroare: format invalid '{entry}'. Folosește subject_id/video_id.", file=sys.stderr)
            sys.exit(1)
        triples.append((parts[0], parts[1], None))

    if args.videos_file:
        txt = Path(args.videos_file).read_text(encoding='utf-8')
        for ln in txt.splitlines():
            ln = ln.strip()
            if not ln or ln.startswith('#'):
                continue
            parts = ln.split()
            if len(parts) < 2:
                print(f"Linie ignorată (format invalid): '{ln}'", file=sys.stderr)
                continue
            subject_id = parts[0]
            video_id   = parts[1]
            ref_id     = parts[2] if len(parts) >= 3 else None
            triples.append((subject_id, video_id, ref_id))

    if not triples:
        print("Eroare: niciun clip specificat (--videos sau --videos-file).", file=sys.stderr)
        sys.exit(1)

    # ── Parsează metodele ────────────────────────────────────────────────────
    methods = {}
    for entry in args.method:
        if ':' not in entry:
            print(f"Eroare: format invalid --method '{entry}'. Folosește NUME:CALE.", file=sys.stderr)
            sys.exit(1)
        name, path = entry.split(':', 1)
        methods[name.strip()] = Path(path.strip())

    lq_base = Path(args.lq)
    gt_base = Path(args.gt)

    # ── Validare ─────────────────────────────────────────────────────────────
    for label, folder in [('lq', lq_base), ('gt', gt_base), *methods.items()]:
        if not folder.is_dir():
            print(f"Eroare: folderul pentru '{label}' nu există: {folder}", file=sys.stderr)
            sys.exit(1)

    # ── Verifică ffmpeg ───────────────────────────────────────────────────────
    if not args.dry_run:
        if subprocess.run(['ffmpeg', '-version'], capture_output=True).returncode != 0:
            print("Eroare: ffmpeg nu a fost găsit în PATH. Instalează ffmpeg și adaugă-l în PATH.", file=sys.stderr)
            sys.exit(1)

    output_base = Path(args.output)
    errors = 0
    tag = '[DRY RUN] ' if args.dry_run else ''

    print(f"{tag}Procesez {len(triples)} clipuri → {output_base}\n")

    for idx, (subject_id, video_id, ref_id) in enumerate(triples, start=args.start_index):
        sample_dir = output_base / f"sample_{idx}"
        print(f"  sample_{idx}/ ← {subject_id}/{video_id}  (ref: {ref_id or 'prima imagine'})")

        if args.clean and sample_dir.exists():
            print(f"    🗑  Șterg {sample_dir}/")
            if not args.dry_run:
                shutil.rmtree(sample_dir)

        # LQ
        lq_dir = find_frames_dir(lq_base, subject_id, video_id, has_videos_subdir=True)
        if lq_dir:
            if not frames_to_mp4(lq_dir, sample_dir / 'lq.mp4', args.fps, args.dry_run):
                errors += 1
        else:
            print(f"    ⚠  [lq] folder negăsit: {lq_base}/{subject_id}/videos/{video_id}/")
            errors += 1

        # GT / HQ
        gt_dir = find_frames_dir(gt_base, subject_id, video_id, has_videos_subdir=True)
        if gt_dir:
            if not frames_to_mp4(gt_dir, sample_dir / 'hq.mp4', args.fps, args.dry_run):
                errors += 1
        else:
            print(f"    ⚠  [gt] folder negăsit: {gt_base}/{subject_id}/videos/{video_id}/")
            errors += 1

        # Metode
        for name, base in methods.items():
            m_dir = find_frames_dir(base, subject_id, video_id, has_videos_subdir=False)
            if m_dir:
                if not frames_to_mp4(m_dir, sample_dir / f"{name}.mp4", args.fps, args.dry_run):
                    errors += 1
            else:
                print(f"    ⚠  [{name}] folder negăsit: {base}/{subject_id}/{video_id}/")
                errors += 1

        # Ref — imaginea specificată în txt sau prima din gt_base/{subject_id}/images/
        ref_src = find_ref_image(gt_base, subject_id, ref_id)
        if ref_src:
            ref_dst = sample_dir / 'ref.jpg'
            print(f"    ✓  ref.jpg  ←  {ref_src.name}")
            if not args.dry_run:
                sample_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ref_src, ref_dst)
        else:
            print(f"    ⚠  [ref] nicio imagine în {gt_base}/{subject_id}/images/ (ref_name={ref_id})")
            errors += 1

        print()

    # ── Sumar ─────────────────────────────────────────────────────────────────
    n_sources = 2 + len(methods) + 1  # lq + gt + metode + ref
    total = len(triples) * n_sources
    ok = total - errors
    print(f"{tag}Gata: {ok}/{total} fișiere {'ar fi procesate' if args.dry_run else 'procesate'}.", end='')
    print(f" ({errors} erori)" if errors else " Totul OK.")

    if errors:
        sys.exit(1)


if __name__ == '__main__':
    main()
