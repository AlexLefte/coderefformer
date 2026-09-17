#!/usr/bin/env python3
r"""
frames_to_mp4.py
----------------
Convertește foldere de cadre (PNG/JPG) în fișiere .mp4 gata pentru site.

Dai direct calea la folderul cu cadre pentru fiecare model — fără nivel de clip_id.

Utilizare:
    python frames_to_mp4.py \
        --ref   D:\data\ref.jpg \
        --models hq:D:\data\gt\clip_folder \
                 lq:D:\data\lq\clip_folder \
                 codeformer:D:\data\cf\clip_folder \
                 coderefformer:D:\data\crf\clip_folder \
                 coderefformer_v:D:\data\crfv\clip_folder \
        --out   static/videos/compare/sample_1 \
        --fps   25

Structură generată:
    static/videos/compare/sample_1/
        ref.jpg
        hq.mp4
        lq.mp4
        codeformer.mp4
        coderefformer.mp4
        coderefformer_v.mp4
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp'}


# ── helpers ───────────────────────────────────────────────────────────────────

def parse_models(raw: list[str]) -> dict[str, Path]:
    """Parsează 'nume:/cale' — gestionează și căi Windows cu drive letter (D:\\...)."""
    result = {}
    for item in raw:
        # Găsim primul ":" care nu e drive letter Windows
        # Drive letter = exact 1 literă urmată de ":"
        # Dacă item e "D:\cale" => nu are nume model => eroare
        # Dacă item e "hq:D:\cale" => split după primul ":"
        colon_idx = item.find(":")
        if colon_idx == -1:
            sys.exit(f"[ERR] Format greșit: '{item}'. Folosește 'nume:/cale'.")

        name = item[:colon_idx]
        path_str = item[colon_idx + 1:]

        # Dacă name e o singură literă, probabil e drive letter fără nume model
        if len(name) == 1 and name.isalpha():
            sys.exit(f"[ERR] Lipsă nume model în '{item}'. Folosește 'hq:D:\\cale'.")

        result[name.strip()] = Path(path_str.strip())
    return result


def find_frames(folder: Path) -> list[Path]:
    frames = sorted(
        [p for p in folder.iterdir() if p.suffix.lower() in {'.png', '.jpg', '.jpeg'}],
        key=lambda p: p.name
    )
    return frames


def encode(frames_dir: Path, out_path: Path, fps: int, crf: int, scale: int | None, min_duration: float = 5.0) -> bool:
    import tempfile, os
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frames = find_frames(frames_dir)
    if not frames:
        print(f"  [SKIP] Nu am găsit cadre în {frames_dir}")
        return False

    # Calculează loop-uri necesare
    n_frames = len(frames)
    duration = n_frames / fps
    loop_count = max(1, int(min_duration / duration + 0.999)) if duration < min_duration else 1
    all_frames = frames * loop_count

    # Scrie fișier concat cu toate cadrele în ordine sortată, indiferent de gaps
    concat_file = out_path.parent / f"_concat_{out_path.stem}.txt"
    with open(concat_file, 'w') as f:
        for frame in all_frames:
            # ffmpeg concat necesită path-uri cu forward slash și escaped apostrofe
            p = str(frame.resolve()).replace("\\", "/")
            f.write(f"file '{p}'\n")
            f.write(f"duration {1/fps:.6f}\n")

    vf_parts = []
    if scale:
        vf_parts.append(f"scale={scale}:{scale}:flags=lanczos")
    vf_parts.append("format=yuv420p")
    vf = ",".join(vf_parts)

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_file),
        "-c:v", "libx264",
        "-crf", str(crf),
        "-preset", "slow",
        "-t", str(min_duration),
        "-vf", vf,
        "-movflags", "+faststart",
        str(out_path),
    ]

    print(f"  → {out_path.name}  [{n_frames} cadre × {loop_count} loop(uri)]")
    result = subprocess.run(cmd, capture_output=True, text=True)
    concat_file.unlink(missing_ok=True)  # curăță fișierul temporar
    if result.returncode != 0:
        print(f"  [ERR] ffmpeg:\n{result.stderr[-800:]}")
        return False
    return True


def copy_file(src: Path, out_path: Path) -> bool:
    """Copiaza fisierul; PNG/WebP → converteste la JPG pentru compatibilitate browser."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() in {'.png', '.webp', '.bmp', '.tiff'}:
        out_path = out_path.with_suffix('.jpg')
        try:
            from PIL import Image
            img = Image.open(src).convert('RGB')
            img.save(out_path, 'JPEG', quality=95)
            print(f"  -> {out_path.name}  [convertit PNG->JPG]")
        except ImportError:
            result = subprocess.run(
                ['ffmpeg', '-y', '-i', str(src), str(out_path)],
                capture_output=True
            )
            if result.returncode != 0:
                print(f"  [ERR] conversie JPG esuat")
                return False
            print(f"  -> {out_path.name}  [convertit PNG->JPG via ffmpeg]")
    else:
        shutil.copy2(src, out_path)
        print(f"  -> {out_path.name}  [copiat]")
    return True


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convertește cadre în .mp4 pentru visual_results.html"
    )
    parser.add_argument(
        "--ref", type=str, default=None,
        metavar="/cale/imagine.jpg",
        help="Cale directă către imaginea de referință (jpg/png/webp)."
    )
    parser.add_argument(
        "--models", nargs="+", required=True,
        metavar="NUME:/cale/folder_cadre",
        help="Perechi nume:cale — calea directă la folderul cu cadre. "
             "Ex: hq:D:\\data\\gt\\clip lq:D:\\data\\lq\\clip codeformer:D:\\data\\cf\\clip"
    )
    parser.add_argument(
        "--out", type=str, required=True,
        metavar="/cale/sample_N",
        help="Folderul de output. Ex: static/videos/compare/sample_1"
    )
    parser.add_argument(
        "--fps", type=int, default=25,
        help="Frame rate (default: 25)"
    )
    parser.add_argument(
        "--crf", type=int, default=18,
        help="CRF libx264 — calitate (18=foarte bun, 23=default ffmpeg). Default: 18"
    )
    parser.add_argument(
        "--scale", type=int, default=None,
        metavar="PX",
        help="Redimensionează la PX×PX (opțional, ex: 512)"
    )
    parser.add_argument(
        "--duration", type=float, default=5.0,
        help="Durata minimă a fiecărui mp4 în secunde — loopează dacă e mai scurt. Default: 5.0"
    )
    args = parser.parse_args()

    models = parse_models(args.models)
    out_root = Path(args.out)

    # Validare
    for name, path in models.items():
        if not path.exists():
            sys.exit(f"[ERR] Folder inexistent pentru modelul '{name}': {path}")

    ref_file = Path(args.ref) if args.ref else None
    if ref_file and not ref_file.is_file():
        sys.exit(f"[ERR] Fișier ref inexistent: {ref_file}")

    print(f"\n[INFO] Output → {out_root}\n")

    ok = 0
    skip = 0

    # ── Referință ──
    if ref_file:
        out_ref = out_root / f"ref{ref_file.suffix}"
        ok += copy_file(ref_file, out_ref)

    # ── Modele video ──
    for model_name, frames_dir in models.items():
        out_mp4 = out_root / f"{model_name}.mp4"
        success = encode(frames_dir, out_mp4, args.fps, args.crf, args.scale, args.duration)
        if success:
            ok += 1
        else:
            skip += 1

    print(f"\n[DONE] {ok} fișiere generate, {skip} sărite/erori.")
    print(f"Output: {out_root.resolve()}")


if __name__ == "__main__":
    main()