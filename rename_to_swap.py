"""
rename_to_swap.py
─────────────────
Place ce script dans le dossier contenant tes images.
Lance : python rename_to_swap.py

Il renomme tous les fichiers image du dossier en :
  swap_01.jpg, swap_02.jpg, swap_03.jpg ...
dans l'ordre alphabétique du nom actuel.

Options configurables ci-dessous.
"""

import os
import sys

# ── Config ────────────────────────────────────────────────────────────────────
EXTENSIONS   = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}
PREFIX       = "swap"          # préfixe → swap_01, swap_02...
START        = 1               # numéro de départ
PADDING      = 2               # chiffres : 2 → swap_01, 3 → swap_001
DRY_RUN      = False           # True = aperçu sans renommer
# ─────────────────────────────────────────────────────────────────────────────

folder = os.path.dirname(os.path.abspath(__file__))

files = sorted(
    f for f in os.listdir(folder)
    if os.path.isfile(os.path.join(folder, f))
    and os.path.splitext(f)[1].lower() in EXTENSIONS
    and f != os.path.basename(__file__)
)

if not files:
    print("❌ Aucun fichier image trouvé dans ce dossier.")
    sys.exit(0)

print(f"📂 Dossier : {folder}")
print(f"🖼️  {len(files)} fichier(s) trouvé(s)\n")

renames = []
for i, filename in enumerate(files, start=START):
    ext      = os.path.splitext(filename)[1].lower()
    new_name = f"{PREFIX}_{str(i).zfill(PADDING)}{ext}"
    renames.append((filename, new_name))

# Aperçu
for old, new in renames:
    marker = "→" if old != new else "="
    print(f"  {old:40s} {marker}  {new}")

if DRY_RUN:
    print("\n⚠️  DRY_RUN activé — aucun fichier renommé.")
    sys.exit(0)

print()
confirm = input("Confirmer le renommage ? (o/n) : ").strip().lower()
if confirm not in ("o", "oui", "y", "yes"):
    print("Annulé.")
    sys.exit(0)

# Renommage en deux passes pour éviter les collisions (ex: a→b, b→c)
tmp_renames = []
for old, new in renames:
    if old == new:
        continue
    tmp = old + ".renaming_tmp"
    os.rename(os.path.join(folder, old), os.path.join(folder, tmp))
    tmp_renames.append((tmp, new))

for tmp, new in tmp_renames:
    os.rename(os.path.join(folder, tmp), os.path.join(folder, new))

print(f"\n✅ {len(tmp_renames)} fichier(s) renommé(s).")
