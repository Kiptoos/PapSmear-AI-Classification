"""
Run from C:\\developer\\PapSmear_AI_v2 with venv activated:
    python sort_dataset.py

Copies only the cropped cell .bmp files (skips .dat annotation files)
from raw/sipakmed/im_<Class>/im_<Class>/CROPPED/ into dataset/<Class>/,
preserving original filenames (e.g. 001_04.bmp) so the source-slide ID
prefix stays intact for the later slide-disjoint CV split.
"""
import os
import shutil

RAW_ROOT = os.path.join('raw', 'sipakmed')
DATASET_ROOT = 'dataset'

# Maps the raw download's folder naming (im_<Class>) to our clean class names
CLASS_FOLDER_MAP = {
    'im_Dyskeratotic': 'Dyskeratotic',
    'im_Koilocytotic': 'Koilocytotic',
    'im_Metaplastic': 'Metaplastic',
    'im_Parabasal': 'Parabasal',
    'im_Superficial-Intermediate': 'Superficial-Intermediate',
}

def sort_dataset():
    total_copied = 0
    summary = {}

    for raw_class_folder, clean_class_name in CLASS_FOLDER_MAP.items():
        # raw/sipakmed/im_Dyskeratotic/im_Dyskeratotic/CROPPED/
        cropped_path = os.path.join(RAW_ROOT, raw_class_folder, raw_class_folder, 'CROPPED')

        if not os.path.exists(cropped_path):
            print(f"WARNING: expected path not found: {cropped_path}")
            print("  -> Check the exact folder name/casing under raw/sipakmed/ and update CLASS_FOLDER_MAP if needed.")
            continue

        dest_path = os.path.join(DATASET_ROOT, clean_class_name)
        os.makedirs(dest_path, exist_ok=True)

        bmp_files = [f for f in os.listdir(cropped_path) if f.lower().endswith('.bmp')]

        for fname in bmp_files:
            src = os.path.join(cropped_path, fname)
            dst = os.path.join(dest_path, fname)
            shutil.copy2(src, dst)

        summary[clean_class_name] = len(bmp_files)
        total_copied += len(bmp_files)
        print(f"{clean_class_name}: {len(bmp_files)} images copied")

    print(f"\nTotal images copied: {total_copied}")
    print("\nExpected SIPaKMeD totals for comparison:")
    print("  Dyskeratotic: 813, Koilocytotic: 825, Metaplastic: 793,")
    print("  Parabasal: 787, Superficial-Intermediate: 831 (4,049 total)")

    mismatches = []
    expected = {
        'Dyskeratotic': 813, 'Koilocytotic': 825, 'Metaplastic': 793,
        'Parabasal': 787, 'Superficial-Intermediate': 831,
    }
    for cls, count in summary.items():
        if count != expected.get(cls):
            mismatches.append(f"  {cls}: got {count}, expected {expected.get(cls)}")

    if mismatches:
        print("\n⚠️  MISMATCHES FOUND — do not proceed to training until resolved:")
        for m in mismatches:
            print(m)
    else:
        print("\n✅ All class counts match expected SIPaKMeD totals.")

if __name__ == '__main__':
    sort_dataset()
