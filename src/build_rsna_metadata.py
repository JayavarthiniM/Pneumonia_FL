import json
import os
import glob
import pandas as pd


MAPPING_FILE = r"data\rsna\mapping\pneumonia-challenge-dataset-mappings_2018.json"
IMAGE_ROOT = r"data\rsna\images"
OUTPUT_FILE = r"data\rsna\mapping\rsna_metadata.csv"


def main():

    print("Loading RSNA mapping...")

    with open(MAPPING_FILE, "r", encoding="utf-8") as f:
        mappings = json.load(f)

    print(f"Mapping entries: {len(mappings)}")

    print("Searching for DICOM files...")

    dicom_files = glob.glob(
        os.path.join(IMAGE_ROOT, "**", "*.dcm"),
        recursive=True
    )

    print(f"DICOM files found: {len(dicom_files)}")

    # Map SOPInstanceUID -> actual DICOM file path
    dicom_by_sop = {}

    for path in dicom_files:
        filename = os.path.basename(path)

        if filename.lower().endswith(".dcm"):
            sop_uid = filename[:-4]
            dicom_by_sop[sop_uid] = os.path.abspath(path)

    records = []

    for item in mappings:

        label = item["subset_init_label"]

        # Exclude unknown
        if label == 0:
            continue

        # RSNA binary target
        if label == 1:
            pneumonia = 1
        elif label == 2:
            pneumonia = 0
        else:
            continue

        sop_uid = item["SOPInstanceUID"]

        dicom_path = dicom_by_sop.get(sop_uid)

        if dicom_path is None:
            print(f"WARNING: DICOM not found: {sop_uid}")
            continue

        records.append({
            "img_id": item["img_id"],
            "subset_img_id": item["subset_img_id"],
            "StudyInstanceUID": item["StudyInstanceUID"],
            "SeriesInstanceUID": item["SeriesInstanceUID"],
            "SOPInstanceUID": sop_uid,
            "subset_group": item["subset_group"],
            "subset_init_label": label,
            "pneumonia": pneumonia,
            "orig_labels": "|".join(item.get("orig_labels", [])),
            "dicom_path": dicom_path
        })

    df = pd.DataFrame(records)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    df.to_csv(OUTPUT_FILE, index=False)

    print("\n===== RSNA METADATA CREATED =====")
    print(f"Total usable images: {len(df)}")
    print(f"Pneumonia: {(df['pneumonia'] == 1).sum()}")
    print(f"No pneumonia: {(df['pneumonia'] == 0).sum()}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()