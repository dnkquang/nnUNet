#!/usr/bin/env python3
"""Convert the CAMUS NIfTI dataset into a leakage-free nnU-Net v2 dataset.

Development cohort:
    patients 0001-0450 -> imagesTr and labelsTr
Final held-out cohort:
    patients 0451-0500 -> imagesTs
    reference masks are copied outside nnUNet_raw

Expected source layout:
    database_nifti/
      patient0001/
        patient0001_2CH_ED.nii.gz
        patient0001_2CH_ED_gt.nii.gz
        ...
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw


VIEWS = ("2CH", "4CH")
PHASES = ("ED", "ES")
TRAIN_FIRST = 1
TRAIN_LAST = 450
TEST_FIRST = 451
TEST_LAST = 500
PATIENT_PATTERN = re.compile(r"^patient(\d{4})$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert CAMUS into Dataset101_CAMUS with a patient-level 450/50 split."
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Path to the CAMUS database_nifti directory.",
    )
    parser.add_argument(
        "--heldout-labels",
        type=Path,
        required=True,
        help=(
            "Directory outside nnUNet_raw where labels for patients 0451-0500 "
            "will be stored for final evaluation."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing Dataset101_CAMUS output before conversion.",
    )
    return parser.parse_args()


def patient_number(patient_id: str) -> int:
    match = PATIENT_PATTERN.fullmatch(patient_id)
    if match is None:
        raise ValueError(
            f"Unexpected patient directory name {patient_id!r}; expected patientXXXX."
        )
    return int(match.group(1))


def expected_source_files(patient_folder: Path) -> list[tuple[str, Path, Path]]:
    patient_id = patient_folder.name
    cases: list[tuple[str, Path, Path]] = []
    for view in VIEWS:
        for phase in PHASES:
            case_id = f"{patient_id}_{view}_{phase}"
            image_path = patient_folder / f"{case_id}.nii.gz"
            label_path = patient_folder / f"{case_id}_gt.nii.gz"
            cases.append((case_id, image_path, label_path))
    return cases


def fail_if_missing(source: Path) -> list[tuple[int, Path, list[tuple[str, Path, Path]]]]:
    indexed: list[tuple[int, Path, list[tuple[str, Path, Path]]]] = []
    problems: list[str] = []

    for number in range(TRAIN_FIRST, TEST_LAST + 1):
        patient_id = f"patient{number:04d}"
        patient_folder = source / patient_id
        if not patient_folder.is_dir():
            problems.append(f"Missing patient directory: {patient_folder}")
            continue

        cases = expected_source_files(patient_folder)
        for case_id, image_path, label_path in cases:
            if not image_path.is_file():
                problems.append(f"Missing image for {case_id}: {image_path}")
            if not label_path.is_file():
                problems.append(f"Missing label for {case_id}: {label_path}")
        indexed.append((number, patient_folder, cases))

    if problems:
        preview = "\n".join(problems[:30])
        remainder = len(problems) - 30
        suffix = f"\n... and {remainder} more problem(s)." if remainder > 0 else ""
        raise FileNotFoundError(
            "CAMUS source validation failed. No files were copied.\n" + preview + suffix
        )

    return indexed


def main() -> None:
    args = parse_args()
    source = args.source.expanduser().resolve()
    heldout_labels = args.heldout_labels.expanduser().resolve()

    if not source.is_dir():
        raise NotADirectoryError(f"CAMUS source directory does not exist: {source}")
    if nnUNet_raw is None:
        raise RuntimeError(
            "nnUNet_raw is not configured. Export nnUNet_raw before running this script."
        )

    output_folder = Path(nnUNet_raw).expanduser().resolve() / "Dataset101_CAMUS"

    if output_folder.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output already exists: {output_folder}\n"
                "Use --overwrite only after confirming that this is the folder to replace."
            )
        shutil.rmtree(output_folder)

    if heldout_labels.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Held-out label directory already exists: {heldout_labels}\n"
                "Use --overwrite only after confirming that this is the folder to replace."
            )
        shutil.rmtree(heldout_labels)

    # Validate the complete source before creating output directories.
    indexed_patients = fail_if_missing(source)

    images_tr = output_folder / "imagesTr"
    labels_tr = output_folder / "labelsTr"
    images_ts = output_folder / "imagesTs"
    images_tr.mkdir(parents=True, exist_ok=False)
    labels_tr.mkdir(parents=True, exist_ok=False)
    images_ts.mkdir(parents=True, exist_ok=False)
    heldout_labels.mkdir(parents=True, exist_ok=False)

    train_cases = 0
    test_cases = 0

    for number, patient_folder, cases in indexed_patients:
        # Re-validate the folder name even though the expected path was constructed above.
        parsed_number = patient_number(patient_folder.name)
        if parsed_number != number:
            raise RuntimeError(
                f"Patient-number mismatch: expected {number}, found {patient_folder.name}."
            )

        for case_id, image_path, label_path in cases:
            nnunet_image_name = f"{case_id}_0000.nii.gz"
            nnunet_label_name = f"{case_id}.nii.gz"

            if TRAIN_FIRST <= number <= TRAIN_LAST:
                shutil.copy2(image_path, images_tr / nnunet_image_name)
                shutil.copy2(label_path, labels_tr / nnunet_label_name)
                train_cases += 1
            elif TEST_FIRST <= number <= TEST_LAST:
                shutil.copy2(image_path, images_ts / nnunet_image_name)
                shutil.copy2(label_path, heldout_labels / nnunet_label_name)
                test_cases += 1
            else:
                raise RuntimeError(f"Patient {number} is outside the configured range.")

    expected_train_cases = (TRAIN_LAST - TRAIN_FIRST + 1) * len(VIEWS) * len(PHASES)
    expected_test_cases = (TEST_LAST - TEST_FIRST + 1) * len(VIEWS) * len(PHASES)

    if train_cases != expected_train_cases:
        raise RuntimeError(
            f"Expected {expected_train_cases} training cases, copied {train_cases}."
        )
    if test_cases != expected_test_cases:
        raise RuntimeError(
            f"Expected {expected_test_cases} test cases, copied {test_cases}."
        )

    generate_dataset_json(
        str(output_folder),
        channel_names={0: "ultrasound"},
        labels={
            "background": 0,
            "LV": 1,
            "Myocardium": 2,
            "LeftAtrium": 3,
        },
        num_training_cases=train_cases,
        file_ending=".nii.gz",
        dataset_name="CAMUS",
        reference="CAMUS Challenge",
        release="1.0",
        description=(
            "CAMUS echocardiography segmentation dataset. Patients 0001-0450 are "
            "used for development; patients 0451-0500 are held out for final testing."
        ),
        license="See the CAMUS Challenge dataset terms.",
        converted_by="dnkquang",
    )

    manifest = {
        "dataset_folder": output_folder.name,
        "source": str(source),
        "development_patients": [TRAIN_FIRST, TRAIN_LAST],
        "heldout_test_patients": [TEST_FIRST, TEST_LAST],
        "training_cases": train_cases,
        "test_cases": test_cases,
        "test_labels_location": str(heldout_labels),
        "views": list(VIEWS),
        "phases": list(PHASES),
    }
    (output_folder / "partition_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Created: {output_folder}")
    print(f"Training cohort: patients {TRAIN_FIRST:04d}-{TRAIN_LAST:04d}")
    print(f"Training cases: {train_cases}")
    print(f"Held-out cohort: patients {TEST_FIRST:04d}-{TEST_LAST:04d}")
    print(f"Held-out image cases: {test_cases}")
    print(f"Held-out labels: {heldout_labels}")
    print(f"dataset.json numTraining: {train_cases}")


if __name__ == "__main__":
    main()
