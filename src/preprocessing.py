from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

SOURCE_FOLDERS = [
    "Real",
    "Altered-Easy",
    "Altered-Medium",
    "Altered-Hard"
]

RAW_DIR = Path("data/raw/SOCOFing/Real")
OUTPUT_DIR = Path("data/processed/Real")

IMG_SIZE = (224, 224)


def normalize_image(img: np.ndarray) -> np.ndarray:
    """
    Normalize pixel intensity.
    """

    img = img.astype(np.float32)

    mean = np.mean(img)
    std = np.std(img)

    img = (img - mean) / (std + 1e-8)

    img = cv2.normalize(
        img,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return img.astype(np.uint8)


def apply_clahe(img: np.ndarray) -> np.ndarray:
    """
    Contrast enhancement.
    """

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    return clahe.apply(img)


def preprocess_image(filepath: Path) -> np.ndarray:

    img = cv2.imread(
        str(filepath),
        cv2.IMREAD_GRAYSCALE
    )

    if img is None:
        raise ValueError(f"Cannot read {filepath}")

    img = normalize_image(img)

    img = apply_clahe(img)

    img = cv2.resize(
        img,
        IMG_SIZE,
        interpolation=cv2.INTER_AREA
    )

    return img


def save_processed_image(
    output_dir: Path,
    src_path: Path,
    processed_img: np.ndarray
):

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    save_path = output_dir / src_path.name

    cv2.imwrite(
        str(save_path),
        processed_img
    )


def main():

    total_success = 0
    total_failed = 0

    for folder in SOURCE_FOLDERS:

        input_dir = Path(
            f"data/raw/SOCOFing/{folder}"
        )

        output_dir = Path(
            f"data/processed/{folder}"
        )

        files = sorted(
            input_dir.glob("*.BMP")
        )

        print(
            f"\nProcessing {folder}"
        )

        success = 0
        failed = 0

        for file in tqdm(files):

            try:

                img = preprocess_image(
                    file
                )

                save_processed_image(
                    output_dir,
                    file,
                    img
                )

                success += 1

            except Exception as e:

                failed += 1

                print(
                    f"[ERROR] "
                    f"{file.name}: {e}"
                )

        print(
            f"{folder}"
        )

        print(
            f"Success : {success}"
        )

        print(
            f"Failed  : {failed}"
        )

        total_success += success
        total_failed += failed

    print("\n==========")
    print("ALL DONE")
    print("==========")

    print(
        f"Total Success : "
        f"{total_success}"
    )

    print(
        f"Total Failed  : "
        f"{total_failed}"
    )
    files = sorted(
        RAW_DIR.glob("*.BMP")
    )

    success = 0
    failed = 0

    for file in tqdm(files):

        try:

            img = preprocess_image(file)

            save_processed_image(
                output_dir,
                file,
                img
            )

            success += 1

        except Exception as e:

            failed += 1

            print(
                f"[ERROR] {file.name}: {e}"
            )

    print("\nDone")
    print(f"Success : {success}")
    print(f"Failed  : {failed}")

    files = sorted(
        RAW_DIR.glob("*.BMP")
    )

    success = 0
    failed = 0

    for file in tqdm(files):

        try:

            img = preprocess_image(file)

            save_processed_image(
                output_dir,
                file,
                img
            )

            success += 1

        except Exception as e:

            failed += 1

            print(
                f"[ERROR] {file.name}: {e}"
            )

    print("\nDone")
    print(f"Success : {success}")
    print(f"Failed  : {failed}")


if __name__ == "__main__":
    main()
