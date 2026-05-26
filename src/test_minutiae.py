from pathlib import Path
import cv2
import matplotlib.pyplot as plt

from minutiae import extract_minutiae

IMG_PATH = Path(
    "data/processed/Real/298__M_Right_thumb_finger.BMP"
)

img = cv2.imread(
    str(IMG_PATH),
    0
)

result = extract_minutiae(
    img
)

plt.figure(figsize=(18,6))

plt.subplot(141)
plt.imshow(
    img,
    cmap="gray"
)
plt.title("Original")

plt.subplot(142)
plt.imshow(
    result["gabor"],
    cmap="gray"
)
plt.title("Gabor")

plt.subplot(143)
plt.imshow(
    result["skeleton"],
    cmap="gray"
)
plt.title("Skeleton")

plt.subplot(144)
plt.imshow(
    cv2.cvtColor(
        result["overlay"],
        cv2.COLOR_BGR2RGB
    )
)
plt.title("Minutiae")

plt.tight_layout()
plt.show()


result = extract_minutiae(
    img
)

print(
    "Endings:",
    len(
        result["endings"]
    )
)

print(
    "Bifurcations:",
    len(
        result["bifurcations"]
    )
)