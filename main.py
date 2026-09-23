import sys

import cv2


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(f"usage: {sys.argv[0]} <image>")

    image = cv2.imread(sys.argv[1])
    if image is None:
        sys.exit(f"cannot read {sys.argv[1]}")

    print(f"shape={image.shape} dtype={image.dtype}")
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
