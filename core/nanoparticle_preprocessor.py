import argparse
import numpy as np
import cv2


class NanoparticlePreprocessor:
    def __init__(
        self,
        crop_size=31,
        threshold_percentile=99.5,
        min_area=2,
        max_area=200,
        blur_ksize=3,
    ):
        self.crop_size = crop_size
        self.threshold_percentile = threshold_percentile
        self.min_area = min_area
        self.max_area = max_area
        self.blur_ksize = blur_ksize

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video: {video_path}")

        crops = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            gray = self._to_gray(frame)
            centers = self._detect_centers(gray)
            crops.extend(self._extract_crops(gray, centers))

        cap.release()

        if not crops:
            raise ValueError("No crops extracted. Loosen detection thresholds.")

        crops = np.stack(crops, axis=0).astype(np.float32)
        return self._flatten_and_normalize(crops)

    def _to_gray(self, frame):
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame.astype(np.float32)

    def _detect_centers(self, gray):
        if self.blur_ksize > 1:
            g = cv2.GaussianBlur(gray, (self.blur_ksize, self.blur_ksize), 0)
        else:
            g = gray

        thresh = np.percentile(g, self.threshold_percentile)
        mask = (g >= thresh).astype(np.uint8)

        n, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

        centers = []
        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            if self.min_area <= area <= self.max_area:
                cx, cy = centroids[i]
                centers.append((int(round(cx)), int(round(cy))))
        return centers

    def _extract_crops(self, gray, centers):
        h, w = gray.shape
        half = self.crop_size // 2
        out = []
        for cx, cy in centers:
            x0, x1 = cx - half, cx + half + 1
            y0, y1 = cy - half, cy + half + 1
            if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
                continue
            out.append(gray[y0:y1, x0:x1])
        return out

    def _flatten_and_normalize(self, crops, eps=1e-8):
        flat = crops.reshape(crops.shape[0], -1)
        norms = np.linalg.norm(flat, axis=1, keepdims=True)
        return flat / (norms + eps)

    def save_sample_grid(self, crops_flat, out_path, n=64, ncols=8):
        n = min(n, crops_flat.shape[0])
        idx = np.random.default_rng(0).choice(crops_flat.shape[0], size=n, replace=False)
        imgs = crops_flat[idx].reshape(n, self.crop_size, self.crop_size)

        import matplotlib.pyplot as plt

        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols, nrows))
        for ax in axes.ravel():
            ax.axis("off")
        for k, ax in enumerate(axes.ravel()[:n]):
            ax.imshow(imgs[k], cmap="gray")
        fig.tight_layout()
        fig.savefig(out_path, dpi=120)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Detect nanoparticles in an NTA video and export centered, "
        "flattened, L2-normalized crops for the 1D homogeneous recovery experiment."
    )
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--crop-size", type=int, default=31)
    parser.add_argument("--threshold-percentile", type=float, default=99.5)
    parser.add_argument("--min-area", type=int, default=2)
    parser.add_argument("--max-area", type=int, default=200)
    parser.add_argument("--blur-ksize", type=int, default=3)
    parser.add_argument("--sample-grid", type=str, default=None)
    args = parser.parse_args()

    pre = NanoparticlePreprocessor(
        crop_size=args.crop_size,
        threshold_percentile=args.threshold_percentile,
        min_area=args.min_area,
        max_area=args.max_area,
        blur_ksize=args.blur_ksize,
    )

    crops = pre.process_video(args.video)
    np.save(args.output, crops)
    print(f"Saved {crops.shape[0]} crops of dim {crops.shape[1]} to {args.output}")

    if args.sample_grid is not None:
        pre.save_sample_grid(crops, args.sample_grid)
        print(f"Saved sample crop grid to {args.sample_grid}")


if __name__ == "__main__":
    main()
