import argparse
import numpy as np
import cv2
from scipy.spatial import cKDTree
from scipy.ndimage import median_filter


class NanoparticlePreprocessor:
    def __init__(
        self,
        crop_size=31,
        threshold_percentile=99.5,
        min_area=2,
        max_area=200,
        blur_ksize=3,
        min_separation=None,
        max_link_distance=8,
        max_gap=2,
        min_track_length=3,
        background_ksize=15,
    ):
        self.crop_size = crop_size
        self.threshold_percentile = threshold_percentile
        self.min_area = min_area
        self.max_area = max_area
        self.blur_ksize = blur_ksize
        self.min_separation = crop_size if min_separation is None else min_separation
        self.max_link_distance = max_link_distance
        self.max_gap = max_gap
        self.min_track_length = min_track_length
        self.background_ksize = background_ksize

    def process_video(self, video_path):
        frames, detections = self._read_and_detect(video_path)
        tracks = self._link(detections)

        crops = []
        for track in tracks:
            if len(track) < self.min_track_length:
                continue
            for f, cx, cy in track:
                crop = self._extract_crop(frames[f], detections[f], cx, cy)
                if crop is not None:
                    crops.append(crop)

        if not crops:
            raise ValueError("No crops extracted. Loosen detection or tracking thresholds.")

        crops = np.stack(crops, axis=0).astype(np.float32)
        return self._flatten_and_normalize(crops)

    def _read_and_detect(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video: {video_path}")

        frames = []
        detections = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            gray = self._to_gray(frame)
            frames.append(gray)
            detections.append(self._detect_centers(gray))
        cap.release()

        if not frames:
            raise ValueError(f"No frames read from {video_path}")
        return frames, detections

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
        return np.array(centers, dtype=np.float32) if centers else np.empty((0, 2), np.float32)

    def _link(self, detections):
        open_tracks = []
        closed_tracks = []

        for f, dets in enumerate(detections):
            n = len(dets)
            claimed = np.zeros(n, dtype=bool)

            if n > 0:
                tree = cKDTree(dets)
                still_open = []
                for track, last_f, last_pos in open_tracks:
                    if f - last_f > self.max_gap:
                        closed_tracks.append(track)
                        continue
                    d, j = tree.query(last_pos)
                    if d <= self.max_link_distance and not claimed[j]:
                        claimed[j] = True
                        cx, cy = dets[j]
                        track.append((f, cx, cy))
                        still_open.append([track, f, dets[j]])
                    else:
                        still_open.append([track, last_f, last_pos])
                open_tracks = still_open

                for j in range(n):
                    if not claimed[j]:
                        cx, cy = dets[j]
                        open_tracks.append([[(f, cx, cy)], f, dets[j]])
            else:
                still_open = []
                for track, last_f, last_pos in open_tracks:
                    if f - last_f > self.max_gap:
                        closed_tracks.append(track)
                    else:
                        still_open.append([track, last_f, last_pos])
                open_tracks = still_open

        for track, _, _ in open_tracks:
            closed_tracks.append(track)
        return closed_tracks

    def _extract_crop(self, gray, frame_dets, cx, cy):
        h, w = gray.shape
        half = self.crop_size // 2
        cx = int(round(cx))
        cy = int(round(cy))

        if self._has_neighbor(cx, cy, frame_dets):
            return None

        x0, x1 = cx - half, cx + half + 1
        y0, y1 = cy - half, cy + half + 1
        if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
            return None

        crop = gray[y0:y1, x0:x1]
        return self._subtract_background(crop)

    def _has_neighbor(self, cx, cy, pts):
        if len(pts) == 0:
            return False
        d2 = (pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2
        return int((d2 < self.min_separation ** 2).sum()) > 1

    def _subtract_background(self, crop):
        k = self.background_ksize
        if k % 2 == 0:
            k += 1
        bg = median_filter(crop.astype(np.float32), size=k, mode="nearest")
        out = crop - bg
        np.clip(out, 0, None, out=out)
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


def report_peak_spread(new_crops, old_path, tag):
    print(f"\nPeak-value spread ({tag}) [percentiles 1/25/50/75/99]:")
    new_pk = new_crops.max(axis=1)
    qs = [1, 25, 50, 75, 99]
    print(f"  new  {tag}: {np.round(np.percentile(new_pk, qs), 4)}  spread(99-1)={float(np.percentile(new_pk,99)-np.percentile(new_pk,1)):.4f}")
    try:
        old = np.load(old_path)
        old_pk = old.max(axis=1)
        print(f"  old  {old_path}: {np.round(np.percentile(old_pk, qs), 4)}  spread(99-1)={float(np.percentile(old_pk,99)-np.percentile(old_pk,1)):.4f}")
    except Exception as e:
        print(f"  old  {old_path}: could not load ({e})")


def main():
    parser = argparse.ArgumentParser(
        description="Track nanoparticles across frames in an NTA video and export "
        "centered, background-subtracted, flattened, L2-normalized crops for the "
        "1D homogeneous recovery experiment. No single-peak filter: defocused "
        "particles are kept so the focal-depth latent varies across the dataset."
    )
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--crop-size", type=int, default=31)
    parser.add_argument("--threshold-percentile", type=float, default=99.5)
    parser.add_argument("--min-area", type=int, default=2)
    parser.add_argument("--max-area", type=int, default=200)
    parser.add_argument("--blur-ksize", type=int, default=3)
    parser.add_argument("--min-separation", type=int, default=None)
    parser.add_argument("--max-link-distance", type=float, default=8.0)
    parser.add_argument("--max-gap", type=int, default=2)
    parser.add_argument("--min-track-length", type=int, default=3)
    parser.add_argument("--background-ksize", type=int, default=15)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--sample-grid", type=str, default=None)
    parser.add_argument("--compare-old", type=str, default=None)
    args = parser.parse_args()

    pre = NanoparticlePreprocessor(
        crop_size=args.crop_size,
        threshold_percentile=args.threshold_percentile,
        min_area=args.min_area,
        max_area=args.max_area,
        blur_ksize=args.blur_ksize,
        min_separation=args.min_separation,
        max_link_distance=args.max_link_distance,
        max_gap=args.max_gap,
        min_track_length=args.min_track_length,
        background_ksize=args.background_ksize,
    )

    if args.max_frames is not None:
        crops = _process_capped(pre, args.video, args.max_frames)
    else:
        crops = pre.process_video(args.video)

    np.save(args.output, crops)
    print(f"Saved {crops.shape[0]} crops of dim {crops.shape[1]} to {args.output}")

    if args.compare_old is not None:
        report_peak_spread(crops, args.compare_old, tag=args.output)

    if args.sample_grid is not None:
        pre.save_sample_grid(crops, args.sample_grid)
        print(f"Saved sample crop grid to {args.sample_grid}")


def _process_capped(pre, video_path, max_frames):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")
    frames, detections = [], []
    while len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        gray = pre._to_gray(frame)
        frames.append(gray)
        detections.append(pre._detect_centers(gray))
    cap.release()
    if not frames:
        raise ValueError(f"No frames read from {video_path}")

    tracks = pre._link(detections)
    crops = []
    for track in tracks:
        if len(track) < pre.min_track_length:
            continue
        for f, cx, cy in track:
            crop = pre._extract_crop(frames[f], detections[f], cx, cy)
            if crop is not None:
                crops.append(crop)
    if not crops:
        raise ValueError("No crops extracted. Loosen detection or tracking thresholds.")
    crops = np.stack(crops, axis=0).astype(np.float32)
    return pre._flatten_and_normalize(crops)


if __name__ == "__main__":
    main()
