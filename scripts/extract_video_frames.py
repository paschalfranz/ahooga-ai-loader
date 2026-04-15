import cv2
from pathlib import Path
import shutil

# -------------------------
# CONFIG
# -------------------------

VIDEO_ROOT = Path("videos_local")          # folder containing your local videos
OUTPUT_ROOT = Path("video_frames")         # folder where extracted frames will be saved
FRAME_EVERY_SECONDS = 1.0                  # sample interval (changed from 2.0 to 1.0)

# Optional safety switch:
# if True, script deletes and recreates video_frames automatically
CLEAR_OUTPUT_ROOT = False


# -------------------------
# FRAME EXTRACTION
# -------------------------

def extract_frames_from_video(video_path: Path, output_dir: Path, every_seconds: float = 1.0) -> int:
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"ERROR: Could not open video: {video_path}")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        print(f"ERROR: Could not read FPS for: {video_path}")
        cap.release()
        return 0

    frame_interval = max(1, int(fps * every_seconds))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_seconds = total_frames / fps if fps > 0 else 0

    output_dir.mkdir(parents=True, exist_ok=True)

    saved_count = 0
    frame_index = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        if frame_index % frame_interval == 0:
            timestamp_seconds = frame_index / fps
            output_name = f"{video_path.stem}_t{timestamp_seconds:07.2f}.jpg"
            output_path = output_dir / output_name
            cv2.imwrite(str(output_path), frame)
            saved_count += 1

        frame_index += 1

    cap.release()

    print(
        f"OK: {video_path.name} | duration={duration_seconds:.1f}s | "
        f"saved={saved_count} frames | every={every_seconds}s"
    )
    return saved_count


def main():
    if not VIDEO_ROOT.exists():
        print(f"ERROR: Video folder not found: {VIDEO_ROOT.resolve()}")
        return

    video_files = []
    for ext in ["*.mp4", "*.mov", "*.avi", "*.mkv"]:
        video_files.extend(VIDEO_ROOT.rglob(ext))

    if not video_files:
        print(f"No videos found in: {VIDEO_ROOT.resolve()}")
        return

    if CLEAR_OUTPUT_ROOT and OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    total_saved = 0
    for video_path in sorted(video_files):
        video_output_dir = OUTPUT_ROOT / video_path.stem
        total_saved += extract_frames_from_video(
            video_path=video_path,
            output_dir=video_output_dir,
            every_seconds=FRAME_EVERY_SECONDS
        )

    print(f"\nDone. Total frames saved: {total_saved}")


if __name__ == "__main__":
    main()