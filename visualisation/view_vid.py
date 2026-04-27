import cv2
import matplotlib.pyplot as plt

VIDEO_PATH = "../data/behaviour/GH-2026-03-10_14-42-25_SV35.mp4"
#%%

def show_frame(frame_id: int, video_path: str = VIDEO_PATH) -> None:
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_id < 0 or frame_id >= total:
        cap.release()
        raise ValueError(f"frame_id {frame_id} out of range [0, {total - 1}]")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError(f"Could not read frame {frame_id}")

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots()
    ax.imshow(frame_rgb)
    ax.set_title(f"Frame {frame_id}")
    ax.axis("off")
    plt.tight_layout()
    return fig, ax, frame_rgb


if __name__ == "__main__":
    fig, ax, frame_rgb = show_frame(35450)
    ax.scatter(1340,500, color='r', s=3)

# %%
