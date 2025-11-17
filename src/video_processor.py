import cv2
import torch
import numpy as np
from .esrgan_x4 import load_realesrgan_x4


def process_video(
    input_path: str,
    output_path: str,
    model_path: str = "models/RealESRGAN_x4plus.pth"
):
    """
    Takes a low-resolution video (240p–480p)
    → runs RealESRGAN x4 super-resolution
    → resizes final frames to 1280x720 (720p)
    → saves the output video.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(" Device:", device)

    # 1. Load ESRGAN x4 model
    model = load_realesrgan_x4(model_path, device=device, use_half=False)

    # 2. Open input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f" Input video: {in_w}x{in_h} @ {fps:.2f} fps, {total_frames} frames")

    out_writer = None
    frame_idx = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Convert to RGB and normalize
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb = frame_rgb.astype(np.float32) / 255.0

        # HWC → CHW tensor
        lr = torch.from_numpy(frame_rgb).permute(2, 0, 1).unsqueeze(0).float()


        lr = lr.to(device)

        with torch.no_grad():
            sr = model(lr)

        # Convert model output back to float32 RGB
        sr = sr.squeeze(0).permute(1, 2, 0).float().cpu().numpy()
        sr = np.clip(sr, 0.0, 1.0)

        # Convert to uint8 image
        sr = (sr * 255.0).round().astype(np.uint8)

        # RealESRGAN x4 output size (about 4× input)
        sr_h, sr_w, _ = sr.shape

        # Resize to EXACT 720p output (optional)
        target_w, target_h = 1280, 720
        sr_resized = cv2.resize(sr, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

        # Initialize VideoWriter once
        if out_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_writer = cv2.VideoWriter(output_path, fourcc, fps, (target_w, target_h))
            print(f" Output video: {target_w}x{target_h} @ {fps:.2f} fps")

        # Convert back to BGR for writing
        sr_bgr = cv2.cvtColor(sr_resized, cv2.COLOR_RGB2BGR)
        out_writer.write(sr_bgr)

        if frame_idx % 50 == 0:
            print(f"  Processed {frame_idx}/{total_frames} frames...")

    cap.release()
    if out_writer is not None:
        out_writer.release()

    print(" Finished super-resolution")
    print(f"   Saved to: {output_path}")
