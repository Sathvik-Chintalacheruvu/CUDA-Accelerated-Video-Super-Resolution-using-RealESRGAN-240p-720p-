import os
from src.video_processor import process_video

def main():
    print(" GPU-Accelerated ESRGAN x4 Video Upscaler (240p -> 720p) ")

    # Correct folder locations for your project
    input_dir = "data/input_videos"
    output_dir = "data/output_videos"

    # Make sure they exist
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Input file (place your video here: data/input_videos/)
    input_name = "sample_240p.mp4"
    input_path = os.path.join(input_dir, input_name)

    # Output file
    output_name = os.path.splitext(input_name)[0] + "_sr720p.mp4"
    output_path = os.path.join(output_dir, output_name)

    print(f" Input:  {input_path}")
    print(f" Output: {output_path}")

    process_video(input_path, output_path)

    print("\n Upscaling Completed Successfully!")
    print(f" Output saved at: {output_path}")

if __name__ == "__main__":
    main()
