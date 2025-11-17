CUDA-Accelerated Video Super-Resolution using RealESRGAN (240p → 720p)
DSP High Performance Computing Project – NJIT

This project implements a GPU-accelerated video super-resolution system using the RealESRGAN x4 model, PyTorch, and CUDA. The system enhances low-resolution 240p videos to 720p resolution by applying deep learning-based upscaling on each video frame. All heavy computations are executed on the GPU to achieve efficient high-performance processing.

1. Objective

The objective of this project is to design a high-performance deep-learning pipeline capable of:

Upscaling low-resolution videos (240p) to higher resolution (720p)

Utilizing CUDA acceleration for faster inference

Applying state-of-the-art GAN-based super-resolution (RealESRGAN)

Demonstrating DSP-based video enhancement using GPU computing

2. Features

CUDA-accelerated super-resolution inference

Converts 240p input videos to enhanced 720p output

Uses RealESRGAN x4 pretrained model

Modular code structure with separate model, processing, and execution scripts

Stable FP32 model inference compatible with mid-range GPUs (e.g., GTX 1650)

Automatic preprocessing and postprocessing pipeline

Clean and reproducible project structure for academic evaluation

3. System Pipeline

Input video is loaded using OpenCV.

Frames are extracted sequentially.

Each frame is converted from BGR to RGB and normalized.

Frames are converted into PyTorch tensors and moved to the CUDA device.

The RealESRGAN model performs 4× super-resolution.

Output frames are resized to exactly 1280×720.

Frames are reconstructed back into a video file using OpenCV.

The enhanced 720p video is saved to the output folder.

4. Project Structure
DSPproject/
│── main.py
│── README.md
│── requirements.txt
│── models/
│    └── RealESRGAN_x4plus.pth
│── data/
│    ├── input_videos/
│    └── output_videos/
│── src/
│    ├── esrgan_x4.py
│    └── video_processor.py

5. Technologies Used
Component	Technology
Deep Learning	PyTorch
Super-Resolution Model	RealESRGAN x4
GPU Acceleration	NVIDIA CUDA
Video Processing	OpenCV
Language	Python 3.x
6. How to Run the Project
Step 1: Install dependencies
pip install -r requirements.txt

Step 2: Add a low-resolution input video

Place a 240p (or similar) video inside:

data/input_videos/


Example file:

sample_240p.mp4

Step 3: Execute the main program
python main.py

Step 4: Retrieve the output video

Final enhanced video will appear in:

data/output_videos/

7. Performance Characteristics

Fully utilizes GPU acceleration through PyTorch CUDA backend.

Processes frames at approximately 1–3 FPS depending on video size and GPU capability.

Produces significantly better quality output than traditional interpolation methods.

Compatible with GPUs that have at least 4 GB VRAM (GTX 1650 tested).

8. Example Results (Conceptual)

Input (240p):

Low detail

Pixelation visible

Heavy compression artifacts

Output (720p):

Sharper edges

Reduced noise

Improved detail and readability

Cleaner facial features and motion stability

9. Author

Sathvik Chintalacheruvu
M.S. Computer Engineering
New Jersey Institute of Technology
DSP High-Performance Computing Laboratory

10. Future Work

Tile-based inference for processing larger videos on low-VRAM GPUs

Real-time upscaling pipeline using webcam streams

Temporal consistency enhancement using optical flow

Integration with RealESRNET + ESRGAN hybrid models
