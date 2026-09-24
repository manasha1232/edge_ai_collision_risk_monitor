"""
Generate Synthetic Dashcam Traffic Video for Edge-AI Pedestrian Collision Risk Monitor
Day 29 - 30-Day Computer Vision Challenge
"""

import os
import cv2
import numpy as np

def create_synthetic_traffic_demo(output_dir="output", num_frames=90):
    os.makedirs(output_dir, exist_ok=True)
    video_path = os.path.join(output_dir, "demo_dashcam.mp4")
    image_path = os.path.join(output_dir, "demo_dashcam_frame.jpg")

    width, height = 1280, 720
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 30.0, (width, height))

    print(f"[INFO] Generating {num_frames} synthetic dashcam traffic frames...")

    for frame_idx in range(num_frames):
        # Create road dashcam view background
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Sky
        frame[:300, :] = (200, 160, 100)
        # Horizon & Road
        for y in range(300, height):
            val = int(80 - 40 * ((y - 300) / 420.0))
            frame[y, :] = (val, val, val + 10)

        # Draw road perspective lane lines
        cv2.line(frame, (540, 300), (100, 720), (255, 255, 255), 4)
        cv2.line(frame, (740, 300), (1180, 720), (255, 255, 255), 4)
        cv2.line(frame, (640, 300), (640, 720), (0, 255, 255), 2) # Center line

        # Pedestrian approaching host vehicle (Centroid moves down and gets larger)
        t = frame_idx / float(num_frames) # 0.0 to 1.0
        p_y = int(320 + t * 320) # 320 -> 640
        p_x = int(600 + t * 40)  # 600 -> 640 (Dead center collision path)
        p_size = int(30 + t * 140) # Scale grows as distance decreases

        # Draw pedestrian representation (Bounding box + Head circle)
        cv2.rectangle(frame, (p_x - p_size//2, p_y - p_size), (p_x + p_size//2, p_y), (0, 100, 255), -1)
        cv2.circle(frame, (p_x, p_y - p_size - p_size//4), p_size//4, (0, 165, 255), -1)

        # Status text
        cv2.putText(frame, f"Dashcam Frame: {frame_idx+1}/{num_frames} | Target Approaching",
                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        out.write(frame)

        if frame_idx == num_frames - 1:
            cv2.imwrite(image_path, frame)

    out.release()
    print(f"[SUCCESS] Synthetic dashcam video saved to: {video_path}")
    print(f"[SUCCESS] Sample test frame saved to: {image_path}")

if __name__ == "__main__":
    create_synthetic_traffic_demo()
