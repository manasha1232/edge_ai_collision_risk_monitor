# 🚗 Edge-AI Autonomous Pedestrian Collision Risk Monitor

[![Day](https://img.shields.io/badge/Day-29--30-blue?style=for-the-badge&logo=python)](https://github.com/manasha1232/30-Day-Computer-Vision-Challenge)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.0.0-green?style=for-the-badge&logo=opencv)](https://opencv.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?style=for-the-badge&logo=pytorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-red?style=for-the-badge)](LICENSE)

An Autonomous Vehicle Safety & Edge-AI module for **Real-Time Pedestrian Collision Risk Estimation**. Utilizes a pinhole camera distance model, centroid velocity vector tracking, and Time-To-Collision ($\text{TTC} = d / v$) kinematics to trigger dynamic cockpit warning displays.

---

## 🌟 Key Features

- 📐 **Pinhole Camera Distance Geometry**: Estimates real-world object distance $d = \frac{f \cdot H_{\text{real}}}{h_{\text{bbox}}}$ ($H_{\text{real}} = 1.7\text{ m}$).
- ⚡ **Relative Velocity Vector Tracking**: Calculates relative approach velocity $v = -\frac{\Delta d}{\Delta t}$ ($\text{m/s}$ or $\text{km/h}$).
- 🚨 **Autonomous Safety Risk State Machine**:
  - `SAFE`: $\text{TTC} > 4.0\text{ s}$ (Green HUD)
  - `WARNING`: $2.0\text{ s} < \text{TTC} \le 4.0\text{ s}$ (Yellow HUD)
  - `CRITICAL BRAKE`: $\text{TTC} \le 2.0\text{ s}$ (Red Flash Banner HUD)
- 📊 **Telemetry Audit Exporter**: Exports JSON telemetry logs capturing minimum TTC, peak relative speed, and risk duration statistics.

---

## 🛠️ Installation & Setup

```bash
# Clone the repository
git clone https://github.com/manasha1232/edge_ai_collision_risk_monitor.git
cd edge_ai_collision_risk_monitor

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Execution Guide

### 1️⃣ Run with Synthetic Dashcam Traffic Generator
```bash
python generate_demo_traffic.py
python collision_risk_monitor.py
```

---

## 📊 Sample Output Telemetry JSON

```json
{
    "project": "Edge-AI Autonomous Pedestrian Collision Risk Monitor",
    "day": 29,
    "status": "SUCCESS",
    "resolution": {
        "width": 1280,
        "height": 720
    },
    "performance": {
        "total_frames_processed": 90,
        "execution_duration_sec": 1.45,
        "average_fps": 62.07
    },
    "collision_risk_metrics": {
        "min_time_to_collision_sec": 1.15,
        "max_relative_velocity_kmh": 28.45,
        "risk_state_counts": {
            "SAFE": 35,
            "WARNING": 25,
            "CRITICAL_BRAKE": 30
        }
    }
}
```

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).
