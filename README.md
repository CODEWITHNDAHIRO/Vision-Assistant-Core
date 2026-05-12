
#  Vision-Assistant-Core
**An AI-powered gesture control system with smart power management and integrated telemetry.**

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Build](https://img.shields.io/badge/build-v1.0-orange.svg)

## Overview
Vision-Assistant-Core is a professional-grade Computer Vision application built over a 14-day development sprint. It leverages **MediaPipe** for high-fidelity hand tracking and features a custom-built **State Machine** for power efficiency (Privacy Guard/Standby Mode). The project transitioned from a monolithic script to a modular, configurable, and documented system.

### Key Milestones:
*   **Version Control:** Managed a professional Git branching workflow to increase commits and pull requests.
*   **Hardware Integration:** Resolved macOS security permissions and optimized camera lifecycle management.
*   **DevOps:** Implemented automated deployment via Shell scripting for seamless environment activation.
*   **Data Engineering:** Developed a telemetry engine to log and analyze gesture frequency in real-time.

---

##  Tech Stack
*   **Core Language:** Python 3.10+
*   **Automation:** Shell (Bash)
*   **AI/ML:** MediaPipe (Hand Landmarking)
*   **Computer Vision:** OpenCV
*   **Data Management:** JSON (Configuration), CSV (Usage Analytics)
*   **Operating System:** macOS (Optimized for Apple Silicon)

---

## Features
*   **Real-time Gesture Recognition:** Sub-millisecond hand landmarking via MediaPipe.
*   **Privacy Guard (Standby Mode):** Automatically dims the viewfinder and lowers CPU usage if no hand is detected for 5 seconds.
*   **External Configuration:** Manage system settings (notifications, camera index) via `config.json` without touching the code.
*   **Usage Telemetry:** Local logging of every interaction for performance auditing.
*   **Native Notifications:** System-level alerts via macOS `osascript`.

---

## Installation & Setup

### 1. Prerequisites
Ensure you have **Conda** installed and your camera permissions enabled for your terminal.

### 2. Clone and Environment Setup
```bash
git clone [https://github.com/CODEWITHNDAHIRO/Vision-Assistant-Core.git](https://github.com/CODEWITHNDAHIRO/Vision-Assistant-Core.git)
cd Vision-Assistant-Core
conda create -n vision-env python=3.10
conda activate vision-env
pip install opencv-python mediapipe pandas

### 3.Permissions
Give the launch script execution rights:
chmod +x run_assistant.sh

### 4.Usage
To launch the assistant with the automated deployment script:
./run_assistant.sh

active: use gestures to trigger system events
standby: remove hand from view for 5s to trigger " dimmed mode".
quit: press q in the camera window to exit safely

 ### Project Evolution & Debugging
This project includes a comprehensive FIX_LOG.md that documents the resolution of 7 critical issues, including:
1.    Nested function definitions.
2.    Camera lifecycle "Release" errors.
3.    MacOS-specific hardware permission blocks.
4.    Notification system implementation.
 ###Repository Structure:
 .
├── assistant.py         # Main AI Logic
├── config.json          # System Configuration
├── run_assistant.sh     # Automation Script
├── summarize_stats.py   # Analytics Summarizer
├── FIX_LOG.md           # Debugging History
├── .gitignore           # Version Control Settings
└── logs/                # Local Telemetry (Ignored by Git)
Author
Ndahiro P. — CODEWITHNDAHIRO (https://github.com/CODEWITHNDAHIRO)
Aspiring AI Expert & Computer vision Engineer
Developed during a 14-day AI Development Sprint.