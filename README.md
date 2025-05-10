# ⚡ Event Camera Motion Compensation Lab

This repository provides a hands-on lab for experimenting with motion compensation in **event-based vision systems**. Event cameras, unlike traditional frame-based cameras, record changes in intensity asynchronously, offering ultra-low latency and high temporal resolution—making them ideal for fast-motion applications.
 
📜 Main Script: `motion_tutorial.py`  
📂 Provided Sequences:  
- `shorter_sequence/`, `Longer_sequence/`  
- `with aee/`, `without_aee/`

🎓 Instructor: **Jad Mansour – PhD Student, Universitat de Girona**  
🏫 Used in: **Perception Lab** on Event-Based Vision

---

## 🧪 Objective

This lab focuses on:

- Warping event streams to perform motion compensation
- Visualizing intermediate outputs such as:
  - Image of Warped Events (IWE)
  - Ground truth flow
  - Compensation flow
  - Pre/post warp event distributions
- Comparing performance with and without **Average Endpoint Error (AEE)** optimization

---

## 📂 Dataset

📥 **Download the full dataset:**  
[🔗 Google Drive – Event Camera Lab Files](https://drive.google.com/drive/folders/1tCNAaVO-GMZh2o2z6vUaNSqhjgWZGaWt?usp=sharing)

> After downloading, unzip `Event_camera_part2.zip` and place it in the project root directory.

---

## 🗃️ Folder Structure

```bash
Event_camera_part2/
├── ewiz/                  # Event warping and interpolation toolkit
├── Longer_sequence/       # Full motion sequence visualizations and metrics
├── shorter_sequence/      # Short motion sequence for quick tests
├── with aee/              # Output when AEE optimization is applied
├── without_aee/           # Output without AEE optimization
├── motion_tutorial.py     # Main lab script (entry point)
