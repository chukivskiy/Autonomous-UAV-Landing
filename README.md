# Autonomous UAV Landing

**System for autonomous emergency landing of VTOL drones** in non-anthropogenic environments.

The project uses **semantic segmentation** to find a safe landing zone, Archimedean spiral search for area coverage, and PID-controlled descent (only on the Z-axis).

---

### Demo Video

![Autonomous Landing Demo](https://github.com/user-attachments/assets/f0ec7b8a-8c8c-472b-9189-40bccf9911b1)

---

### Key Features

- Real-time image processing
- Priority-based landing zone evaluation (grass → ground → sand → etc.)
- **Archimedean spiral** search pattern
- PID controller for vertical descent
- Adaptive yaw when fixing the landing target
- Integrated with **ROS 2**, **PX4**, and **NVIDIA Isaac Sim**

---

### Tech Stack

- **ROS 2** (Humble / Jazzy)
- **PX4** + **MAVROS**
- **NVIDIA Isaac Sim**
- **PyTorch** + **segmentation-models-pytorch**
- **OpenCV**, **CvBridge**, **Albumentations**

---

### How to Run

```bash
source /opt/ros/humble/setup.bash
# Also source Isaac Sim / PX4 environment if needed

# Semantic image processor
python3 image_processor.py

# Landing controller
python3 landing_controller.py

# Spiral search (optional)
python3 flight_search.py




