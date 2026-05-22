#!/usr/bin/env python3
import sys
import numpy as np
if np.__version__.startswith('2.'):
    print("ПОМИЛКА: NumPy 2.x несумісний!")
    sys.exit(1)

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, PointStamped
from cv_bridge import CvBridge
import cv2
import math
import matplotlib.pyplot as plt
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

# === КОНФІГУРАЦІЯ ===
LANDING_SIZE_M = 1.0
MIN_LANDING_AREA_M2 = LANDING_SIZE_M ** 2
MIN_DISTANCE_M = 0.5          # буфер безпеки до краю
UPDATE_INTERVAL = 0.1
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = "/home/drone_landing/Weights/best_model2.pth"
NUM_CLASSES = 8
TARGET_HEIGHT, TARGET_WIDTH = 432, 768

# Пріоритет класів (від найкращого до гіршого)
PRIORITY_ORDER = [1, 2, 4, 3, 5, 6, 7]  # grass, ground, sand, pebbles, stone, tree, water

class LandingAnalyzerNode(Node):
    def __init__(self):
        super().__init__('landing_analyzer')

        mavros_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                                history=HistoryPolicy.KEEP_LAST, depth=1)
        image_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                               history=HistoryPolicy.KEEP_LAST, depth=10)

        self.bridge = CvBridge()
        self.latest_info = None
        self.current_height = None
        self.frame_count = 0
        self.last_update = 0.0
        self.fov_h_deg = None
        self.fov_v_deg = None

        # Matplotlib
        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(18, 8))
        self.fig.suptitle("Очікування висоти...", fontsize=16, color='orange')
        plt.tight_layout()

        # Підписки
        self.create_subscription(CameraInfo, '/camera_info', self.info_callback, 10)
        self.create_subscription(Image, '/rgb', self.image_callback, image_qos)
        self.create_subscription(PoseStamped, '/mavros/local_position/pose',
                                 self.pose_callback, mavros_qos)

        # Публікатори
        self.mask_pub = self.create_publisher(Image, '/landing_mask', 10)        # binary free mask
        self.debug_pub = self.create_publisher(Image, '/debug_image', 10)
        self.landing_target_pub = self.create_publisher(PointStamped, '/landing_target', 10)

        self.get_logger().info('Запущено з семантичною сегментацією + пріоритетами посадки...')

        # === Модель ===
        self.model = smp.Unet(
            encoder_name="efficientnet-b4",
            encoder_weights=None,
            in_channels=3,
            classes=NUM_CLASSES,
        )
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
        self.model.to(DEVICE)
        self.model.eval()

        self.transform = A.Compose([
            A.Resize(TARGET_HEIGHT, TARGET_WIDTH),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

        # Кольори для візуалізації
        self.class_colors = np.array([
            [0, 0, 0],      # 0 background
            [0, 204, 0],    # 1 grass
            [153, 102, 51], # 2 ground
            [128, 128, 128],# 3 pebbles
            [242, 230, 153],# 4 sand
            [77, 77, 77],   # 5 stone
            [0, 102, 0],    # 6 tree
            [0, 128, 255],  # 7 water
        ], dtype=np.uint8)

    def info_callback(self, msg):
        if self.latest_info is None:
            self.latest_info = msg
            w, h = msg.width, msg.height
            fx, fy = msg.k[0], msg.k[4]
            if fx > 0 and fy > 0:
                self.fov_h_deg = math.degrees(2 * math.atan(w / (2 * fx)))
                self.fov_v_deg = math.degrees(2 * math.atan(h / (2 * fy)))
                self.get_logger().info(f'FOV: horiz {self.fov_h_deg:.1f}°, vert {self.fov_v_deg:.1f}°')

    def pose_callback(self, msg):
        h = msg.pose.position.z
        self.current_height = max(h, 0.1)

    def get_semantic_mask(self, img_bgr):
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        augmented = self.transform(image=rgb)
        tensor = augmented['image'].unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = self.model(tensor)
        pred = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)

        h, w = img_bgr.shape[:2]
        return cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)

    def pixel_to_m2(self, pixel_area, w, h):
        if not self.fov_h_deg or self.current_height is None:
            return 0.0
        gw = 2 * self.current_height * math.tan(math.radians(self.fov_h_deg) / 2)
        gh = 2 * self.current_height * math.tan(math.radians(self.fov_v_deg) / 2)
        mpp = ((gw / w) + (gh / h)) / 2
        return pixel_area * (mpp ** 2)

    def find_best_landing_spot(self, semantic_mask):
        """Повертає (cx, cy, best_class) або None"""
        img_h, img_w = semantic_mask.shape
        if self.current_height is None or not self.fov_h_deg:
            return None

        # Розмір у пікселях
        ground_w_m = 2 * self.current_height * math.tan(math.radians(self.fov_h_deg) / 2)
        ground_h_m = 2 * self.current_height * math.tan(math.radians(self.fov_v_deg) / 2)
        px_per_m = ((img_w / ground_w_m) + (img_h / ground_h_m)) / 2
        req_px = int(round(LANDING_SIZE_M * px_per_m))
        safety_px = int(round(MIN_DISTANCE_M * px_per_m))

        if req_px < 10:
            return None

        best_spot = None
        best_priority_idx = len(PRIORITY_ORDER) + 1
        best_area = 0

        for priority_idx, cls in enumerate(PRIORITY_ORDER):
            class_mask = (semantic_mask == cls).astype(np.uint8) * 255

            # Знаходимо компоненти
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                class_mask, connectivity=8)

            for label in range(1, num_labels):
                area_px = stats[label, cv2.CC_STAT_AREA]
                if area_px < req_px * req_px // 2:   # фільтрація
                    continue

                x, y, w, h, _ = stats[label]
                if w < req_px or h < req_px:
                    continue

                component_mask = (labels == label).astype(np.uint8) * 255
                dist_map = cv2.distanceTransform(component_mask, cv2.DIST_L2, 5)

                _, max_val, _, max_loc = cv2.minMaxLoc(dist_map)
                if max_val < safety_px:
                    continue

                # Пріоритетність
                if (priority_idx < best_priority_idx or
                    (priority_idx == best_priority_idx and area_px > best_area)):
                    best_priority_idx = priority_idx
                    best_area = area_px
                    best_spot = (max_loc[0], max_loc[1], cls)   # x, y, class

        return best_spot

    def compute_relative_position(self, cx, cy, img_w, img_h):
        center_x = img_w / 2.0
        center_y = img_h / 2.0

        ground_w_m = 2 * self.current_height * math.tan(math.radians(self.fov_h_deg) / 2)
        ground_h_m = 2 * self.current_height * math.tan(math.radians(self.fov_v_deg) / 2)

        m_per_px_h = ground_w_m / img_w
        m_per_px_v = ground_h_m / img_h

        # камера вниз, y на зображенні = вперед (North), x = вправо (East)
        delta_x = (cx - center_x) * m_per_px_h      # East
        delta_y = -(cy - center_y) * m_per_px_v     # North (мінус бо y зростає вниз)
        return delta_x, delta_y, 0.0

    def image_callback(self, msg):
        if self.latest_info is None or self.current_height is None:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'CvBridge error: {e}')
            return

        semantic_mask = self.get_semantic_mask(cv_image)

        # Створюємо маску для загальної інформації (grass + ground + sand)
        free_mask = np.zeros_like(semantic_mask, dtype=np.uint8)
        for cls in [1, 2, 4]:                     # grass, ground, sand
            free_mask[semantic_mask == cls] = 255

        free_area_px = np.sum(free_mask == 255)
        free_area_m2 = self.pixel_to_m2(free_area_px, cv_image.shape[1], cv_image.shape[0])
        can_land_anywhere = free_area_m2 >= MIN_LANDING_AREA_M2

        # === Пошук найкращої точки ===
        best = self.find_best_landing_spot(semantic_mask)
        landing_center_px = None
        best_class = None

        if best:
            cx, cy, best_class = best
            landing_center_px = (cx, cy)

            rel_pos = self.compute_relative_position(cx, cy, cv_image.shape[1], cv_image.shape[0])
            target_msg = PointStamped()
            target_msg.header = msg.header
            target_msg.point.x = rel_pos[0]
            target_msg.point.y = rel_pos[1]
            target_msg.point.z = rel_pos[2]
            self.landing_target_pub.publish(target_msg)

            status = f"МОЖНА СІСТИ ({self.class_name(best_class)})"
            color = 'green'
        else:
            status = "НЕБЕЗПЕЧНО"
            color = 'red'

        # Публікація масок
        self.mask_pub.publish(self.bridge.cv2_to_imgmsg(free_mask, 'mono8'))

        # Debug image
        debug = cv_image.copy()
        overlay = np.zeros_like(cv_image)
        overlay[free_mask == 255] = [0, 255, 0]
        overlay[free_mask == 0] = [0, 0, 255]
        cv2.addWeighted(overlay, 0.35, debug, 0.65, 0, debug)

        if landing_center_px:
            cx, cy = landing_center_px
            cv2.circle(debug, (cx, cy), 18, (0, 255, 255), -1)
            cv2.circle(debug, (cx, cy), 22, (255, 255, 255), 3)
            cv2.putText(debug, self.class_name(best_class), (cx+25, cy-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        self.debug_pub.publish(self.bridge.cv2_to_imgmsg(debug, 'bgr8'))

        # Matplotlib update
        now = self.get_clock().now().nanoseconds / 1e9
        if now - self.last_update >= UPDATE_INTERVAL:
            self.frame_count += 1
            self.ax1.clear()
            self.ax1.imshow(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
            self.ax1.set_title("Оригінал")
            self.ax1.axis('off')
            if landing_center_px:
                self.ax1.plot(landing_center_px[0], landing_center_px[1], 'yo', markersize=14, markeredgecolor='red')

            self.ax2.clear()
            color_mask = self.class_colors[semantic_mask]
            self.ax2.imshow(color_mask)
            self.ax2.set_title("Семантична сегментація")
            self.ax2.axis('off')

            info = (f"Кадр {self.frame_count} | Висота {self.current_height:.2f} м\n"
                    f"Вільна площа: {free_area_m2:.2f} м² | ПОСАДКА: {status}")
            self.fig.suptitle(info, fontsize=14, color=color)
            plt.draw()
            plt.pause(0.05)
            self.last_update = now

        self.get_logger().info(
            f"Вільно: {free_area_m2:.2f} м² | {status} | h={self.current_height:.2f} м"
        )

    def class_name(self, cls_id):
        names = {0: "background", 1: "ТРАВА", 2: "ГРУНТ", 3: "ГРАВІЙ",
                 4: "ПІСОК", 5: "КАМІННЯ", 6: "ДЕРЕВО", 7: "ВОДА"}
        return names.get(cls_id, "unknown")


def main():
    rclpy.init()
    node = LandingAnalyzerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nЗупинено.")
    finally:
        plt.ioff()
        plt.close('all')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
