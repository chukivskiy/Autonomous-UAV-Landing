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

# === КОНФІГУРАЦІЯ ===
LANDING_SIZE_M = 1.0  # Шукаємо зону >=1м x 1м
MIN_LANDING_AREA_M2 = LANDING_SIZE_M ** 2  # 1.0 м²
MIN_DISTANCE_M = 0.5  # Мінімальна відстань до краю (буфер безпеки, м)
UPDATE_INTERVAL = 0.1
EXPANSION_RADIUS = 15

class LandingAnalyzerNode(Node):
    def __init__(self):
        super().__init__('landing_analyzer')
        mavros_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
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
        self.create_subscription(
            PoseStamped, '/mavros/local_position/pose',
            self.pose_callback, mavros_qos
        )
        # Публікатори
        self.mask_pub = self.create_publisher(Image, '/landing_mask', 10)
        self.debug_pub = self.create_publisher(Image, '/debug_image', 10)
        self.landing_target_pub = self.create_publisher(PointStamped, '/landing_target', 10)  # Новий: публікує відносну позицію точки (delta_x, delta_y, delta_z=0)
        self.get_logger().info('Запущено. Очікування висоти з MAVROS (Best Effort)...')

    def info_callback(self, msg):
        if self.latest_info is None:
            self.latest_info = msg
            width = msg.width
            height = msg.height
            fx = msg.k[0]
            fy = msg.k[4]
            if fx > 0 and fy > 0:
                self.fov_h_deg = math.degrees(2.0 * math.atan(width / (2.0 * fx)))
                self.fov_v_deg = math.degrees(2.0 * math.atan(height / (2.0 * fy)))
                self.get_logger().info(
                    f'FOV: Горизонтальний {self.fov_h_deg:.1f}°, Вертикальний {self.fov_v_deg:.1f}°'
                )

    def pose_callback(self, msg):
        height = msg.pose.position.z
        if height > 0.1:
            self.current_height = height
            self.get_logger().info(f'Висота отримана: {self.current_height:.2f} м')
        else:
            self.current_height = 0.1

    def line_expansion(self, edges, radius):
        expanded = np.zeros_like(edges)
        points = np.column_stack(np.where(edges == 255))
        for y, x in points:
            cv2.circle(expanded, (x, y), radius, 255, -1)
        return expanded

    def get_final_mask(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
        median = np.median(blurred)
        low = int(max(0, 0.66 * median))
        high = int(min(255, 1.33 * median))
        edges = cv2.Canny(blurred, low, high)
        expanded = self.line_expansion(edges, EXPANSION_RADIUS)
        kernel = np.ones((3, 3), np.uint8)
        return cv2.morphologyEx(expanded, cv2.MORPH_CLOSE, kernel, iterations=2)

    def pixel_to_m2(self, pixel_area, img_width, img_height):
        if not self.fov_h_deg or not self.current_height:
            return 0.0
        ground_width_m = 2 * self.current_height * math.tan(math.radians(self.fov_h_deg) / 2)
        ground_height_m = 2 * self.current_height * math.tan(math.radians(self.fov_v_deg) / 2)
        m_per_pixel_h = ground_width_m / img_width
        m_per_pixel_v = ground_height_m / img_height
        m_per_pixel = (m_per_pixel_h + m_per_pixel_v) / 2
        return pixel_area * (m_per_pixel ** 2)

    def find_landing_spot(self, free_mask):
        img_h, img_w = free_mask.shape
        # Обчислюємо розмір 1м x 1м у пікселях та пікселі на метр
        ground_width_m = 2 * self.current_height * math.tan(math.radians(self.fov_h_deg) / 2)
        ground_height_m = 2 * self.current_height * math.tan(math.radians(self.fov_v_deg) / 2)
        px_per_m_h = img_w / ground_width_m
        px_per_m_v = img_h / ground_height_m
        px_per_m = (px_per_m_h + px_per_m_v) / 2
        required_size = int(round(LANDING_SIZE_M * px_per_m))  # пікселів на сторону
        required_dist_px = int(round(MIN_DISTANCE_M * px_per_m))  # Мінімальна відстань у пікселях
        if required_size < 10:
            return None
        # Шукаємо найбільший суцільний компонент
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(free_mask, connectivity=8)
        best_label = None
        best_area = 0
        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area > best_area:
                best_area = area
                best_label = label
        if best_label is None:
            return None
        # Створюємо маску тільки для найкращого компонента
        component_mask = (labels == best_label).astype(np.uint8) * 255
        # Перевіряємо розмір компонента
        x, y, w, h, _ = stats[best_label]
        if w < required_size or h < required_size:
            return None
        # Distance Transform: карта відстаней до найближчого краю
        dist_map = cv2.distanceTransform(component_mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        # Знаходимо максимальну відстань та її позицію
        max_dist = np.max(dist_map)
        if max_dist < required_dist_px:  # Замала відстань — небезпечно
            return None
        # Позиція максимуму (y, x) — точка з max відстанню
        cy, cx = np.unravel_index(np.argmax(dist_map), dist_map.shape)
        return (cx, cy)  # (x, y) у пікселях

    def compute_relative_position(self, landing_center_px, img_width, img_height):
        if landing_center_px is None:
            return None
        cx, cy = landing_center_px
        center_x = img_width / 2.0
        center_y = img_height / 2.0
        ground_width_m = 2 * self.current_height * math.tan(math.radians(self.fov_h_deg) / 2)
        ground_height_m = 2 * self.current_height * math.tan(math.radians(self.fov_v_deg) / 2)
        m_per_pixel_h = ground_width_m / img_width
        m_per_pixel_v = ground_height_m / img_height
        
        # delta_y (вправо/East) = (cx - center_x) * m_per_pixel_h (x в зображенні — вправо)
        delta_y = -(cy - center_y) * m_per_pixel_v  # Вперед/назад
        delta_x = (cx - center_x) * m_per_pixel_h  # Ліво/право
        delta_z = 0.0  # Для посадки — спочатку на поточній висоті
        return (delta_x, delta_y, delta_z)

    def image_callback(self, msg):
        if not self.latest_info or self.current_height is None:
            self.get_logger().warning('Немає camera_info або висоти — пропускаю кадр')
            return
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'CvBridge: {e}')
            return
        # ОБРОБКА
        obstacle_mask = self.get_final_mask(cv_image)
        free_mask = 255 - obstacle_mask
        free_pixels = np.sum(free_mask == 255)
        free_area_m2 = self.pixel_to_m2(free_pixels, cv_image.shape[1], cv_image.shape[0])
        can_land = free_area_m2 >= MIN_LANDING_AREA_M2
        landing_status = "МОЖНА СІСТИ" if can_land else "НЕБЕЗПЕЧНО"
        landing_color = 'green' if can_land else 'red'
        # Пошук місця посадки
        landing_center_px = self.find_landing_spot(free_mask)
        # Обчислення відносної позиції та публікація
        relative_pos = self.compute_relative_position(landing_center_px, cv_image.shape[1], cv_image.shape[0])
        if relative_pos is not None:
            target_msg = PointStamped()
            target_msg.header = msg.header
            target_msg.point.x, target_msg.point.y, target_msg.point.z = relative_pos
            self.landing_target_pub.publish(target_msg)
            self.get_logger().info(f'Опубліковано ціль: delta=({target_msg.point.x:.2f}, {target_msg.point.y:.2f}, {target_msg.point.z:.2f}) | Площа={free_area_m2:.2f}м² | Можна сідати={can_land}')
        else:
            self.get_logger().debug('Не знайдено місця посадки — не публікую target')
        # ПУБЛІКАЦІЯ маски та дебаг
        mask_msg = self.bridge.cv2_to_imgmsg(free_mask, 'mono8')
        mask_msg.header = msg.header
        self.mask_pub.publish(mask_msg)
        debug = cv_image.copy()
        overlay = np.zeros_like(cv_image)
        overlay[free_mask == 255] = [0, 255, 0]
        overlay[free_mask == 0] = [0, 0, 255]
        cv2.addWeighted(overlay, 0.4, debug, 0.6, 0, debug)
        if landing_center_px is not None:
            cx, cy = landing_center_px
            cv2.circle(debug, (cx, cy), 15, (0, 0, 255), -1)
            cv2.circle(debug, (cx, cy), 20, (255, 255, 255), 3)
        debug_msg = self.bridge.cv2_to_imgmsg(debug, 'bgr8')
        debug_msg.header = msg.header
        self.debug_pub.publish(debug_msg)
        # ВІЗУАЛІЗАЦІЯ Matplotlib
        current_time = self.get_clock().now().nanoseconds / 1e9
        if current_time - self.last_update >= UPDATE_INTERVAL:
            self.frame_count += 1
            self.ax1.clear()
            self.ax1.imshow(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
            self.ax1.set_title("Оригінал")
            self.ax1.axis('off')
            if landing_center_px is not None:
                self.ax1.plot(landing_center_px[0], landing_center_px[1], 'ro', markersize=12, markeredgewidth=3, markerfacecolor='red')
            self.ax2.clear()
            self.ax2.imshow(free_mask, cmap='gray')
            self.ax2.set_title("Вільна зона (біле)", color='green')
            self.ax2.axis('off')
            if landing_center_px is not None:
                self.ax2.plot(landing_center_px[0], landing_center_px[1], 'ro', markersize=12, markeredgewidth=3)
            info = (
                f"Кадр: {self.frame_count} | Висота: {self.current_height:.2f} м\n"
                f"Вільно: {free_area_m2:.2f} м² (потрібно ≥{MIN_LANDING_AREA_M2} м²)\n"
                f"ПОСАДКА: {landing_status}"
            )
            if landing_center_px is not None:
                info += "\nЗНАЙДЕНО МІСЦЕ ДЛЯ ПОСАДКИ (червона точка)"
                landing_color = 'green'
            self.fig.suptitle(info, fontsize=14, color=landing_color)
            plt.draw()
            plt.pause(0.001)
            self.last_update = current_time
        # ЛОГ
        status_extra = " + місце знайдено" if landing_center_px is not None else ""
        self.get_logger().info(
            f'Вільно: {free_area_m2:.2f} м² | {landing_status}{status_extra} | h={self.current_height:.2f}м'
        )

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
