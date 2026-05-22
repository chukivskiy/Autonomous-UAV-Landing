#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from mavros_msgs.srv import SetMode, CommandBool
import time
import math

class HorizontalSpiral(Node):
    def __init__(self):
        super().__init__('horizontal_spiral')

        # ================== НАЛАШТУВАННЯ ==================
        self.target_height = 10.0      # висота, яку підтримуємо (м)
        self.initial_radius = 8.0      # початковий радіус 
        self.max_radius = 50.0         # максимальний радіус спіралі (м)
        self.radial_step = 3.0         # на скільки метрів радіус зростає за один повний виток
        self.linear_speed = 0.5        # м/с — спокійна швидкість 
        # =================================================

        self.pub = self.create_publisher(PoseStamped, '/mavros/setpoint_position/local', 10)

        self.set_mode = self.create_client(SetMode, '/mavros/set_mode')
        self.arm = self.create_client(CommandBool, '/mavros/cmd/arming')

        self.timer = self.create_timer(0.1, self.send_setpoint)

        self.subscription = self.create_subscription(
            PoseStamped,
            '/mavros/local_position/pose',
            self.position_callback,
            10)

        # Змінні спіралі
        self.angle = 0.0
        self.radius = self.initial_radius
        self.center_x = 0.0
        self.center_y = 0.0

        self.get_logger().info(f"Горизонтальна спіраль готова!")
        self.get_logger().info(f"Висота = {self.target_height} м | Швидкість = {self.linear_speed} м/с | Max R = {self.max_radius} м")

    def position_callback(self, msg):
        pass

    def send_setpoint(self):
        msg = PoseStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()

        # Точка на спіралі
        x = self.center_x + self.radius * math.cos(self.angle)
        y = self.center_y + self.radius * math.sin(self.angle)

        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = self.target_height   # ← постійно тримаємо висоту

        # Орієнтація вперед (по ходу руху)
        yaw = self.angle + math.pi / 2
        msg.pose.orientation.z = math.sin(yaw / 2.0)
        msg.pose.orientation.w = math.cos(yaw / 2.0)

        self.pub.publish(msg)

        # Плавне збільшення кута і радіуса
        angular_step = (self.linear_speed / self.radius) * 0.1
        self.angle += angular_step

        # Архімедова спіраль
        self.radius = self.initial_radius + (self.radial_step / (2 * math.pi)) * self.angle

        # Лог кожні 5 секунд (кожен виток)
        if int(self.angle / (2 * math.pi)) > int((self.angle - angular_step) / (2 * math.pi)):
            self.get_logger().info(f"Виток {int(self.angle/(2*math.pi))+1} | Радіус ≈ {self.radius:.1f} м")

        # Зупинка при досягненні максимального радіусу
        if self.radius >= self.max_radius:
            self.get_logger().info("Досягнуто максимального радіусу. Спіраль завершена.")
            self.timer.cancel()

    def run(self):
        self.get_logger().info("Чекаємо сервіси MAVROS...")
        while not self.set_mode.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Чекаємо /mavros/set_mode...')
        while not self.arm.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Чекаємо /mavros/cmd/arming...')

        time.sleep(1.5)

        # Переходимо в OFFBOARD
        req = SetMode.Request(custom_mode='OFFBOARD')
        future = self.set_mode.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        self.get_logger().info(f"Починаємо рухатися по горизонтальній спіралі на висоті {self.target_height} м")


def main():
    rclpy.init()
    node = HorizontalSpiral()
    try:
        node.run()
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Зупинено користувачем")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
