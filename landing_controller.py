#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from geometry_msgs.msg import PoseStamped, TwistStamped, PointStamped, Quaternion
from mavros_msgs.srv import SetMode, CommandBool, CommandTOL
from mavros_msgs.msg import State
import time
import math


def quaternion_from_euler(roll, pitch, yaw):
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    qx = cy * cp * sr - sy * sp * cr
    qy = sy * cp * sr + cy * sp * cr
    qz = sy * cp * cr - cy * sp * sr
    qw = cy * cp * cr + sy * sp * sr
    return (qx, qy, qz, qw)


class PID:
    def __init__(self, kp, ki=0.0, kd=0.0, integral_limit=1.5):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = integral_limit
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = None

    def update(self, error, dt=0.05):
        current_time = time.time()
        if self.prev_time is None:
            self.prev_time = current_time
            self.prev_error = error
            return 0.0
        if dt <= 0.001:
            dt = 0.05
        self.integral += error * dt
        self.integral = max(min(self.integral, self.integral_limit), -self.integral_limit)
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        self.prev_time = current_time
        return self.kp * error + self.ki * self.integral + self.kd * derivative


class LandingFlyer(Node):
    def __init__(self):
        super().__init__('landing_flyer')
        qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST
        )

        # Publishers
        self.position_pub = self.create_publisher(PoseStamped, '/mavros/setpoint_position/local', 10)
        self.velocity_pub = self.create_publisher(TwistStamped, '/mavros/setpoint_velocity/cmd_vel', 10)

        # Subscribers
        self.create_subscription(PoseStamped, '/mavros/local_position/pose', self.pose_callback, qos)
        self.create_subscription(State, '/mavros/state', self.state_callback, qos)
        self.create_subscription(PointStamped, '/landing_target', self.target_callback, 10)
        self.create_subscription(TwistStamped, '/mavros/local_position/velocity', self.velocity_callback, qos)

        # Services
        self.set_mode_client = self.create_client(SetMode, '/mavros/set_mode')
        self.arming_client = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.land_client = self.create_client(CommandTOL, '/mavros/cmd/land')

        self.create_timer(0.05, self.send_setpoint)

        # ================== PARAMETERS ==================
        self.tolerance = 0.40
        self.min_descent_velocity = -0.40

        # PID для Z
        self.pid_z = PID(kp=1.8, ki=0.35, kd=0.45, integral_limit=2.0)

        # State variables
        self.current_pose = None
        self.current_velocity = None
        self.current_state = State()
        self.fixed_goal = None
        self.goal_x = 0.0
        self.goal_y = 0.0
        self.initial_height = 0.0
        self.reference_yaw = -90.0
        self.is_landing = False
        self.descent_mode = False
        self.descent_start_time = None
        self.descent_start_height = None
        self.target_detected_time = None
        self.land_command_time = None

        # Metrics
        self.horizontal_errors = []
        self.horizontal_speeds = []
        self.vertical_speeds = []
        self.max_horiz_speed = 0.0
        self.max_vert_descent = 0.0

        self.get_logger().info("LandingFlyer запущено")

    def pose_callback(self, msg):
        self.current_pose = msg.pose

    def velocity_callback(self, msg):
        self.current_velocity = msg.twist
        h = math.hypot(msg.twist.linear.x, msg.twist.linear.y)
        v = abs(msg.twist.linear.z)
        if h > self.max_horiz_speed:
            self.max_horiz_speed = h
        if v > self.max_vert_descent:
            self.max_vert_descent = v

    def state_callback(self, msg):
        self.current_state = msg

    def target_callback(self, msg):
        if self.current_pose is None or self.fixed_goal is not None:
            return

        q = self.current_pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.reference_yaw = math.atan2(siny_cosp, cosy_cosp)
        self.reference_yaw -= math.radians(90.0)

        dx_body = msg.point.x
        dy_body = msg.point.y
        cos_y = math.cos(self.reference_yaw)
        sin_y = math.sin(self.reference_yaw)

        dx_local = dx_body * cos_y - dy_body * sin_y
        dy_local = dx_body * sin_y + dy_body * cos_y

        self.fixed_goal = (
            self.current_pose.position.x + dx_local,
            self.current_pose.position.y + dy_local,
            self.current_pose.position.z
        )
        self.goal_x, self.goal_y, _ = self.fixed_goal
        self.initial_height = self.current_pose.position.z
        self.target_detected_time = time.time()

        yaw_deg = math.degrees(self.reference_yaw)
        self.get_logger().info(
            f'Ціль зафіксована! (реальний yaw = {yaw_deg:.1f}°) '
            f'Підлітаємо горизонтально на висоті {self.initial_height:.2f} м'
        )

    def send_setpoint(self):
        if self.current_pose is None or self.fixed_goal is None:
            return

        err_x = self.goal_x - self.current_pose.position.x
        err_y = self.goal_y - self.current_pose.position.y
        err_z = self.initial_height - self.current_pose.position.z  # для PID Z
        dist_xy = math.hypot(err_x, err_y)
        current_z = self.current_pose.position.z

        if not self.descent_mode and dist_xy < self.tolerance:
            self.descent_mode = True
            self.descent_start_time = time.time()
            self.descent_start_height = current_z
            self.get_logger().info("НАД ЦІЛЛЮ! Перехід у velocity mode")

        if not self.descent_mode:
            # Position mode (горизонтальний підліт)
            pos_msg = PoseStamped()
            pos_msg.header.frame_id = "fcu"
            pos_msg.header.stamp = self.get_clock().now().to_msg()
            pos_msg.pose.position.x = self.goal_x
            pos_msg.pose.position.y = self.goal_y
            pos_msg.pose.position.z = self.initial_height
            qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, self.reference_yaw)
            pos_msg.pose.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)
            self.position_pub.publish(pos_msg)
            cmd_z_vel = 0.0
        else:
            # Velocity mode
            vel_msg = TwistStamped()
            vel_msg.header.frame_id = "fcu"
            vel_msg.header.stamp = self.get_clock().now().to_msg()
            vel_x = err_x * 1.1
            vel_y = err_y * 1.1
            vel_z = self.pid_z.update(err_z)

            # Обмеження
            max_horiz = 1.6 if current_z > 2.5 else 0.9
            vel_msg.twist.linear.x = max(min(vel_x, max_horiz), -max_horiz)
            vel_msg.twist.linear.y = max(min(vel_y, max_horiz), -max_horiz)
            vel_msg.twist.linear.z = max(min(vel_z, -0.3), -2.2)  # обмежуємо вертикальну швидкість

            self.velocity_pub.publish(vel_msg)
            cmd_z_vel = vel_z

            if current_z < 0.65 and dist_xy < 0.28 and not self.is_landing:
                self.get_logger().info("Висота низька + хороше вирівнювання → LAND")
                self.land()
                self.is_landing = True

        # Logging
        if int(time.time()) % 2 == 0:
            status = "ЗНИЖЕННЯ (PID Z)" if self.descent_mode else "ГОРИЗОНТАЛЬНИЙ"
            real_z = self.current_velocity.linear.z if self.current_velocity else 0.0
            self.get_logger().info(
                f'[{status}] Дист: {dist_xy:.2f}m | Вис: {current_z:.2f}m | '
                f'Real Vert: {real_z:+.2f} м/с | Cmd Vert: {cmd_z_vel:+.2f} м/с'
            )

        if dist_xy < 6.0:
            self.horizontal_errors.append(dist_xy)
        if self.current_velocity is not None:
            h_speed = math.hypot(self.current_velocity.linear.x, self.current_velocity.linear.y)
            if h_speed > 0.05:
                self.horizontal_speeds.append(h_speed)
            if self.descent_mode:
                self.vertical_speeds.append(abs(self.current_velocity.linear.z))

    def land(self):
        self.land_command_time = time.time()
        req = CommandTOL.Request(altitude=0.0)
        future = self.land_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        self.get_logger().info("Команда посадки надіслана")
        self.print_metrics()

    def print_metrics(self):
        total = time.time() - self.target_detected_time if self.target_detected_time else 0.0
        descent = (self.land_command_time - self.descent_start_time) if self.descent_start_time else 0.0
        horiz_filtered = [s for s in self.horizontal_speeds if s > 0.1]
        avg_h = sum(horiz_filtered) / len(horiz_filtered) if horiz_filtered else 0.0
        avg_v = sum(self.vertical_speeds) / len(self.vertical_speeds) if self.vertical_speeds else 0.0
        if self.descent_start_height and descent > 0:
            end_z = self.current_pose.position.z if self.current_pose else 0.0
            avg_v_distance = max(0.0, (self.descent_start_height - end_z) / descent)
        else:
            avg_v_distance = 0.0
        final_err = self.horizontal_errors[-1] if self.horizontal_errors else 0.0

        self.get_logger().info("╔══════════════════════ МЕТРИКИ ПОСАДКИ ══════════════════════╗")
        self.get_logger().info(f"Час від виявлення цілі до посадки : {total:.2f} сек")
        self.get_logger().info(f"Час зниження : {descent:.2f} сек")
        self.get_logger().info(f"Макс. горизонтальна швидкість : {self.max_horiz_speed:.2f} м/с")
        self.get_logger().info(f"Середня горизонтальна швидкість : {avg_h:.2f} м/с")
        self.get_logger().info(f"Макс. вертикальна швидкість : {self.max_vert_descent:.2f} м/с")
        self.get_logger().info(f"Середня вертикальна швидкість : {avg_v_distance:.3f} м/с")
        self.get_logger().info(f"Фінальна похибка : {final_err * 100:.1f} см")
        self.get_logger().info("╚══════════════════════════════════════════════════════════════╝")

    def wait_for_topics(self):
        self.get_logger().info("Чекаємо підключення до MAVROS...")
        while rclpy.ok() and not (self.current_state.connected and self.current_pose is not None):
            rclpy.spin_once(self, timeout_sec=0.5)
        self.get_logger().info("Підключено до MAVROS")
        return True

    def go(self):
        for srv, name in [(self.set_mode_client, "set_mode"),
                          (self.arming_client, "arming"),
                          (self.land_client, "land")]:
            while not srv.wait_for_service(timeout_sec=2.0):
                self.get_logger().warn(f"Чекаємо сервіс {name}...")
        if not self.wait_for_topics():
            return

        req = SetMode.Request(custom_mode="OFFBOARD")
        future = self.set_mode_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10)
        if future.result() and future.result().mode_sent:
            self.get_logger().info("OFFBOARD увімкнено")
        else:
            self.get_logger().error("Не вдалося увімкнути OFFBOARD")
            return

        time.sleep(1.5)

        req = CommandBool.Request(value=True)
        future = self.arming_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10)
        if future.result() and future.result().success:
            self.get_logger().info("Дрон озброєно. Чекаємо ціль...")
        else:
            self.get_logger().error("Не вдалося озброїти")

        rclpy.spin(self)

    def destroy_node(self):
        self.get_logger().info("LandingFlyer завершує роботу")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = LandingFlyer()
    try:
        node.go()
    except KeyboardInterrupt:
        node.get_logger().info("Зупинено користувачем")
    except Exception as e:
        node.get_logger().error(f"Помилка: {e}")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()