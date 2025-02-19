import time
import numpy as np
import moderngl
import moderngl_window
from moderngl_window import geometry

# Функция вычисления матрицы вида (look-at) для камеры
def look_at(cam_pos, target, up=np.array([0.0, 1.0, 0.0], dtype='f4')):
    forward = target - cam_pos
    forward /= np.linalg.norm(forward)
    right = np.cross(up, forward)
    right /= np.linalg.norm(right)
    true_up = np.cross(forward, right)
    return np.array([right, true_up, forward], dtype='f4').T

# Простая реализация фильтра Калмана для 3D-положения
class KalmanFilter3D:
    def __init__(self, dt, process_noise, measurement_noise):
        self.dt = dt
        self.x = np.zeros((6, 1), dtype='f4')  # [p_x, p_y, p_z, v_x, v_y, v_z]
        self.P = np.eye(6, dtype='f4')
        self.A = np.eye(6, dtype='f4')
        for i in range(3):
            self.A[i, i+3] = dt
        self.H = np.zeros((3, 6), dtype='f4')
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = 1.0
        self.Q = process_noise * np.eye(6, dtype='f4')
        self.R = measurement_noise * np.eye(3, dtype='f4')
    
    def predict(self):
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q
    
    def update(self, z):
        z = z.reshape((3, 1))
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6, dtype='f4') - K @ self.H) @ self.P
        return self.x[:3].flatten()

class RayMarching(moderngl_window.WindowConfig):
    title = "Ray Marching (Orbit Camera with Kalman & Spring-Damper)"
    fullscreen = True
    resource_dir = '.'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Загрузка шейдеров
        with open(self.resource_dir + '/shaders/vertex.glsl', 'r', encoding='utf-8') as f:
            vertex_shader_source = f.read()
        with open(self.resource_dir + '/shaders/fragment.glsl', 'r', encoding='utf-8') as f:
            fragment_shader_source = f.read()
        
        self.program = self.ctx.program(
            vertex_shader=vertex_shader_source,
            fragment_shader=fragment_shader_source
        )
        self.texture = self.load_texture_2d(self.resource_dir + '/textures/texture.png')
        self.program["u_texture"].value = 0
        self.quad = geometry.quad_fs()
        self.start_time = time.time()

        # Целевая позиция (объект, за которым отслеживаем) — здесь симулируем движение
        self.camera_target = np.array([0.0, 0.0, 0.0], dtype='f4')
        self.kf = KalmanFilter3D(dt=0.016, process_noise=0.01, measurement_noise=0.1)

        # Параметры орбитальной камеры
        self.distance = 7.0   # расстояние от камеры до цели
        self.yaw = 0.0        # горизонтальный угол (в радианах)
        self.pitch = 0.0      # вертикальный угол (в радианах)

        # Фактическое положение камеры и параметры демпфирования
        self.camera_pos = np.array([0.0, 0.0, -7.0], dtype='f4')
        self.camera_velocity = np.zeros(3, dtype='f4')
        self.spring_stiffness = 10.0
        self.spring_damping = 2.0

    def smooth_position(self, current, target, smoothing, dt):
        """Apply exponential smoothing to position"""
        alpha = 1.0 - np.exp(-smoothing * dt)
        return current + (target - current) * alpha

    def check_camera_collision(self, position):
        """Simple sphere collision check"""
        # Example: Check collision with sphere at origin
        sphere_center = np.array([0.0, 0.0, 0.0])
        sphere_radius = 1.5
        
        dist_to_center = np.linalg.norm(position - sphere_center)
        if dist_to_center < sphere_radius:
            # Push camera out of collision
            dir_from_center = (position - sphere_center) / dist_to_center
            return sphere_center + dir_from_center * sphere_radius
        return position

    def update_camera_rotation(self, dt):
        # Apply friction to rotation velocity
        self.yaw_velocity *= np.exp(-self.rotation_friction * dt)
        self.pitch_velocity *= np.exp(-self.rotation_friction * dt)
        
        # Update angles with velocity
        self.orbit_yaw += self.yaw_velocity * dt
        self.orbit_pitch += self.pitch_velocity * dt

    def add_camera_shake(self, intensity):
        """Add trauma-based camera shake"""
        self.shake_trauma = min(1.0, self.shake_trauma + intensity)

    def update_camera_shake(self, dt):
        """Update camera shake effect"""
        if self.shake_trauma > 0:
            self.shake_trauma = max(0.0, self.shake_trauma - self.shake_decay * dt)
            shake_power = self.shake_trauma * self.shake_trauma  # Quadratic falloff
            self.shake_offset = np.random.normal(0, 0.1, 3) * shake_power
        else:
            self.shake_offset.fill(0)

    def update_fov(self, dt):
        """Smooth FOV transitions"""
        self.current_fov = self.smooth_position(
            self.current_fov,
            self.target_fov,
            self.fov_smooth_speed,
            dt
        )
        return np.radians(self.current_fov)

    def update_camera_constraints(self):
        """Apply additional camera constraints"""
        # Constrain camera height
        min_height = 0.5
        if self.camera_pos[1] < min_height:
            self.camera_pos[1] = min_height
            self.camera_velocity[1] = max(0, self.camera_velocity[1])

        # Add velocity dampening when near ground
        ground_influence = 1.0 - min(1.0, (self.camera_pos[1] - min_height) / 2.0)
        self.camera_velocity *= (1.0 - ground_influence * 0.1)

    def on_render(self, current_time: float, frame_time: float):
        elapsed = time.time() - self.start_time
        self.program["u_time"].value = elapsed
        width, height = self.wnd.size
        self.program["u_resolution"].value = (width, height)

        # Симуляция движения цели (например, по синусоиде)
        sim_target = np.array([np.sin(elapsed), 0.0, np.cos(elapsed)], dtype='f4')
        self.camera_target = sim_target
        self.kf.predict()
        filtered_target = self.kf.update(self.camera_target)

        # Вычисляем желаемое положение камеры для орбитальной модели
        # offset вычисляется на основе текущих углов (yaw, pitch)
        offset = np.array([
            np.cos(self.pitch) * np.sin(self.yaw),
            np.sin(self.pitch),
            np.cos(self.pitch) * np.cos(self.yaw)
        ], dtype='f4')
        desired_pos = filtered_target - self.distance * offset

        # Пружинная модель для плавного перемещения камеры
        dt = frame_time
        acceleration = self.spring_stiffness * (desired_pos - self.camera_pos) - self.spring_damping * self.camera_velocity
        self.camera_velocity += acceleration * dt
        self.camera_pos += self.camera_velocity * dt

        # Вычисляем матрицу ориентации камеры (u_camRot), чтобы в шейдере:
        # vec3 rd = normalize(u_camRot * vec3(uv, 1.0));
        # получалось направление от камеры к цели.
        world_up = np.array([0.0, 1.0, 0.0], dtype='f4')
        forward = filtered_target - self.camera_pos
        forward /= np.linalg.norm(forward)
        # Чтобы экранное движение вправо соответствовало правому направлению,
        # вычисляем right как cross(world_up, forward)
        right = np.cross(world_up, forward)
        right /= np.linalg.norm(right)
        up = np.cross(forward, right)
        # Собираем матрицу, столбцы: right, up, forward
        cam_rot = np.column_stack((right, up, forward))
        
        # Передаем положение и матрицу в шейдер
        self.program["u_camPos"].value = tuple(self.camera_pos)
        self.program["u_camRot"].write(cam_rot.astype('f4').tobytes())

        self.texture.use(location=0)
        self.ctx.clear(0.0, 0.0, 0.0)
        self.quad.render(self.program)

    def key_event(self, key, action, modifiers):
        if action not in (self.wnd.keys.ACTION_PRESS, self.wnd.keys.ACTION_REPEAT):
            return
        if key == self.wnd.keys.ESCAPE:
            self.wnd.close()

    def mouse_drag_event(self, x, y, dx, dy, buttons):
        sensitivity = 0.005  # настройте чувствительность по необходимости
        self.yaw   += dx * sensitivity
        self.pitch += dy * sensitivity
        # Ограничиваем pitch, чтобы камера не перевернулась
        max_pitch = np.radians(89.0)
        self.pitch = np.clip(self.pitch, -max_pitch, max_pitch)

    def mouse_scroll_event(self, x_offset: float, y_offset: float):
        """Handle mouse scroll for camera zoom"""
        zoom_speed = 0.5
        self.orbit_distance -= y_offset * zoom_speed
        # Clamp the distance between min and max values
        self.orbit_distance = np.clip(self.orbit_distance, 
                                    self.min_orbit_distance, 
                                    self.max_orbit_distance)

if __name__ == '__main__':
    moderngl_window.run_window_config(RayMarching)
