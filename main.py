import time
import numpy as np
import moderngl
import moderngl_window
from moderngl_window import geometry

# Реализация простого фильтра Калмана для 3D-положения
class KalmanFilter3D:
    def __init__(self, dt, process_noise, measurement_noise):
        self.dt = dt
        self.x = np.zeros((6, 1), dtype='f4')  # состояние: [p, v]
        self.P = np.eye(6, dtype='f4')
        self.A = np.eye(6, dtype='f4')
        for i in range(3):
            self.A[i, i + 3] = dt
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
    title = "Ray Marching (Camera Tracking with Kalman & Spring-Damper)"
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

        # Начальное положение камеры и её ориентация
        self.camera_pos = np.array([0.0, 0.0, -7.0], dtype='f4')
        self.camera_rot = np.eye(3, dtype='f4')

        # Целевая позиция для отслеживания (обновляемая каждый кадр)
        self.camera_target = self.camera_pos.copy()
        self.camera_velocity = np.zeros(3, dtype='f4')
        self.spring_stiffness = 5.0
        self.spring_damping = 1.0

        # Инициализация фильтра Калмана (dt ≈ 16 мс)
        self.kf = KalmanFilter3D(dt=0.016, process_noise=0.01, measurement_noise=0.1)

    def on_render(self, current_time: float, frame_time: float):
        elapsed = time.time() - self.start_time
        self.program["u_time"].value = elapsed
        width, height = self.wnd.size
        self.program["u_resolution"].value = (width, height)

        # Симуляция движения объекта: обновляем camera_target динамически
        # Здесь, например, объект движется по синусоиде по оси X и косинусоиде по оси Z
        self.camera_target = np.array([np.sin(elapsed), 0.0, -7.0 + np.cos(elapsed)], dtype='f4')

        # Обновляем фильтр Калмана и получаем сглаженное положение цели
        self.kf.predict()
        filtered_target = self.kf.update(self.camera_target)

        # Система пружинного демпфирования для плавного перемещения камеры
        dt = frame_time
        acceleration = self.spring_stiffness * (filtered_target - self.camera_pos) - self.spring_damping * self.camera_velocity
        self.camera_velocity += acceleration * dt
        self.camera_pos += self.camera_velocity * dt

        # Передача параметров камеры в шейдер
        self.program["u_camPos"].value = tuple(self.camera_pos)
        self.program["u_camRot"].write(self.camera_rot.astype('f4').tobytes())

        self.texture.use(location=0)
        self.ctx.clear(0.0, 0.0, 0.0)
        self.quad.render(self.program)

    def key_event(self, key, action, modifiers):
        move_speed = 10.0
        if action not in (self.wnd.keys.ACTION_PRESS, self.wnd.keys.ACTION_REPEAT):
            return
        if key == self.wnd.keys.ESCAPE:
            self.wnd.close()
            return

        movements = {
            self.wnd.keys.W: (np.array([0.0, 0.0, move_speed], dtype='f4'), True),
            self.wnd.keys.S: (np.array([0.0, 0.0, -move_speed], dtype='f4'), True),
            self.wnd.keys.A: (np.array([-move_speed, 0.0, 0.0], dtype='f4'), True),
            self.wnd.keys.D: (np.array([move_speed, 0.0, 0.0], dtype='f4'), True),
            self.wnd.keys.Q: (np.array([0.0, move_speed, 0.0], dtype='f4'), False),
            self.wnd.keys.E: (np.array([0.0, -move_speed, 0.0], dtype='f4'), False),
        }
        if key in movements:
            delta, use_local = movements[key]
            if use_local:
                self.camera_target += self.camera_rot @ delta
            else:
                self.camera_target += delta
            return

        if key in (self.wnd.keys.LEFT, self.wnd.keys.RIGHT):
            angle = np.radians(5) if key == self.wnd.keys.LEFT else -np.radians(5)
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            rot = np.array([
                [cos_a,  0.0, -sin_a],
                [0.0,    1.0,  0.0],
                [sin_a,  0.0,  cos_a]
            ], dtype='f4')
            self.camera_rot = rot @ self.camera_rot

if __name__ == '__main__':
    moderngl_window.run_window_config(RayMarching)
