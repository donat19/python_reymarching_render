import time
import numpy as np
import moderngl
import moderngl_window
from moderngl_window import geometry

class RayMarching(moderngl_window.WindowConfig):
    title = "Ray Marching with ModernGL (Camera Control)"
    fullscreen = True  # Запуск в полноэкранном режиме
    resource_dir = '.'  # Папка с ресурсами (шейдерами и текстурами)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Загружаем шейдеры
        with open(self.resource_dir + '/shaders/vertex.glsl', 'r', encoding='utf-8') as f:
            vertex_shader_source = f.read()
        with open(self.resource_dir + '/shaders/fragment.glsl', 'r', encoding='utf-8') as f:
            fragment_shader_source = f.read()

        # Компилируем шейдерную программу
        self.program = self.ctx.program(
            vertex_shader=vertex_shader_source,
            fragment_shader=fragment_shader_source
        )

        # Загружаем текстуру и привязываем uniform-сэмплер
        self.texture = self.load_texture_2d(self.resource_dir + '/textures/texture.png')
        self.program["u_texture"].value = 0

        # Создаем полноэкранный прямоугольник
        self.quad = geometry.quad_fs()
        self.start_time = time.time()

        # Инициализирую параметры камеры:
        # Начальное положение камеры
        self.camera_pos = np.array([0.0, 0.0, -7.0], dtype='f4')
        # Изначально камера направлена вдоль оси Z, поэтому матрица вращения – единичная
        self.camera_rot = np.eye(3, dtype='f4')

    def on_render(self, current_time: float, frame_time: float):
        # Обновляем время
        self.program["u_time"].value = time.time() - self.start_time
        width, height = self.wnd.size
        self.program["u_resolution"].value = (width, height)

        # Передаю параметры камеры в шейдер
        self.program["u_camPos"].value = tuple(self.camera_pos)
        # Передаю матрицу вращения камеры (3x3) как 9 последовательных чисел
        self.program["u_camRot"].write(self.camera_rot.astype('f4').tobytes())

        # Активирую текстуру, очищаю экран и рендерю прямоугольник
        self.texture.use(location=0)
        self.ctx.clear(0.0, 0.0, 0.0)
        self.quad.render(self.program)

def key_event(self, key, action, modifiers):
    move_speed = 10.0

    # Обрабатываем только события нажатия или повторного нажатия
    if action not in (self.wnd.keys.ACTION_PRESS, self.wnd.keys.ACTION_REPEAT):
        return

    if key == self.wnd.keys.ESCAPE:
        self.wnd.close()
        return

    # Словарь для смещений камеры вдоль локальных и глобальных осей
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
            # Смещение вдоль локальной оси: применяем поворот камеры
            self.camera_pos += self.camera_rot @ delta
        else:
            # Смещение в мировых координатах
            self.camera_pos += delta
        return

    # Обработка поворотов камеры
    if key in (self.wnd.keys.LEFT, self.wnd.keys.RIGHT):
        angle = np.radians(5) if key == self.wnd.keys.LEFT else -np.radians(5)
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        # Матрица поворота вокруг оси Y
        rot = np.array([
            [cos_a,  0.0, -sin_a],
            [0.0,    1.0,  0.0],
            [sin_a,  0.0,  cos_a]
        ], dtype='f4')
        self.camera_rot = rot @ self.camera_rot


if __name__ == '__main__':
    moderngl_window.run_window_config(RayMarching)
