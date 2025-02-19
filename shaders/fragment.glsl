#version 330 core
in vec2 vUv;
out vec4 fragColor;

uniform float u_time;
uniform vec2 u_resolution;
uniform sampler2D u_texture;

// Новые uniform-переменные для управления камерой
uniform vec3 u_camPos;
uniform mat3 u_camRot;

#define MAX_STEPS 500
#define MAX_DIST 1000.0
#define SURFACE_DIST 0.00001

// Функция SDF для сферы
float sphereSDF(vec3 p, vec3 center, float radius) {
    return length(p - center) - radius;
}

float sceneSDF(vec3 p) {
    // Пример: анимированная сфера с вращением
    float angle = u_time;
    mat3 rotationY = mat3(
        cos(angle), 0.0, sin(angle),
        0.0,        1.0, 0.0,
        -sin(angle), 0.0, cos(angle)
    );
    vec3 rotatedP = rotationY * p;
    vec3 center = vec3(sin(u_time), 0.0, 3.0);
    return sphereSDF(rotatedP, center, 1.0);
}

float rayMarch(vec3 ro, vec3 rd) {
    float distanceTraveled = 0.0;
    for (int i = 0; i < MAX_STEPS; i++) {
        vec3 p = ro + rd * distanceTraveled;
        float distanceToScene = sceneSDF(p);
        if (distanceToScene < SURFACE_DIST) {
            return distanceTraveled;
        }
        distanceTraveled += distanceToScene;
        if (distanceTraveled >= MAX_DIST) break;
    }
    return distanceTraveled;
}

void main() {
    // Преобразуем координаты фрагмента в диапазон [-1, 1]
    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution.xy) / u_resolution.y;
    
    // Используем параметры камеры, переданные из Python
    vec3 ro = u_camPos;
    // Преобразуем координаты экрана в направление луча в мировых координатах
    vec3 rd = normalize(u_camRot * vec3(uv, 1.0));
    
    float d = rayMarch(ro, rd);
    vec3 color;
    
    if (d < MAX_DIST) {
        vec3 hitPoint = ro + rd * d;
        
        // Применяем ту же анимацию, что и в sceneSDF, для вычисления локальных координат
        float angle = u_time;
        mat3 rotationY = mat3(
            cos(angle), 0.0, sin(angle),
            0.0,        1.0, 0.0,
            -sin(angle), 0.0, cos(angle)
        );
        vec3 rotatedHit = rotationY * hitPoint;
        vec3 center = vec3(sin(u_time), 0.0, 3.0);
        vec3 localHit = rotatedHit - center;
        
        // Вычисляем сферические UV-координаты для текстурирования сферы
        float u_coord = 0.5 + atan(localHit.z, localHit.x) / (2.0 * 3.14159265);
        float v_coord = 0.5 - asin(clamp(localHit.y, -1.0, 1.0)) / 3.14159265;
        vec2 sphereUV = vec2(u_coord, v_coord);
        
        // Беру цвет из текстуры без освещения
        color = texture(u_texture, sphereUV).rgb;
    } else {
        // Фон
        color = vec3(0.0);
    }
    
    fragColor = vec4(color, 1.0);
}
