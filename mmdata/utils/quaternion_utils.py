import math
import numpy as np


def slerp(x, y, t, eps=1e-5):
    s = 1 - t
    cos = x[0] * y[0] + x[1] * y[1] + x[2] * y[2] + x[3] * y[3]
    direction = 1.0 if (cos >= 0.0) else -1.0
    sqr_sin = 1.0 - cos * cos

    if sqr_sin > eps:
        sin = math.sqrt(sqr_sin)
        length = math.atan2(sin, cos * direction)
        s = math.sin(s * length) / sin
        t = math.sin(t * length) / sin

    t_direction = t * direction
    value = np.array([
        x[0] * s + y[0] * t_direction,
        x[1] * s + y[1] * t_direction,
        x[2] * s + y[2] * t_direction,
        x[3] * s + y[3] * t_direction,
    ])

    if s == 1.0 - t:
        # normalize in case we just did a lerp
        f = 1.0 / math.sqrt(value[0] * value[0] + value[1] * value[1] + value[2] * value[2] + value[3] * value[3])
        value = value * f
    return value


def quaternion_from_rotation_matrix(m):
    trace = m[0, 0] + m[1, 1] + m[2, 2]

    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        q = np.array([
            (m[2, 1] - m[1, 2]) * s,
            (m[0, 2] - m[2, 0]) * s,
            (m[1, 0] - m[0, 1]) * s,
            0.25 / s,
        ])
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = np.array([
            0.25 * s,
            (m[0, 1] + m[1, 0]) / s,
            (m[0, 2] + m[2, 0]) / s,
            (m[2, 1] - m[1, 2]) / s,
        ])
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = np.array([
            (m[0, 1] + m[1, 0]) / s,
            0.25 * s,
            (m[1, 2] + m[2, 1]) / s,
            (m[0, 2] - m[2, 0]) / s,
        ])
    else:
        s = 2.0 * math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = np.array([
            (m[0, 2] + m[2, 0]) / s,
            (m[1, 2] + m[2, 1]) / s,
            0.25 * s,
            (m[1, 0] - m[0, 1]) / s,
        ])
    return q


def vector_apply_quaternion(v, q):
    ix = q[3] * v[0] + q[1] * v[2] - q[2] * v[1]
    iy = q[3] * v[1] + q[2] * v[0] - q[0] * v[2]
    iz = q[3] * v[2] + q[0] * v[1] - q[1] * v[0]
    iw = -q[0] * v[0] - q[1] * v[1] - q[2] * v[2]
    new_v = np.array([
        ix * q[3] + iw * -q[0] + iy * -q[2] - iz * -q[1],
        iy * q[3] + iw * -q[1] + iz * -q[0] - ix * -q[2],
        iz * q[3] + iw * -q[2] + ix * -q[1] - iy * -q[0],
    ])
    return new_v


def set_quaternion_from_axis_angle(axis, angle):
    half_angle = angle / 2
    s = math.sin(half_angle)
    q = np.array([
        axis[0] * s,
        axis[1] * s,
        axis[2] * s,
        math.cos(half_angle),
    ])
    return q


def multiply_quaternions(a, b):
    qax, qay, qaz, qaw = a[0], a[1], a[2], a[3]
    qbx, qby, qbz, qbw = b[0], b[1], b[2], b[3]

    q = np.array([
        qax * qbw + qaw * qbx + qay * qbz - qaz * qby,
        qay * qbw + qaw * qby + qaz * qbx - qax * qbz,
        qaz * qbw + qaw * qbz + qax * qby - qay * qbx,
        qaw * qbw - qax * qbx - qay * qby - qaz * qbz,
    ])
    return q


def invert_quaternion(q):
    new_q = np.array([-q[0], -q[1], -q[2], q[3]])
    return new_q


def euler_to_quaternion(e):
    c1 = math.cos(e[0] / 2)
    c2 = math.cos(e[1] / 2)
    c3 = math.cos(e[2] / 2)
    s1 = math.sin(e[0] / 2)
    s2 = math.sin(e[1] / 2)
    s3 = math.sin(e[2] / 2)
    q = np.array([
        s1 * c2 * c3 + c1 * s2 * s3,
        c1 * s2 * c3 - s1 * c2 * s3,
        c1 * c2 * s3 + s1 * s2 * c3,
        c1 * c2 * c3 - s1 * s2 * s3,
    ])
    return q


def quaternion_to_euler(q):
    # quaternion to rotation matrix
    p = np.zeros([3])
    s = np.ones([3])
    qx2, qy2, qz2 = (q[0] + q[0]), (q[1] + q[1]), (q[2] + q[2])
    qxx, qxy, qxz = (q[0] * qx2), (q[0] * qy2), (q[0] * qz2)
    qyy, qyz, qzz = (q[1] * qy2), (q[1] * qz2), (q[2] * qz2)
    qwx, qwy, qwz = (q[3] * qx2), (q[3] * qy2), (q[3] * qz2)

    rotation_matrix = np.array([
        [(1.0 - (qyy + qzz)) * s[0], (qxy - qwz) * s[1], (qxz + qwy) * s[2], p[0]],
        [(qxy + qwz) * s[0], (1.0 - (qxx + qzz)) * s[1], (qyz - qwx) * s[2], p[1]],
        [(qxz - qwy) * s[0], (qyz + qwx) * s[1], (1.0 - (qxx + qyy)) * s[2], p[2]],
        [0.0, 0.0, 0.0, 1.0],
    ])

    # rotation matrix to euler
    y = math.asin(min(max(rotation_matrix[0, 2], -1.0), 1.0))
    if abs(rotation_matrix[0, 2]) < 0.9999999:
        x = math.atan2(-rotation_matrix[1, 2], rotation_matrix[2, 2])
        z = math.atan2(-rotation_matrix[0, 1], rotation_matrix[0, 0])
    else:
        x = math.atan2(-rotation_matrix[2, 1], rotation_matrix[1, 1])
        z = 0.0
    return np.array([x, y, z])
