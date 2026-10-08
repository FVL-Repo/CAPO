from __future__ import annotations

import numpy as np


def quat_wxyz_to_matrix(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def tcp_pose_to_matrix(tcp_pose: np.ndarray) -> np.ndarray:
    x, y, z, qw, qx, qy, qz = np.asarray(tcp_pose, dtype=np.float64)
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = quat_wxyz_to_matrix(qw, qx, qy, qz)
    mat[:3, 3] = [x, y, z]
    return mat


def euler_delta_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def orthogonalize(rot: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(rot)
    det = np.linalg.det(u @ vt)
    return u @ np.diag([1.0, 1.0, det]) @ vt


def matrix_to_quat_wxyz(rot: np.ndarray) -> np.ndarray:
    trace = rot[0, 0] + rot[1, 1] + rot[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (rot[2, 1] - rot[1, 2]) * s
        qy = (rot[0, 2] - rot[2, 0]) * s
        qz = (rot[1, 0] - rot[0, 1]) * s
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = 2.0 * np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2])
        qw = (rot[2, 1] - rot[1, 2]) / s
        qx = 0.25 * s
        qy = (rot[0, 1] + rot[1, 0]) / s
        qz = (rot[0, 2] + rot[2, 0]) / s
    elif rot[1, 1] > rot[2, 2]:
        s = 2.0 * np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2])
        qw = (rot[0, 2] - rot[2, 0]) / s
        qx = (rot[0, 1] + rot[1, 0]) / s
        qy = 0.25 * s
        qz = (rot[1, 2] + rot[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1])
        qw = (rot[1, 0] - rot[0, 1]) / s
        qx = (rot[0, 2] + rot[2, 0]) / s
        qy = (rot[1, 2] + rot[2, 1]) / s
        qz = 0.25 * s
    quat = np.array([qw, qx, qy, qz], dtype=np.float64)
    quat /= np.linalg.norm(quat) + 1e-12
    if quat[0] < 0:
        quat = -quat
    return quat


def quat_wxyz_to_rot6d(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    rot = quat_wxyz_to_matrix(qw, qx, qy, qz)
    return np.array([rot[0, 0], rot[1, 0], rot[2, 0], rot[0, 1], rot[1, 1], rot[2, 1]], dtype=np.float32)


def rot6d_to_quat_wxyz(r6d: np.ndarray) -> np.ndarray:
    c1 = np.asarray(r6d[:3], dtype=np.float64)
    c2 = np.asarray(r6d[3:6], dtype=np.float64)
    c1 /= np.linalg.norm(c1) + 1e-12
    c2 -= np.dot(c2, c1) * c1
    c2 /= np.linalg.norm(c2) + 1e-12
    c3 = np.cross(c1, c2)
    return matrix_to_quat_wxyz(np.column_stack([c1, c2, c3]))


def clip_norm(value: np.ndarray, max_norm: float | None) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if max_norm is None or max_norm <= 0:
        return arr
    norm = float(np.linalg.norm(arr))
    if norm <= max_norm or norm < 1e-12:
        return arr
    return arr * (max_norm / norm)
