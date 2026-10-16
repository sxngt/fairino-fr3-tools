"""Active eye-to-hand calibration with a gripper-mounted ArUco board (NumPy only).

A fixed camera ``C`` looks at the robot. A small ArUco board ``G`` is taped to the gripper body.
For a flange pose ``T_BF`` (forward kinematics) the board corners project to

    u = π(K, dist, X⁻¹ · T_BF · Y · p_G)

with the unknowns ``X = T_BC`` (camera in base) and ``Y = T_FG`` (board on flange). Both are
estimated together by Levenberg-Marquardt on the corner reprojection. A kinematic error moves all
corners of one pose together, so each pose is whitened with ``σ²I + J_f Σ_FK J_fᵀ``; without it the
estimate is overconfident.

What a user needs is usually not ``X`` itself but where camera measurements land in the base frame
at the places the robot works. So candidate poses are scored by how much they would shrink the
covariance of a set of **task points** (default: a small grid on the table) mapped through ``X``,
per second of motion. The robot goes to the best pose, observes, re-solves and repeats until the
points are known well enough or the time budget runs out.

Initialisation needs no guess: three or more sightings with some rotation between them give a
closed-form ``AX = XB`` solution (:func:`init_from_sightings`). A rough camera pose, if known, can
be given as a weak prior instead.

그리퍼 몸체의 ArUco 보드로 고정 카메라 외부 파라미터 X = T_BC 와 보드 장착 Y = T_FG 를 재투영
오차로 함께 푼다. 기구학 오차는 한 자세의 모든 코너를 함께 움직이므로 자세별로 백색화한다. 후보
자세는 "작업 점들(기본: 테이블 위 격자)의 베이스 기준 공분산이 이동 1초당 얼마나 줄어드는가"로
고른다. 초기값은 회전이 다른 세 번 이상의 관측으로 AX = XB 닫힌 해를 구해 얻는다(추측 불필요).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import numpy as np


# ----------------------------------------------------------------------------- SE(3) helpers
def skew(v) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def rotvec_to_matrix(r) -> np.ndarray:
    """Rodrigues: rotation vector -> matrix. / 회전벡터 → 회전행렬."""
    r = np.asarray(r, dtype=float)
    a = float(np.linalg.norm(r))
    if a < 1e-12:
        return np.eye(3) + skew(r)
    k = skew(r / a)
    return np.eye(3) + math.sin(a) * k + (1.0 - math.cos(a)) * k @ k


def matrix_to_rotvec(m) -> np.ndarray:
    """Inverse Rodrigues. / 회전행렬 → 회전벡터."""
    m = np.asarray(m, dtype=float)
    c = max(-1.0, min(1.0, (np.trace(m) - 1.0) / 2.0))
    a = math.acos(c)
    w = np.array([m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1]])
    if a < 1e-9:
        return w / 2.0
    if math.pi - a < 1e-6:
        # Near π: take the axis from the symmetric part. / π 근처: 대칭부에서 축.
        b = (m + np.eye(3)) / 2.0
        axis = np.sqrt(np.clip(np.diag(b), 0.0, None))
        i = int(np.argmax(axis))
        axis = b[:, i] / max(axis[i], 1e-12)
        return a * axis / np.linalg.norm(axis)
    return a * w / (2.0 * math.sin(a))


def perturb(t: np.ndarray, d) -> np.ndarray:
    """Right perturbation ``T · [R(d_r), d_t]``. / 오른쪽 섭동."""
    e = np.eye(4)
    e[:3, :3] = rotvec_to_matrix(d[:3])
    e[:3, 3] = d[3:6]
    return t @ e


def pose_error(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Tangent difference ``[rotvec(R_aᵀR_b), R_aᵀ(t_b − t_a)]``. / 두 포즈의 차이(접공간)."""
    d = np.linalg.inv(a) @ b
    return np.concatenate([matrix_to_rotvec(d[:3, :3]), d[:3, 3]])


def average_poses(poses, weights=None) -> np.ndarray:
    """Weighted mean of rigid poses (chordal rotation mean). / 강체 포즈 가중 평균."""
    poses = [np.asarray(p, dtype=float) for p in poses]
    w = np.ones(len(poses)) if weights is None else np.asarray(weights, dtype=float)
    w = w / w.sum()
    m = sum(wi * p[:3, :3] for wi, p in zip(w, poses, strict=True))
    u, _, vt = np.linalg.svd(m)
    out = np.eye(4)
    out[:3, :3] = u @ np.diag([1.0, 1.0, float(np.linalg.det(u @ vt))]) @ vt
    out[:3, 3] = sum(wi * p[:3, 3] for wi, p in zip(w, poses, strict=True))
    return out


def project(k: np.ndarray, dist, t_cam_obj: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Project object points with OpenCV's distortion model (k1 k2 p1 p2 k3), NumPy only.

    OpenCV 왜곡 모델 투영(NumPy). 후보 수백 개를 빠르게 평가하려고 OpenCV 를 쓰지 않는다.
    """
    p = np.asarray(pts, dtype=float) @ t_cam_obj[:3, :3].T + t_cam_obj[:3, 3]
    z = p[:, 2:3]
    xy = p[:, :2] / np.where(np.abs(z) < 1e-9, 1e-9, z)
    d = np.zeros(5)
    d[: min(5, len(dist))] = np.asarray(dist, dtype=float)[:5]
    k1, k2, p1, p2, k3 = d
    x, y = xy[:, 0], xy[:, 1]
    r2 = x * x + y * y
    radial = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return np.stack([k[0, 0] * xd + k[0, 1] * yd + k[0, 2], k[1, 1] * yd + k[1, 2]], axis=1)


def undistort_points(k: np.ndarray, dist, uv: np.ndarray, iterations: int = 20) -> np.ndarray:
    """Normalised image coordinates of distorted pixels (fixed-point iteration). / 왜곡 제거 좌표."""
    d = np.zeros(5)
    d[: min(5, len(dist))] = np.asarray(dist, dtype=float)[:5]
    k1, k2, p1, p2, k3 = d
    uv = np.asarray(uv, dtype=float)
    yd = (uv[:, 1] - k[1, 2]) / k[1, 1]
    xd = (uv[:, 0] - k[0, 2] - k[0, 1] * yd) / k[0, 0]
    x, y = xd.copy(), yd.copy()
    for _ in range(iterations):
        r2 = x * x + y * y
        radial = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
        x = (xd - (2 * p1 * x * y + p2 * (r2 + 2 * x * x))) / radial
        y = (yd - (p1 * (r2 + 2 * y * y) + 2 * p2 * x * y)) / radial
    return np.stack([x, y], axis=1)


def planar_pose(k: np.ndarray, dist, obj: np.ndarray, uv: np.ndarray, refine: int = 10) -> np.ndarray:
    """Pose of a planar target (object z = 0) from its image points, NumPy only.

    Homography from undistorted points, decomposed into ``[R | t]``, then Gauss-Newton on the
    reprojection error.
    평면 표적(z = 0)의 포즈: 호모그래피 분해 후 재투영 오차로 가우스-뉴턴 보정.
    """
    obj = np.asarray(obj, dtype=float)
    uv = np.asarray(uv, dtype=float)
    xy = undistort_points(k, dist, uv)
    rows = []
    for (x_obj, y_obj, _), (x, y) in zip(obj, xy, strict=True):
        rows.append([x_obj, y_obj, 1, 0, 0, 0, -x * x_obj, -x * y_obj, -x])
        rows.append([0, 0, 0, x_obj, y_obj, 1, -y * x_obj, -y * y_obj, -y])
    h = np.linalg.svd(np.asarray(rows))[2][-1].reshape(3, 3)
    scale = 1.0 / np.linalg.norm(h[:, 0])
    if h[2, 2] * scale < 0:
        scale = -scale
    r1, r2, t = h[:, 0] * scale, h[:, 1] * scale, h[:, 2] * scale
    u, _, vt = np.linalg.svd(np.column_stack([r1, r2, np.cross(r1, r2)]))
    pose = np.eye(4)
    pose[:3, :3] = u @ np.diag([1.0, 1.0, float(np.linalg.det(u @ vt))]) @ vt
    pose[:3, 3] = t
    for _ in range(refine):
        r0 = (project(k, dist, pose, obj) - uv).ravel()
        jac = np.zeros((r0.size, 6))
        for i in range(6):
            d = np.zeros(6)
            d[i] = 1e-7
            jac[:, i] = ((project(k, dist, perturb(pose, d), obj) - uv).ravel() - r0) / 1e-7
        step = np.linalg.lstsq(jac, -r0, rcond=None)[0]
        pose = perturb(pose, step)
        if np.abs(step).max() < 1e-10:
            break
    return pose


def solve_ax_xb(a_list, b_list) -> np.ndarray:
    """Least-squares ``X`` with ``A_i X = X B_i`` (Park-Martin rotation, then translation).

    ``A_i X = X B_i`` 의 최소제곱 해(회전: Park-Martin, 이어서 병진).
    """
    alphas = [matrix_to_rotvec(a[:3, :3]) for a in a_list]
    betas = [matrix_to_rotvec(b[:3, :3]) for b in b_list]
    m = sum(np.outer(b, a) for a, b in zip(alphas, betas, strict=True))
    u, _, vt = np.linalg.svd(m)
    rot = vt.T @ np.diag([1.0, 1.0, float(np.linalg.det(vt.T @ u.T))]) @ u.T
    lhs = np.vstack([a[:3, :3] - np.eye(3) for a in a_list])
    rhs = np.concatenate([rot @ b[:3, 3] - a[:3, 3] for a, b in zip(a_list, b_list, strict=True)])
    out = np.eye(4)
    out[:3, :3] = rot
    out[:3, 3] = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    return out


def rotation_spread_deg(poses) -> float:
    """Largest relative rotation between any two poses (deg). / 두 포즈 사이 최대 상대 회전."""
    best = 0.0
    for i in range(len(poses)):
        for j in range(i):
            d = poses[i][:3, :3].T @ poses[j][:3, :3]
            best = max(best, math.degrees(np.linalg.norm(matrix_to_rotvec(d))))
    return best


def init_from_sightings(flanges, boards_in_camera) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form ``(X, Y)`` from ≥ 3 sightings with rotation between them. / 닫힌 해 초기값.

    ``T_BF_i Y = X T_CG_i`` gives ``A X = X B`` with ``A = T_BF_j T_BF_i⁻¹`` and
    ``B = T_CG_j T_CG_i⁻¹`` for every pair.
    """
    a_list, b_list = [], []
    for i in range(len(flanges)):
        for j in range(i + 1, len(flanges)):
            a_list.append(flanges[j] @ np.linalg.inv(flanges[i]))
            b_list.append(boards_in_camera[j] @ np.linalg.inv(boards_in_camera[i]))
    x = solve_ax_xb(a_list, b_list)
    y = average_poses([np.linalg.inv(f) @ x @ g for f, g in zip(flanges, boards_in_camera, strict=True)])
    return x, y


# ----------------------------------------------------------------------------- the gripper board
@dataclass(frozen=True)
class GripperBoard:
    """Marker corners of the gripper board in its own frame. / 그리퍼 보드의 마커 모서리.

    Frame: origin at the board centre, z = board normal (markers face), corners in OpenCV order
    (top-left, top-right, bottom-right, bottom-left).
    """

    dictionary: str
    size: float
    centers: dict  # id -> (x, y)

    @classmethod
    def default(cls) -> GripperBoard:
        """2×2 DICT_4X4_50 markers 10-13, 25 mm, 30 mm apart (fits the AG-95 body side).

        2×2 마커(10–13, 25 mm, 간격 30 mm). AG-95 몸체 옆면에 붙는 크기.
        """
        c = 0.015
        return cls("DICT_4X4_50", 0.025, {10: (-c, c), 11: (c, c), 12: (-c, -c), 13: (c, -c)})

    @classmethod
    def from_json(cls, path: str) -> GripperBoard:
        with open(path, encoding="utf-8") as f:
            m = json.load(f)
        return cls(m["dict"], float(m["marker_size_m"]), {int(k): tuple(v) for k, v in m["markers"].items()})

    def to_json(self, path: str) -> None:
        data = {
            "dict": self.dictionary,
            "marker_size_m": self.size,
            "markers": {str(k): list(v) for k, v in self.centers.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @property
    def ids(self) -> tuple[int, ...]:
        return tuple(sorted(self.centers))

    def corners(self, marker_id: int) -> np.ndarray:
        cx, cy = self.centers[marker_id]
        h = self.size / 2.0
        return np.array([[cx - h, cy + h, 0], [cx + h, cy + h, 0], [cx + h, cy - h, 0], [cx - h, cy - h, 0]])

    def object_points(self, ids=None) -> np.ndarray:
        ids = self.ids if ids is None else ids
        return np.concatenate([self.corners(int(i)) for i in ids])


# ----------------------------------------------------------------------------- the estimator
@dataclass
class Observation:
    flange_in_base: np.ndarray  # T_BF from forward kinematics / 정기구학 플랜지 포즈
    object_points: np.ndarray  # (n, 3) board-frame corners / 보드 좌표 코너
    image_points: np.ndarray  # (n, 2) detected corners (px) / 검출 코너
    tag: str = ""


@dataclass
class Prediction:
    """What the camera should see at a candidate pose. / 후보 자세에서 보일 것의 예측."""

    image_points: np.ndarray
    facing_cos: float  # board normal vs. ray to the camera / 보드 법선과 시선의 코사인
    marker_px: float  # smallest projected marker edge (px) / 가장 작은 마커 변
    visible: bool


@dataclass
class ActiveHandEye:
    """Joint estimate of camera-in-base ``X`` and board-on-flange ``Y`` with active pose choice.

    카메라-베이스 X 와 보드-플랜지 Y 의 공동 추정 + 능동 자세 선택.

    Parameters
    ----------
    camera_matrix, distortion
        Intrinsics of the fixed camera.
    board
        Gripper board geometry.
    camera_in_base_prior
        Optional rough ``X`` (weak prior). Without it, initialise with :meth:`initialise`.
    prior_sigma
        (rotation rad, translation m) standard deviation of the prior.
    pixel_sigma
        Corner noise floor (px); the working value is re-estimated from the residuals.
    fk_sigma
        Per-pose kinematic error (rad, m).
    """

    camera_matrix: np.ndarray
    distortion: np.ndarray
    board: GripperBoard
    camera_in_base_prior: np.ndarray | None = None
    prior_sigma: tuple[float, float] = (math.radians(10.0), 0.05)
    pixel_sigma: float = 0.35
    fk_sigma: tuple[float, float] = (math.radians(0.03), 0.0003)
    image_size: tuple[int, int] = (1920, 1080)
    min_facing_cos: float = math.cos(math.radians(70.0))
    min_marker_px: float = 18.0
    margin_px: float = 40.0
    observations: list = field(default_factory=list)

    def __post_init__(self):
        self.camera_matrix = np.asarray(self.camera_matrix, dtype=float)
        self.distortion = np.asarray(self.distortion, dtype=float)
        prior = self.camera_in_base_prior
        self.X: np.ndarray | None = None if prior is None else np.asarray(prior, dtype=float).copy()
        self.Y: np.ndarray | None = None
        self._mount_prior: np.ndarray | None = None
        self.sigma = self.pixel_sigma
        self._prior_w = np.array([1 / self.prior_sigma[0]] * 3 + [1 / self.prior_sigma[1]] * 3)
        self._fk_var0 = np.array([self.fk_sigma[0] ** 2] * 3 + [self.fk_sigma[1] ** 2] * 3)
        self._fk_var = self._fk_var0.copy()
        self._mount_w = np.array([1 / math.radians(90.0)] * 3 + [1 / 0.5] * 3)  # keeps Y regular / Y 정칙화

    # ---- state / 상태
    @property
    def ready(self) -> bool:
        return self.X is not None and self.Y is not None

    def initialise(self, x: np.ndarray, y: np.ndarray) -> None:
        """Set the starting ``X`` and ``Y`` (e.g. from :func:`init_from_sightings`). / 초기값 설정."""
        self.X = np.asarray(x, dtype=float).copy()
        self.Y = np.asarray(y, dtype=float).copy()
        self._mount_prior = self.Y.copy()

    def initialise_mount(self, flange_in_base: np.ndarray, board_in_camera: np.ndarray) -> None:
        """First sighting with a camera prior: ``Y = T_BF⁻¹ · X · T_CG``. / 사전값으로 Y 초기화."""
        if self.X is None:
            raise ValueError("no camera prior: use initialise() with init_from_sightings()")
        self.initialise(self.X, np.linalg.inv(flange_in_base) @ self.X @ board_in_camera)

    def board_in_camera(self, flange_in_base: np.ndarray, x=None, y=None) -> np.ndarray:
        x = self.X if x is None else x
        y = self.Y if y is None else y
        return np.linalg.inv(x) @ flange_in_base @ y

    def add(self, flange_in_base, ids, image_points, tag: str = "") -> None:
        """Add one (averaged) detection: marker ids and (n·4, 2) corners. / 관측 하나 추가."""
        self.observations.append(
            Observation(
                np.asarray(flange_in_base, dtype=float),
                self.board.object_points(ids),
                np.asarray(image_points, dtype=float).reshape(-1, 2),
                tag,
            )
        )

    # ---- residuals and Jacobians (12 local parameters: δX, δY) / 잔차와 야코비안
    def _data_residual(self, x, y, obs=None) -> np.ndarray:
        obs = self.observations if obs is None else obs
        if not obs:
            return np.zeros(0)
        out = []
        for o in obs:
            uv = project(
                self.camera_matrix,
                self.distortion,
                self.board_in_camera(o.flange_in_base, x, y),
                o.object_points,
            )
            out.append((uv - o.image_points).ravel())
        return np.concatenate(out)

    def _whitener(self, o: Observation) -> np.ndarray:
        """``W`` with ``WᵀW = (σ²I + J_f Σ_f J_fᵀ)⁻¹`` for one pose. / 자세 하나의 백색화 행렬.

        A kinematic error moves all corners of a pose together, so they are not independent.
        기구학 오차는 한 자세의 모든 코너를 함께 움직이므로 코너 잡음은 독립이 아니다.
        """
        k, dist = self.camera_matrix, self.distortion
        base = project(k, dist, self.board_in_camera(o.flange_in_base), o.object_points).ravel()
        jac = np.zeros((base.size, 6))
        x_inv = np.linalg.inv(self.X)
        for i in range(6):
            d = np.zeros(6)
            d[i] = 1e-6
            moved = x_inv @ perturb(o.flange_in_base, d) @ self.Y
            jac[:, i] = (project(k, dist, moved, o.object_points).ravel() - base) / 1e-6
        cov = (self.sigma**2) * np.eye(base.size) + jac @ np.diag(self._fk_var) @ jac.T
        return np.linalg.cholesky(np.linalg.inv(cov)).T

    def _prior_residual(self, x, y) -> np.ndarray:
        r = [self._mount_w * pose_error(self._mount_prior, y)]
        if self.camera_in_base_prior is not None:
            r.append(self._prior_w * pose_error(self.camera_in_base_prior, x))
        return np.concatenate(r)

    def _jacobian(self, fn, h: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
        r0 = fn(self.X, self.Y)
        jac = np.zeros((r0.size, 12))
        for i in range(12):
            d = np.zeros(12)
            d[i] = h
            jac[:, i] = (fn(perturb(self.X, d[:6]), perturb(self.Y, d[6:])) - r0) / h
        return r0, jac

    def _block_whiten(self, obs) -> np.ndarray:
        ws = [self._whitener(o) for o in obs]
        n = sum(w.shape[0] for w in ws)
        out = np.zeros((n, n))
        i = 0
        for w in ws:
            k = w.shape[0]
            out[i : i + k, i : i + k] = w
            i += k
        return out

    def solve(self, iterations: int = 12, huber: float = 3.0) -> dict:
        """Levenberg-Marquardt over all poses, whitened per pose, Huber-weighted. / 비선형 최소제곱."""
        if not self.ready or not self.observations:
            return {}
        lam = 1e-4
        cost = None
        for _ in range(iterations):
            wb = self._block_whiten(self.observations)
            rd, jd = self._jacobian(self._data_residual)
            rw, jw = wb @ rd, wb @ jd
            rp, jp = self._jacobian(self._prior_residual)
            a = np.abs(rw)
            w = np.where(a <= huber, 1.0, huber / np.maximum(a, 1e-12))
            h = (jw * w[:, None]).T @ jw + jp.T @ jp
            g = (jw * w[:, None]).T @ rw + jp.T @ rp
            cost0 = float((w * rw * rw).sum() + (rp * rp).sum())
            step = np.zeros(12)
            for _try in range(6):
                step = -np.linalg.solve(h + lam * np.diag(np.diag(h) + 1e-9), g)
                x1, y1 = perturb(self.X, step[:6]), perturb(self.Y, step[6:])
                rw1 = wb @ self._data_residual(x1, y1)
                rp1 = self._prior_residual(x1, y1)
                a1 = np.abs(rw1)
                w1 = np.where(a1 <= huber, 1.0, huber / np.maximum(a1, 1e-12))
                cost1 = float((w1 * rw1 * rw1).sum() + (rp1 * rp1).sum())
                if cost1 <= cost0:
                    self.X, self.Y, cost = x1, y1, cost1
                    lam = max(lam / 4, 1e-9)
                    break
                lam *= 8
            else:
                break
            if np.linalg.norm(step) < 1e-8:
                break
        rd = self._data_residual(self.X, self.Y)
        rw = self._block_whiten(self.observations) @ rd
        rms = float(np.sqrt(np.mean(rd**2))) if rd.size else float("nan")
        # Inflate the noise model when the data disagree more than it says (never deflate).
        # 데이터가 잡음 모델보다 더 어긋나면 모델을 키운다(줄이지는 않음).
        dof = rd.size - 12
        chi = float((rw**2).sum() / dof) if dof >= 24 else 1.0
        if chi > 1.0:
            self.sigma = self.pixel_sigma * math.sqrt(chi)
            self._fk_var = self._fk_var0 * chi
        return {
            "reproj_rms_px": rms,
            "chi2_per_dof": chi,
            "sigma_px": self.sigma,
            "cost": cost,
            "n_obs": len(self.observations),
        }

    def information(self) -> np.ndarray:
        """Posterior information (12×12) at the current estimate. / 현재 추정의 정보 행렬."""
        _, jp = self._jacobian(self._prior_residual)
        h = jp.T @ jp
        if self.observations:
            _, jd = self._jacobian(self._data_residual)
            jw = self._block_whiten(self.observations) @ jd
            h = h + jw.T @ jw
        return h

    def pose_information(self, flange_in_base: np.ndarray, ids=None) -> np.ndarray:
        """Predicted information of observing the board at ``flange_in_base``. / 후보의 예측 정보."""
        obj = self.board.object_points(ids)
        pseudo = Observation(flange_in_base, obj, np.zeros((len(obj), 2)))
        _, jac = self._jacobian(lambda x, y: self._data_residual(x, y, [pseudo]))
        jw = self._whitener(pseudo) @ jac
        return jw.T @ jw

    def predict(self, flange_in_base: np.ndarray) -> Prediction:
        """Where the board will appear and whether it will be detectable. / 보드가 보일 위치·가시성."""
        t = self.board_in_camera(flange_in_base)
        uv = project(self.camera_matrix, self.distortion, t, self.board.object_points())
        to_cam = -t[:3, 3] / max(np.linalg.norm(t[:3, 3]), 1e-9)
        facing = float(t[:3, 2] @ to_cam)
        quads = uv.reshape(-1, 4, 2)
        edge = float(min(np.linalg.norm(q[(i + 1) % 4] - q[i]) for q in quads for i in range(4)))
        w, h = self.image_size
        m = self.margin_px
        inside = bool(np.all((uv[:, 0] > m) & (uv[:, 0] < w - m) & (uv[:, 1] > m) & (uv[:, 1] < h - m)))
        visible = inside and t[2, 3] > 0.1 and facing > self.min_facing_cos and edge > self.min_marker_px
        return Prediction(uv, facing, edge, visible)

    # ---- what the user needs: task points in the base frame / 베이스 기준 작업 점
    @staticmethod
    def point_covariance(cov_x: np.ndarray, x: np.ndarray, points_camera) -> np.ndarray:
        """Covariance (k, 3, 3) of camera points mapped to the base by ``X``. / 베이스 기준 공분산.

        With ``X = X̂ · [R(δr), δt]``: ``p_B = R_X (R(δr) p_C + δt) + t_X``, so
        ``∂p_B/∂δr = −R_X [p_C]×`` and ``∂p_B/∂δt = R_X``.
        """
        r = x[:3, :3]
        out = []
        for p in np.asarray(points_camera, dtype=float):
            jac = np.hstack([-r @ skew(p), r])
            out.append(jac @ cov_x @ jac.T)
        return np.stack(out)

    def task_uncertainty(self, information: np.ndarray, points_base) -> float:
        """RMS 1σ (m) of the task points as seen through the camera. / 작업 점 1σ RMS (m).

        A point the camera measures at ``p_C`` lands at ``X · p_C``; ``p_C`` is taken at the
        current estimate, ``p_C = X̂⁻¹ p_B``.
        """
        try:
            cov = np.linalg.inv(information)[:6, :6]
        except np.linalg.LinAlgError:
            return float("inf")  # X not constrained yet / X 가 아직 정해지지 않음
        pts = np.asarray(points_base, dtype=float)
        x_inv = np.linalg.inv(self.X)
        p_cam = pts @ x_inv[:3, :3].T + x_inv[:3, 3]
        c = self.point_covariance(cov, self.X, p_cam)
        return float(np.sqrt(np.mean(np.trace(c, axis1=1, axis2=2))))

    def mount_consistency(self) -> dict:
        """Spread of the per-pose implied mount ``Y_i = T_BF⁻¹ X T_CG,i``. / 자세별 장착 일관성."""
        devs, angs = [], []
        for o in self.observations:
            t = planar_pose(self.camera_matrix, self.distortion, o.object_points, o.image_points)
            e = pose_error(self.Y, np.linalg.inv(o.flange_in_base) @ self.X @ t)
            angs.append(math.degrees(np.linalg.norm(e[:3])))
            devs.append(1000 * np.linalg.norm(e[3:]))
        if not devs:
            return {}
        return {
            "pos_rms_mm": float(np.sqrt(np.mean(np.square(devs)))),
            "rot_rms_deg": float(np.sqrt(np.mean(np.square(angs)))),
        }

    def leave_one_out_mm(self, points_base) -> float:
        """Largest shift of the task points when one pose is left out (mm). / 한 자세 제외 시 최대 변화."""
        pts = np.asarray(points_base, dtype=float)

        def mapped(x):
            x_inv = np.linalg.inv(self.X)
            p_cam = pts @ x_inv[:3, :3].T + x_inv[:3, 3]
            return p_cam @ x[:3, :3].T + x[:3, 3]

        full = mapped(self.X)
        worst = 0.0
        for i in range(len(self.observations)):
            sub = ActiveHandEye(
                self.camera_matrix,
                self.distortion,
                self.board,
                self.camera_in_base_prior,
                self.prior_sigma,
                self.pixel_sigma,
                self.fk_sigma,
                self.image_size,
            )
            sub.initialise(self.X, self.Y)
            sub._mount_prior = self._mount_prior
            sub.observations = [o for j, o in enumerate(self.observations) if j != i]
            sub.solve()
            shift = np.sqrt(np.mean(np.sum((mapped(sub.X) - full) ** 2, axis=1)))
            worst = max(worst, float(shift) * 1000.0)
        return worst


# ----------------------------------------------------------------------------- choosing the next pose
@dataclass
class Candidate:
    """A reachable pose the robot may visit. / 로봇이 갈 수 있는 후보 자세."""

    joints: np.ndarray
    flange_in_base: np.ndarray
    label: str = ""
    used: bool = False
    failed: int = 0


def move_time(
    dq: np.ndarray, v_max: float = math.radians(24.0), a_max: float = math.radians(36.0), floor: float = 0.8
) -> float:
    """Duration of a minimum-jerk joint move within speed and acceleration limits.

    최소 저크 관절 이동 시간: 최고 속도 1.875Δ/T ≤ v, 최고 가속도 5.77Δ/T² ≤ a.
    """
    d = float(np.max(np.abs(dq)))
    return max(floor, 1.875 * d / v_max, math.sqrt(5.7735 * d / a_max))


def choose_next(
    est: ActiveHandEye, candidates: list, current_joints: np.ndarray, points_base, dwell_s: float = 1.4
) -> tuple[Candidate | None, dict]:
    """Greedy information-per-second choice. / 초당 정보 이득이 가장 큰 후보.

    Score = log(σ_now² / σ_after²) / (move time + dwell), where σ is the RMS 1σ of the task points.
    Candidates where the board would not be detectable are skipped.
    """
    h = est.information()
    now = est.task_uncertainty(h, points_base)
    best, best_score, best_info = None, -1.0, {}
    for c in candidates:
        if c.used or c.failed >= 1:
            continue
        pred = est.predict(c.flange_in_base)
        if not pred.visible:
            continue
        after = est.task_uncertainty(h + est.pose_information(c.flange_in_base), points_base)
        t = move_time(c.joints - current_joints) + dwell_s
        score = 2.0 * math.log(now / max(after, 1e-12)) / t
        if score > best_score:
            best, best_score = c, score
            best_info = {
                "sigma_now_mm": now * 1e3,
                "sigma_after_mm": after * 1e3,
                "seconds": t,
                "score": score,
                "facing": pred.facing_cos,
                "marker_px": pred.marker_px,
            }
    return best, best_info
