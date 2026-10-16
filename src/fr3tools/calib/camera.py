"""Camera intrinsics from a chessboard (``fr3-camera-calib``), a camera source and the ArUco
detector for the gripper board. Needs OpenCV (``pip install 'fr3tools[vision]'``).

Calibration recipe (it is what made the reprojection error drop below 1 px):

* hold the board (or the camera) still for a moment at each view; blurred views are rejected
  (``--still-px``) and so are views too close to the previous capture (``--min-move``);
* tilt it: views that are all parallel to the image plane cannot separate the focal length from
  the distance. The run does not finish until the board normals spread by ``--min-spread`` degrees;
* cover the corners of the image, near and far;
* with few views, fix the aspect ratio (square pixels) and k3, as done here by default;
* the result is valid for one resolution only: calibrate again after changing it.

체스보드로 카메라 내부 파라미터 보정. 매 뷰에서 잠깐 멈추고(블러 제외), 보드를 기울여(법선 퍼짐
``--min-spread``), 화면 구석·가까이·멀리를 덮는다. 뷰가 적으면 화소비와 k3 를 고정한다. 해상도가
바뀌면 다시 보정한다.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np

from fr3tools.calib.handeye import GripperBoard, planar_pose


def _cv2():
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("this needs OpenCV: pip install 'fr3tools[vision]'") from exc
    return cv2


# ----------------------------------------------------------------------------- pure helpers
def chessboard_points(cols: int, rows: int, square: float) -> np.ndarray:
    """Inner-corner coordinates (z = 0) in OpenCV's order, row by row. / 내부 코너 좌표."""
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    return np.hstack([grid * square, np.zeros((len(grid), 1))]).astype(np.float32)


def view_normal(obj: np.ndarray, img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Board normal in the camera frame from a rough camera model (view-diversity check).

    대략의 카메라 모델로 구한 카메라 좌표 보드 법선(뷰 다양성 판정용).
    """
    w, h = size
    k = np.array([[0.8 * w, 0.0, w / 2.0], [0.0, 0.8 * w, h / 2.0], [0.0, 0.0, 1.0]])
    n = planar_pose(k, np.zeros(5), obj, img, refine=3)[:3, 2]
    return n if n[2] < 0 else -n


def normal_spread_deg(normals) -> float:
    """Largest angle between any two board normals (deg). / 보드 법선 사이 최대 각."""
    best = 0.0
    for i, a in enumerate(normals):
        for b in normals[:i]:
            best = max(best, float(np.degrees(np.arccos(np.clip(abs(float(a @ b)), -1.0, 1.0)))))
    return best


def calibrate(objs, imgs, size: tuple[int, int], fix_aspect: bool = True, fix_k3: bool = True) -> dict:
    """``cv2.calibrateCamera`` with per-view errors and one round of outlier rejection.

    뷰별 오차를 구하고 이상치 뷰(블러·오검출)를 한 번 제외한 뒤 다시 보정한다.
    """
    cv2 = _cv2()
    flags = (cv2.CALIB_FIX_ASPECT_RATIO if fix_aspect else 0) | (cv2.CALIB_FIX_K3 if fix_k3 else 0)

    def run(o, i):
        k0 = np.array([[1.0, 0.0, size[0] / 2.0], [0.0, 1.0, size[1] / 2.0], [0.0, 0.0, 1.0]])
        rms, k, dist, rvecs, tvecs = cv2.calibrateCamera(
            [np.asarray(v, np.float32) for v in o],
            [np.asarray(v, np.float32) for v in i],
            size,
            k0,
            None,
            flags=flags,
        )
        per = []
        for op, ip, r, t in zip(o, i, rvecs, tvecs, strict=True):
            pp, _ = cv2.projectPoints(np.asarray(op, np.float32), r, t, k, dist)
            err = pp.reshape(-1, 2) - np.asarray(ip).reshape(-1, 2)
            per.append(float(np.sqrt(np.mean(np.sum(err**2, axis=1)))))
        return rms, k, dist, per

    objs, imgs = list(objs), list(imgs)
    rms, k, dist, per = run(objs, imgs)
    bad = [i for i, e in enumerate(per) if e > max(1.0, 2.5 * float(np.median(per)))]
    if bad and len(objs) - len(bad) >= 8:
        objs = [o for i, o in enumerate(objs) if i not in bad]
        imgs = [o for i, o in enumerate(imgs) if i not in bad]
        rms, k, dist, per = run(objs, imgs)
    return {
        "width": int(size[0]),
        "height": int(size[1]),
        "K": k.tolist(),
        "dist": dist.ravel().tolist(),
        "rms_px": float(rms),
        "per_view_rms_px": per,
        "n_views": len(objs),
        "rejected_views": len(bad) if bad and len(objs) >= 8 else 0,
    }


def save_intrinsics(path: str, result: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


def load_intrinsics(path: str) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """(K, dist, (width, height)) from a ``fr3-camera-calib`` JSON. / 보정 결과 읽기."""
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return (
        np.asarray(d["K"], dtype=float),
        np.asarray(d["dist"], dtype=float),
        (int(d["width"]), int(d["height"])),
    )


# ----------------------------------------------------------------------------- camera and detector
class CameraSource:
    """OpenCV camera returning the freshest RGB frame. / 가장 최신 RGB 프레임을 주는 카메라.

    USB cameras buffer a few frames; :meth:`read` drops the stale ones so a frame taken after the
    robot settled really shows the settled robot.
    USB 카메라는 프레임을 버퍼링하므로 낡은 프레임을 버린다(로봇 정지 후의 프레임을 보장).
    """

    def __init__(self, device="0", width: int = 1920, height: int = 1080, flush: int = 3):
        cv2 = _cv2()
        dev = int(device) if str(device).isdigit() else device
        self.cap = cv2.VideoCapture(dev)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.flush = flush
        if not self.cap.isOpened():
            raise RuntimeError(f"camera {device} did not open")

    def read(self) -> np.ndarray:
        cv2 = _cv2()
        for _ in range(self.flush):
            self.cap.grab()
        ok, bgr = self.cap.read()
        if not ok:
            raise RuntimeError("camera read failed")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        self.cap.release()


class ArucoBoardDetector:
    """Detect the gripper board: ``(ids, corners (n·4, 2), board_in_camera)`` or ``None``.

    그리퍼 보드 검출: (id 목록, 코너, 카메라 기준 보드 포즈) 또는 None.
    """

    def __init__(self, board: GripperBoard, k: np.ndarray, dist: np.ndarray, min_markers: int = 2):
        cv2 = _cv2()
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, board.dictionary))
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(dictionary, params)
        self.board, self.k, self.dist, self.min_markers = board, k, dist, min_markers

    def detect(self, rgb: np.ndarray):
        cv2 = _cv2()
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is None:
            return None
        found = {
            int(i): c.reshape(4, 2)
            for i, c in zip(ids.ravel(), corners, strict=True)
            if int(i) in self.board.centers
        }
        if len(found) < self.min_markers:
            return None
        order = sorted(found)
        uv = np.concatenate([found[i] for i in order])
        return order, uv, planar_pose(self.k, self.dist, self.board.object_points(order), uv)


# ----------------------------------------------------------------------------- CLI
def _detect_chessboard(gray, pattern):
    cv2 = _cv2()
    ok, corners = cv2.findChessboardCorners(
        gray, pattern, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
    )
    if not ok:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria).reshape(-1, 2)


def main(argv=None) -> None:
    """``fr3-camera-calib``: live capture (auto) or a folder of images. / 실시간 캡처 또는 이미지 폴더."""
    cv2 = _cv2()
    ap = argparse.ArgumentParser(description="chessboard intrinsics calibration")
    ap.add_argument("--cols", type=int, default=9, help="inner corners per row")
    ap.add_argument("--rows", type=int, default=6, help="inner corners per column")
    ap.add_argument("--square-mm", type=float, default=25.0)
    ap.add_argument("--images", help="glob of images (offline mode), e.g. 'shots/*.png'")
    ap.add_argument("--device", default="0")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--frames", type=int, default=25, help="views to collect")
    ap.add_argument("--interval", type=float, default=1.0, help="min seconds between captures")
    ap.add_argument("--min-move", type=float, default=25.0, help="px the corners must move between captures")
    ap.add_argument("--still-px", type=float, default=1.5, help="max corner motion per frame (blur guard)")
    ap.add_argument("--min-spread", type=float, default=10.0, help="required spread of board normals (deg)")
    ap.add_argument("--max-rms", type=float, default=1.0, help="pass threshold (px)")
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--out", default="calib_out/camera.json")
    a = ap.parse_args(argv)
    pattern = (a.cols, a.rows)
    obj = chessboard_points(a.cols, a.rows, a.square_mm / 1000.0)
    objs, imgs, normals = [], [], []
    size = None

    if a.images:
        for path in sorted(glob.glob(a.images)):
            gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            size = (gray.shape[1], gray.shape[0])
            corners = _detect_chessboard(gray, pattern)
            print(f"{path}: {'ok' if corners is not None else 'no board'}")
            if corners is not None:
                objs.append(obj)
                imgs.append(corners)
                normals.append(view_normal(obj, corners, size))
    else:
        cam = CameraSource(a.device, a.width, a.height, flush=0)
        prev = last = None
        t_last = 0.0
        t0 = time.time()
        print("move the board, stop for a second at each view, tilt it; Ctrl-C to stop early", flush=True)
        try:
            while len(objs) < a.frames or normal_spread_deg(normals) < a.min_spread:
                if time.time() - t0 > a.timeout or len(objs) >= 2 * a.frames:
                    break
                gray = cv2.cvtColor(cam.read(), cv2.COLOR_RGB2GRAY)
                size = (gray.shape[1], gray.shape[0])
                corners = _detect_chessboard(gray, pattern)
                if corners is None:
                    last = None
                    continue
                still = (
                    last is not None and float(np.mean(np.linalg.norm(corners - last, axis=1))) < a.still_px
                )
                last = corners
                moved = prev is None or float(np.mean(np.linalg.norm(corners - prev, axis=1))) >= a.min_move
                if still and moved and time.time() - t_last >= a.interval:
                    objs.append(obj)
                    imgs.append(corners)
                    normals.append(view_normal(obj, corners, size))
                    prev, t_last = corners, time.time()
                    print(
                        f"captured {len(objs)}/{a.frames}, spread {normal_spread_deg(normals):.0f} deg",
                        flush=True,
                    )
        except KeyboardInterrupt:
            pass
        finally:
            cam.close()

    if len(objs) < 8 or size is None:
        raise SystemExit(f"FAIL: {len(objs)} usable views (< 8)")
    result = calibrate(objs, imgs, size)
    result["board"] = {"type": "chessboard", "cols": a.cols, "rows": a.rows, "square_m": a.square_mm / 1000.0}
    result["normal_spread_deg"] = normal_spread_deg(normals)
    save_intrinsics(a.out, result)
    k = np.asarray(result["K"])
    ok = result["rms_px"] < a.max_rms and result["normal_spread_deg"] >= a.min_spread
    print(
        f"{size[0]}x{size[1]} fx={k[0, 0]:.1f} fy={k[1, 1]:.1f} cx={k[0, 2]:.1f} cy={k[1, 2]:.1f} "
        f"rms {result['rms_px']:.3f} px over {result['n_views']} views, "
        f"spread {result['normal_spread_deg']:.0f} deg "
        f"-> {'PASS' if ok else 'WARN (tilt the board more, check focus and flatness)'}; saved {a.out}"
    )
