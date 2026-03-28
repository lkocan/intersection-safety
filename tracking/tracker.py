import numpy as np
from scipy.optimize import linear_sum_assignment


class KalmanFilter3D:
    """Stav: [x, y, z, l, w, h, rot, vx, vy, vz]"""

    def __init__(self, bbox: np.ndarray):
        self.F = np.eye(10, dtype=np.float32)
        self.F[0, 7] = 1.0
        self.F[1, 8] = 1.0
        self.F[2, 9] = 1.0

        self.H = np.zeros((7, 10), dtype=np.float32)
        self.H[:7, :7] = np.eye(7)

        self.Q = np.eye(10, dtype=np.float32)
        self.Q[7:, 7:] *= 0.1

        self.R = np.eye(7, dtype=np.float32)
        self.R[:3, :3] *= 0.5
        self.R[3:6, 3:6] *= 1.0
        self.R[6, 6] *= 0.5

        self.P = np.eye(10, dtype=np.float32) * 10.0
        self.x = np.zeros(10, dtype=np.float32)
        self.x[:7] = bbox

    def predict(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:7]

    def update(self, bbox: np.ndarray):
        y = bbox - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(10) - K @ self.H) @ self.P

    def get_state(self) -> np.ndarray:
        return self.x[:7].copy()

    @property
    def velocity(self) -> np.ndarray:
        """Vráti [vx, vy] v m/frame."""
        return self.x[7:9].copy()


class Track:
    _id_counter = 0

    def __init__(self, bbox: np.ndarray, class_id: int, score: float):
        Track._id_counter += 1
        self.id                = Track._id_counter
        self.class_id          = class_id
        self.score             = score
        self.kf                = KalmanFilter3D(bbox)
        self.hits              = 1
        self.age               = 1
        self.time_since_update = 0
        self.history           = [bbox[:3].copy()]

    def predict(self) -> np.ndarray:
        self.age += 1
        self.time_since_update += 1
        return self.kf.predict()

    def update(self, bbox: np.ndarray, score: float):
        self.hits += 1
        self.time_since_update = 0
        self.score = score
        self.kf.update(bbox)
        self.history.append(self.kf.get_state()[:3].copy())

    @property
    def state(self) -> np.ndarray:
        return self.kf.get_state()

    @property
    def velocity(self) -> np.ndarray:
        return self.kf.velocity

    @property
    def is_confirmed(self) -> bool:
        return self.hits >= 2


def iou_bev(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ax2 = a[0] - a[3]/2, a[0] + a[3]/2
    ay1, ay2 = a[1] - a[4]/2, a[1] + a[4]/2
    bx1, bx2 = b[0] - b[3]/2, b[0] + b[3]/2
    by1, by2 = b[1] - b[4]/2, b[1] + b[4]/2
    ix = max(0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    union = a[3]*a[4] + b[3]*b[4] - inter
    return inter / union if union > 0 else 0.0


def center_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2))


class Tracker3D:
    def __init__(
        self,
        max_age:        int   = 3,
        min_hits:       int   = 2,
        dist_threshold: float = 8.0,
    ):
        self.max_age        = max_age
        self.min_hits       = min_hits
        self.dist_threshold = dist_threshold
        self.tracks: list   = []
        self.frame_count    = 0

    def update(self, detections: np.ndarray) -> list:
        """
        detections: (N, 9) — [x,y,z,l,w,h,rot,class_id,score]
        Vráti list potvrdených trackov.
        """
        self.frame_count += 1

        predicted = [t.predict() for t in self.tracks]

        if len(self.tracks) > 0 and len(detections) > 0:
            matched, unmatched_dets, unmatched_trks = self._associate(
                detections, predicted
            )
        else:
            matched        = []
            unmatched_dets = list(range(len(detections)))
            unmatched_trks = list(range(len(self.tracks)))

        for d_idx, t_idx in matched:
            self.tracks[t_idx].update(detections[d_idx][:7], detections[d_idx][8])

        for d_idx in unmatched_dets:
            d = detections[d_idx]
            self.tracks.append(Track(d[:7], int(d[7]), float(d[8])))

        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        results = []
        for t in self.tracks:
            if not t.is_confirmed:
                continue
            s = t.state
            results.append({
                'id':       t.id,
                'class_id': t.class_id,
                'x':        float(s[0]),
                'y':        float(s[1]),
                'z':        float(s[2]),
                'l':        float(s[3]),
                'w':        float(s[4]),
                'h':        float(s[5]),
                'rot':      float(s[6]),
                'vx':       float(t.velocity[0]),
                'vy':       float(t.velocity[1]),
                'score':    t.score,
                'age':      t.age,
                'history':  [p.tolist() for p in t.history],
            })
        return results

    def _associate(self, detections, predicted):
        n_det = len(detections)
        n_trk = len(predicted)
        cost  = np.full((n_det, n_trk), 1e6, dtype=np.float32)

        for d_i, det in enumerate(detections):
            for t_i, pred in enumerate(predicted):
                dist = center_distance(det[:7], pred)
                if dist <= self.dist_threshold:
                    cost[d_i, t_i] = dist * (1.0 - iou_bev(det[:7], pred))

        d_idx, t_idx = linear_sum_assignment(cost)

        matched        = []
        unmatched_dets = list(range(n_det))
        unmatched_trks = list(range(n_trk))

        for d, t in zip(d_idx, t_idx):
            if cost[d, t] >= 1e6:
                continue
            matched.append((d, t))
            unmatched_dets.remove(d)
            unmatched_trks.remove(t)

        return matched, unmatched_dets, unmatched_trks

    def reset(self):
        self.tracks      = []
        self.frame_count = 0
        Track._id_counter = 0
