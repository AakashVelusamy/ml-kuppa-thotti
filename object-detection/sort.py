import numpy as np
from filterpy.kalman import KalmanFilter
from typing import Optional, List, Tuple
import numpy.typing as npt

def linear_assignment(cost_matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.int32]:
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[i, j] for i, j in enumerate(x) if j >= 0], dtype=np.int32)
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)), dtype=np.int32)

def iou(bb_test: npt.NDArray[np.float64], bb_gt: npt.NDArray[np.float64]) -> float:
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
              + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh + 1e-6)
    return o

class KalmanBoxTracker:
    count = 0

    def __init__(self, bbox: npt.NDArray[np.float64]):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1,0,0,0,1,0,0],
                              [0,1,0,0,0,1,0],
                              [0,0,1,0,0,0,1],
                              [0,0,0,1,0,0,0],
                              [0,0,0,0,1,0,0],
                              [0,0,0,0,0,1,0],
                              [0,0,0,0,0,0,1]], dtype=np.float64)
        self.kf.H = np.array([[1,0,0,0,0,0,0], # type: ignore
                              [0,1,0,0,0,0,0],
                              [0,0,1,0,0,0,0],
                              [0,0,0,1,0,0,0]], dtype=np.float64)
        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01
        self.kf.x[:4] = self.convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history: List[npt.NDArray[np.float64]] = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox: npt.NDArray[np.float64]) -> None:
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self.convert_bbox_to_z(bbox))

    def predict(self) -> npt.NDArray[np.float64]:
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self) -> npt.NDArray[np.float64]:
        return self.convert_x_to_bbox(self.kf.x)

    @staticmethod
    def convert_bbox_to_z(bbox: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w / 2.
        y = bbox[1] + h / 2.
        s = w * h
        r = w / float(h + 1e-6)
        return np.array([x, y, s, r], dtype=np.float64).reshape((4, 1))

    @staticmethod
    def convert_x_to_bbox(x: npt.NDArray[np.float64], score: Optional[float] = None) -> npt.NDArray[np.float64]:
        w = np.sqrt(x[2] * x[3])
        h = x[2] / (w + 1e-6)
        return np.array([x[0] - w/2., x[1] - h/2., x[0] + w/2., x[1] + h/2.], dtype=np.float64).reshape((1, 4))

class Sort:
    def __init__(self, max_age: int = 1, min_hits: int = 3, iou_threshold: float = 0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers: List[KalmanBoxTracker] = []
        self.frame_count = 0

    def update(self, dets: npt.NDArray[np.float64] = np.empty((0, 5), dtype=np.float64)) -> npt.NDArray[np.float64]:
        self.frame_count += 1
        trks = np.zeros((len(self.trackers), 5), dtype=np.float64)
        to_del: List[int] = []
        ret: List[npt.NDArray[np.float64]] = []

        for t, _ in enumerate(trks):
            pos = self.trackers[t].predict().flatten()
            trks[t, :] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)

        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        matched, unmatched_dets, unmatched_trks = self.associate_detections_to_trackers(dets, trks.astype(np.float64))

        for t, trk in enumerate(self.trackers):
            if t not in unmatched_trks:
                d = dets[matched[np.where(matched[:,1]==t)[0],0].astype(int), :]
                trk.update(d[0,:])

        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i,:])
            self.trackers.append(trk)

        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id+1])).reshape(1, -1))
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5), dtype=np.float64)

    def associate_detections_to_trackers(
        self, 
        dets: npt.NDArray[np.float64], 
        trks: npt.NDArray[np.float64]
    ) -> Tuple[npt.NDArray[np.int32], npt.NDArray[np.int32], npt.NDArray[np.int32]]:

        if len(trks) == 0:
            return (np.empty((0,2), dtype=np.int32),
                    np.arange(len(dets), dtype=np.int32),
                    np.empty((0,), dtype=np.int32))

        iou_matrix = np.zeros((len(dets), len(trks)), dtype=np.float64)
        for d, det in enumerate(dets):
            for t, trk in enumerate(trks):
                iou_matrix[d, t] = iou(det, trk)

        matched_indices = linear_assignment(-iou_matrix)

        unmatched_dets: List[int] = [d for d in range(len(dets)) if d not in matched_indices[:,0]]
        unmatched_trks: List[int] = [t for t in range(len(trks)) if t not in matched_indices[:,1]]

        matches: List[npt.NDArray[np.int32]] = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < self.iou_threshold:
                unmatched_dets.append(m[0])
                unmatched_trks.append(m[1])
            else:
                matches.append(m.reshape(1,2))
        if len(matches) == 0:
            matches_arr = np.empty((0,2), dtype=np.int32)
        else:
            matches_arr = np.concatenate(matches, axis=0)
        return matches_arr, np.array(unmatched_dets, dtype=np.int32), np.array(unmatched_trks, dtype=np.int32)
