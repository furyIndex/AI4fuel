import numpy as np


class LogTargetScaler:

    def __init__(self, base: float = 10.0, eps: float = None, with_std: bool = False):
        self.base = float(base)
        self.eps = eps
        self.with_std = bool(with_std)
        self._std = None
        self._mean_ = 0.0
        self._scale_ = 1.0
        self.eps_ = None
        self.z_min_ = None
        self.z_max_ = None
        self.y_max_ = None
        self.clip_margin = 1.0

    @staticmethod
    def _ensure_1d(a):
        import numpy as np
        a = np.asarray(a)
        if a.ndim == 2 and a.shape[1] == 1:
            return a.ravel()
        return a.ravel()

    def fit(self, y):
        y = self._ensure_1d(y)
        # auto epsilon from the smallest positive
        if self.eps is None:
            pos = y[y > 0]
            min_pos = float(np.min(pos)) if pos.size > 0 else 1e-12
            self.eps_ = max(1e-12, 0.1 * min_pos)
        else:
            self.eps_ = float(self.eps)
        z = self._to_log(y)
        self.z_min_ = float(z.min())
        self.z_max_ = float(z.max())
        self.y_max_ = float(np.max(y)) if y.size > 0 else 1.0
        if self.with_std:
            from sklearn.preprocessing import StandardScaler
            self._std = StandardScaler()
            z2d = z.reshape(-1, 1)
            _ = self._std.fit_transform(z2d)
            self._mean_ = float(self._std.mean_[0])
            self._scale_ = float(self._std.scale_[0])
        return self

    def _to_log(self, y):
        y = self._ensure_1d(y).astype(float)
        return np.log(y + self.eps_) / np.log(self.base)

    def transform(self, y):
        z = self._to_log(y)
        if self._std is not None:
            z = (z - self._mean_) / (self._scale_ + 1e-12)
        return z

    def inverse_transform(self, z):
        z = self._ensure_1d(z).astype(float)
        if self._std is not None:
            z = z * (self._scale_ + 1e-12) + self._mean_
        z_lo = (self.z_min_ if self.z_min_ is not None else -50.0) - self.clip_margin
        z_hi = (self.z_max_ if self.z_max_ is not None else 50.0) + self.clip_margin
        z = np.clip(z, z_lo, z_hi)
        y = (self.base ** z) - (self.eps_ if self.eps_ is not None else 0.0)
        y = np.nan_to_num(y, nan=0.0, posinf=(self.y_max_ * 1e3 if self.y_max_ is not None else 1e12), neginf=0.0)
        return y

    def __repr__(self):
        return f"LogTargetScaler(base={self.base}, eps={self.eps_}, with_std={self._std is not None})"


def inv_if_scaler(arr, scaler):
    '''
    反归一化、反log化
    :param arr:
    :param scaler:
    :return:
    '''
    if scaler is None:
        return arr
    import numpy as np
    out = scaler.inverse_transform(arr.reshape(-1, 1)).ravel()
    out = np.nan_to_num(out, nan=0.0, posinf=(np.max(out[np.isfinite(out)]) if np.any(np.isfinite(out)) else 1e12),
                        neginf=0.0)
    return out