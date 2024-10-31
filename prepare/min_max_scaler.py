import numpy as np
from sklearn.preprocessing import MinMaxScaler



class TargetScaler():
    def __init__(self, min, max):
        self.min_max_scaler = MinMaxScaler(feature_range=(min, max))

    def batchMSE(self, x, y):
        output_cpu = x.detach().cpu().numpy()
        target_cpu = y.detach().cpu().numpy()
        # 反归一化输出
        original_predictions = self.min_max_scaler.inverse_transform(output_cpu.reshape(-1, 1)).astype(np.float32)
        original_target = self.min_max_scaler.inverse_transform(target_cpu.reshape(-1, 1)).astype(np.float32)
        batch_mse = np.mean((original_predictions - original_target) ** 2)
        return batch_mse
