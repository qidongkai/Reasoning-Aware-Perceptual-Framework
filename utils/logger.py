import time
import os
import json


class Logger:
    """日志记录器"""

    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, f"train_{time.strftime('%Y%m%d_%H%M%S')}.log")

    def log(self, msg):
        """记录日志"""
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        log_msg = f"[{timestamp}] {msg}"

        # 打印到控制台
        print(log_msg)

        # 写入文件
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_msg + '\n')

    def log_step(self, epoch, step, loss, lr):
        """记录训练步骤"""
        self.log(f"Epoch {epoch + 1} Step {step} - Loss: {loss:.4f}, LR: {lr:.6f}")

    def log_val(self, epoch, metrics):
        """记录验证结果"""
        self.log(f"Epoch {epoch + 1} Val - {json.dumps(metrics, indent=2)}")

    def log_epoch(self, epoch, train_loss, val_metrics):
        """记录epoch总结"""
        self.log(
            f"Epoch {epoch + 1} Summary - Train Loss: {train_loss:.4f}, Val Metrics: {json.dumps(val_metrics, indent=2)}")