import numpy as np


class DSReasoner:
    def __init__(self, config):
        self.conflict_thresh = config['conflict_threshold']

    def build_bpa(self, scores):
        """构建基本概率分配(BPA)"""
        s = np.array(scores)
        s = s / (s.sum() + 1e-8)  # 防止除零
        unknown = 1 - s[0] if len(s) > 0 else 1.0
        return np.concatenate([s, [unknown]])

    def combine(self, m1, m2):
        """D-S证据融合"""
        if m1 is None or m2 is None:
            return None, 1.0

        n = len(m1)
        m = np.zeros(n)
        K = 0.0  # 冲突系数

        for i in range(n):
            for j in range(n):
                if i == j:
                    m[i] += m1[i] * m2[j]
                else:
                    K += m1[i] * m2[j]

        if K > self.conflict_thresh:
            return None, K

        m /= (1 - K + 1e-8)
        return m, K

    def infer(self, visual, knowledge, env):
        """推理主流程"""
        # 构建各证据源的BPA
        m_vis = self.build_bpa(visual)
        m_kno = self.build_bpa(knowledge)
        m_env = self.build_bpa(env)

        # 逐步融合
        m_vk, K1 = self.combine(m_vis, m_kno)
        if K1 > self.conflict_thresh:
            return m_vk, K1, "unknown"

        m_final, K2 = self.combine(m_vk, m_env)
        if m_final is None:
            return None, K2, "unknown"

        pred = np.argmax(m_final)
        label = "unknown" if pred == len(m_final) - 1 else f"species_{pred}"
        return m_final, K2, label