class DecisionEngine:
    def __init__(self, config):
        self.config = config

    def generate(self, label, confidence, conflict, is_known, similarity, candidates, mask_score):
        """生成结构化的决策结果"""
        return {
            "label": label,
            "confidence": round(confidence, 4),
            "conflict": round(conflict, 4),
            "is_known": is_known,
            "similarity": round(similarity, 4),
            "top3_candidates": candidates[:3],
            "mask_score": round(mask_score, 4)
        }