import json
import torch


class WildPlantKG:
    def __init__(self, kg_path):
        """加载植物知识图谱"""
        with open(kg_path, 'r', encoding='utf-8') as f:
            self.kg = json.load(f)
        self.species = self.kg.get('species', [])
        self.embeddings = torch.stack([
            torch.tensor(s.get('embedding', []), dtype=torch.float32)
            for s in self.species
        ]) if self.species else torch.tensor([])

    def retrieve_topk(self, feat, k=5):
        """基于余弦相似度检索Top-K物种"""
        if len(self.embeddings) == 0 or len(feat) == 0:
            return [], []

        sim = torch.cosine_similarity(
            feat.unsqueeze(0),
            self.embeddings,
            dim=-1
        )
        vals, indices = torch.topk(sim, min(k, len(sim)))
        return [self.species[i] for i in indices], vals.tolist()


class RetrievalModule:
    def __init__(self, config):
        self.kg = WildPlantKG(config['kg_path'])
        self.k = config['topk']
        self.env_thresh = config['env_filter_threshold']

    def __call__(self, feat, env_context):
        """检索并基于环境过滤候选物种"""
        candidates, sims = self.kg.retrieve_topk(feat, self.k)
        filtered = []

        for cand in candidates:
            score = self.env_sim(cand.get('environment', []), env_context)
            if score >= self.env_thresh:
                filtered.append(cand)

        return filtered[:self.k]

    def env_sim(self, sp_env, ctx_env):
        """计算环境相似度（Jaccard系数）"""
        s1, s2 = set(sp_env), set(ctx_env)
        union = s1 | s2
        return len(s1 & s2) / len(union) if union else 0.0