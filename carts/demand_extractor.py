from dataclasses import dataclass
from collections import defaultdict
import random

@dataclass
class MoEConfig:
    num_gpus:int=16; gpus_per_domain:int=4; num_experts:int=16; top_k:int=2
    tokens_per_gpu:int=64; skew:float=0.0; token_bytes:int=2048; seed:int=0
    @property
    def num_domains(self): return self.num_gpus//self.gpus_per_domain
    def domain_of_gpu(self,g): return g//self.gpus_per_domain
    def gpu_of_expert(self,e): return e

@dataclass
class DemandResult:
    cfg:MoEConfig; pair_tokens:dict; routes:list
    def lam(self,di,dj): return len(self.pair_tokens.get((di,dj),[]))
    def unique(self,di,dj): return len(set(self.pair_tokens.get((di,dj),[])))
    def _m(self,fn):
        n=self.cfg.num_domains; return [[fn(i,j) for j in range(n)] for i in range(n)]
    def lambda_matrix(self): return self._m(self.lam)
    def unique_matrix(self): return self._m(self.unique)
    def dedup_savings_matrix(self): return self._m(lambda i,j:self.lam(i,j)-self.unique(i,j))
    def total_pairs(self,inter_only=True):
        return sum(len(t) for (di,dj),t in self.pair_tokens.items() if not(inter_only and di==dj))
    def total_unique(self,inter_only=True):
        return sum(len(set(t)) for (di,dj),t in self.pair_tokens.items() if not(inter_only and di==dj))
    def dedup_fraction(self,inter_only=True):
        p=self.total_pairs(inter_only); return 0.0 if p==0 else (p-self.total_unique(inter_only))/p

def _weights(cfg,rng):
    ex=list(range(cfg.num_experts)); rng.shuffle(ex)
    return {e:1.0/((r+1)**cfg.skew) for r,e in enumerate(ex)}

def _topk(n,w,k,rng):
    pool=list(range(n)); ww=[w[e] for e in pool]; ch=[]
    for _ in range(k):
        tot=sum(ww); rr=rng.random()*tot; acc=0.0
        for i,x in enumerate(ww):
            acc+=x
            if rr<=acc: ch.append(pool.pop(i)); ww.pop(i); break
    return ch

def generate_demand(cfg):
    rng=random.Random(cfg.seed); w=_weights(cfg,rng)
    pt=defaultdict(list); routes=[]
    for g in range(cfg.num_gpus):
        di=cfg.domain_of_gpu(g)
        for t in range(cfg.tokens_per_gpu):
            tid=g*cfg.tokens_per_gpu+t; ex=_topk(cfg.num_experts,w,cfg.top_k,rng)
            routes.append((g,tid,ex))
            for e in ex: pt[(di,cfg.domain_of_gpu(cfg.gpu_of_expert(e)))].append(tid)
    return DemandResult(cfg,dict(pt),routes)

def from_routes(cfg,routes):
    pt=defaultdict(list)
    for (g,tid,ex) in routes:
        di=cfg.domain_of_gpu(g)
        for e in ex: pt[(di,cfg.domain_of_gpu(cfg.gpu_of_expert(e)))].append(tid)
    return DemandResult(cfg,dict(pt),list(routes))

def pm(title,mat):
    n=len(mat); print("\n"+title+"  (rows=src, cols=dst)")
    print("        "+"".join("d%-6d"%j for j in range(n)))
    for i,row in enumerate(mat): print("  d%-4d"%i+"".join("%-7d"%v for v in row))

def validate():
    print("="*60); print("VALIDATION"); print("="*60)
    a=from_routes(MoEConfig(num_gpus=4,gpus_per_domain=2,num_experts=4,top_k=1,tokens_per_gpu=2),
                  [(0,0,[2]),(0,1,[3]),(1,2,[0]),(1,3,[2])])
    assert a.lam(0,1)==3 and a.unique(0,1)==3 and a.lam(0,0)==1; print("  Case A accounting:        PASS")
    b=from_routes(MoEConfig(num_gpus=4,gpus_per_domain=2,num_experts=4,top_k=2,tokens_per_gpu=1),[(0,0,[2,3])])
    assert b.lam(0,1)==2 and b.unique(0,1)==1; print("  Case B dedup distinct<pairs: PASS")
    c=from_routes(MoEConfig(num_gpus=4,gpus_per_domain=2,num_experts=4,top_k=2,tokens_per_gpu=1),[(0,0,[0,2])])
    assert c.lam(0,1)==1 and c.unique(0,1)==1 and c.lam(0,0)==1; print("  Case C split no-dedup:    PASS")
    print("\n  All validation cases passed.\n")

def demo():
    print("="*60); print("DEMO - 16 GPU, 4x4 domains, 16 experts, top-2"); print("="*60)
    for sk in (0.0,1.5):
        r=generate_demand(MoEConfig(skew=sk,seed=0)); print("\n----- skew = %s -----"%sk)
        pm("lambda (pairs)",r.lambda_matrix()); pm("unique (distinct)",r.unique_matrix())
        pm("dedup savings",r.dedup_savings_matrix())
        print("\n  inter pairs : %d"%r.total_pairs()); print("  inter unique: %d"%r.total_unique())
        print("  inter dedup %%: %.1f%%"%(r.dedup_fraction()*100))

if __name__=="__main__":
    validate(); demo()
