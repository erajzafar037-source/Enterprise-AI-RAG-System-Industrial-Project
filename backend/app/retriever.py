from dataclasses import dataclass
import re,numpy as np
try: import faiss
except ImportError: faiss=None
@dataclass
class Chunk: text:str; source:str; page:int|None=None
class Retriever:
    def __init__(self): self.chunks=[]; self.matrix=None; self.index=None
    @staticmethod
    def _tokens(text): return re.findall(r"[a-zA-Z0-9_]{2,}",text.lower())
    def embed(self,texts):
        vocab={}
        for text in texts:
            for token in set(self._tokens(text)):
                if token not in vocab: vocab[token]=len(vocab)
        mat=np.zeros((len(texts),max(len(vocab),1)),dtype="float32")
        for i,text in enumerate(texts):
            for token in self._tokens(text): mat[i,vocab[token]]+=1
        return mat/np.maximum(np.linalg.norm(mat,axis=1,keepdims=True),1e-8)
    def rebuild(self,chunks):
        self.chunks=chunks; self.matrix=self.embed([c.text for c in chunks]) if chunks else np.empty((0,1),dtype="float32"); self.index=None
        if faiss and chunks: self.index=faiss.IndexFlatIP(self.matrix.shape[1]); self.index.add(self.matrix)
    def search(self,query,k=4):
        if not self.chunks:return []
        q=self.embed([query])
        if self.index:
            scores,ids=self.index.search(q,min(k,len(self.chunks))); return [(self.chunks[i],float(scores[0][p])) for p,i in enumerate(ids[0]) if i>=0]
        scores=(self.matrix@q.T).ravel(); ids=np.argsort(-scores)[:k]; return [(self.chunks[i],float(scores[i])) for i in ids]
