"""KDIGO retrieval. FAISS + MiniLM when installed, keyword overlap otherwise."""
import os, re, glob


class KB:
    def __init__(self, folder):
        self.chunks = []
        for f in sorted(glob.glob(os.path.join(folder, "*.md"))):
            for para in re.split(r"\n\s*\n", open(f, encoding="utf-8").read()):
                para = para.strip()
                if len(para) > 40 and not para.startswith("#"):
                    self.chunks.append((os.path.basename(f), para))
        self.index = self.enc = None
        try:
            import faiss, numpy as np
            from sentence_transformers import SentenceTransformer
            self.enc = SentenceTransformer(os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
            emb = self.enc.encode([c[1] for c in self.chunks], normalize_embeddings=True)
            self.index = faiss.IndexFlatIP(emb.shape[1])
            self.index.add(np.asarray(emb, dtype="float32"))
        except Exception:
            pass  # keyword mode

    @property
    def mode(self):
        return "faiss+minilm" if self.index is not None else "keyword"

    def search(self, query, k=3):
        if not self.chunks:
            return []
        if self.index is not None:
            import numpy as np
            q = self.enc.encode([query], normalize_embeddings=True)
            _, ids = self.index.search(np.asarray(q, dtype="float32"), k)
            return [self.chunks[i] for i in ids[0] if i >= 0]
        w = set(re.findall(r"[a-z0-9]+", query.lower()))
        scored = sorted(self.chunks, key=lambda c: -len(w & set(re.findall(r"[a-z0-9]+", c[1].lower()))))
        return scored[:k]
