"""Dual-LLM router. Text with identifiers stays on the local model; only
de-identified text may go to Groq. If PHI is present and no local model is
available, we return None (caller uses a template). We NEVER fall back to Groq with PHI."""
import os, re

GROQ_URL = "https://api.groq.com/openai/v1"
_local_pipe = None


def scrub(text, names=()):
    rx = [(r"\b\d{10}\b", "[PHONE]"), (r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", "[DATE]"),
          (r"[\w.]+@[\w.]+\.\w+", "[EMAIL]"), (r"(?i)\bMRN\s*\d+\b", "[MRN]")]
    first = [n.split()[0] for n in names if n]
    if first:
        rx.append((r"\b(" + "|".join(map(re.escape, first)) + r")\b(\s[A-Z]\.)?", "[NAME]"))
    out, n = text, 0
    for pat, lab in rx:
        out, k = re.subn(pat, lab, out)
        n += k
    return out, n


def _groq(system, user, max_tokens=300):
    import httpx
    r = httpx.post(f"{GROQ_URL}/chat/completions", timeout=30,
                   headers={"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"},
                   json={"model": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"), "temperature": .2,
                         "max_tokens": max_tokens,
                         "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def _local(system, user, max_tokens=300):
    global _local_pipe
    model = os.getenv("HF_MODEL")
    if not model:
        return None
    try:
        if _local_pipe is None:
            from transformers import pipeline
            _local_pipe = pipeline("text-generation", model=model, token=os.getenv("HF_TOKEN") or None)
        out = _local_pipe([{"role": "system", "content": system}, {"role": "user", "content": user}],
                          max_new_tokens=max_tokens, do_sample=False)
        return out[0]["generated_text"][-1]["content"].strip()
    except Exception as e:
        print("local LLM unavailable:", e)
        return None


def generate(system, user, names=()):
    _, n = scrub(user, names)
    if n:
        return dict(text=_local(system, user), route="local-hf", phi_found=n)
    if os.getenv("GROQ_API_KEY"):
        try:
            return dict(text=_groq(system, user), route="groq", phi_found=0)
        except Exception as e:
            print("Groq call failed:", e)
    return dict(text=_local(system, user), route="local-hf", phi_found=0)
