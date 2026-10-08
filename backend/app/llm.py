import asyncio
from .config import settings
from .metrics import LLM_ERRORS
SYSTEM_PROMPT="You are an enterprise knowledge assistant. Answer only from supplied context. If context is insufficient, say so clearly. Cite sources as [1], [2], etc."
class LLM:
    def __init__(self):
        self.client=None
        if settings.openai_api_key:
            from openai import AsyncOpenAI
            self.client=AsyncOpenAI(api_key=settings.openai_api_key)
    def build_prompt(self,question,docs):
        context="\n\n".join(f"[{i+1}] {d.text}" for i,d in enumerate(docs)); return SYSTEM_PROMPT+"\n\nContext:\n"+context+"\n\nQuestion: "+question
    async def stream(self,question,docs):
        if not self.client:
            answer=self._mock(question,docs)
            for word in answer.split(): yield word+" "; await asyncio.sleep(0.01)
            return
        try:
            stream=await self.client.chat.completions.create(model=settings.openai_model,temperature=0,stream=True,messages=[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":self.build_prompt(question,docs)}])
            async for event in stream:
                delta=event.choices[0].delta.content if event.choices else None
                if delta: yield delta
        except Exception: LLM_ERRORS.inc(); raise
    def _mock(self,question,docs):
        if not docs:return "I could not find relevant evidence in the knowledge base."
        return "Based on the indexed evidence: "+" ".join(d.text for d in docs[:2])[:700]
