from pydantic import BaseModel, Field
class QueryRequest(BaseModel):
    question: str = Field(min_length=2,max_length=4000)
    top_k: int | None = Field(default=None,ge=1,le=10)
