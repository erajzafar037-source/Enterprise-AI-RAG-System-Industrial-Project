from pathlib import Path
from pypdf import PdfReader
from .config import settings
from .retriever import Chunk
def chunk_text(text,size=None,overlap=None):
    size=size or settings.chunk_size; overlap=overlap or settings.chunk_overlap; text=" ".join(text.split())
    if not text:return []
    chunks=[]; start=0
    while start<len(text):
        end=min(len(text),start+size); chunks.append(text[start:end])
        if end==len(text):break
        start=max(0,end-overlap)
    return chunks
def extract_pdf(path):
    reader=PdfReader(path); output=[]
    for page_no,page in enumerate(reader.pages,start=1):
        for piece in chunk_text(page.extract_text() or ""): output.append(Chunk(piece,Path(path).name,page_no))
    return output
