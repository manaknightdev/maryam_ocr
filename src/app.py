# app.py
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from src.services.vector_store import VectorStore
import uvicorn
import asyncio

app = FastAPI()
templates = Jinja2Templates(directory="templates")
vector_store = VectorStore()

@app.on_event("startup")
async def startup_event():
    await vector_store.initialize()

@app.get("/", response_class=HTMLResponse)
async def form_get(request: Request):
    return templates.TemplateResponse("search.html", {"request": request, "results": None})

@app.post("/", response_class=HTMLResponse)
async def form_post(request: Request, query: str = Form(...)):
    results = await vector_store.search(query, threshold=0.2)
    return templates.TemplateResponse("search.html", {"request": request, "results": results})

if __name__ == "__main__":
    uvicorn.run("app:app", reload=True)
