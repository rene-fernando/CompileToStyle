from fastapi import FastAPI
from fastapi.responses import FileResponse


app = FastAPI()

@app.get("/image")
async def get_image():
    filepath="./shirts/g.png"
    return FileResponse("hello world")
