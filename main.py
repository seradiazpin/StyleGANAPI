import uvicorn
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from routes import views

app = FastAPI()

# Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://thumbnailgenerator-c1e1b.web.app/", "http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(views.router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
#sudo gunicorn --bind :80 --keyfile server.key --certfile server.crt --ca-certs ca-crt.pem --cert-reqs 2 main:app -t 120
