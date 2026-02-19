from fastapi import FastAPI
from prometheus_client import Counter, generate_latest
from fastapi.responses import Response

app = FastAPI(title="Agentic Support Triage API")

tickets_counter = Counter("tickets_total", "Total number of tickets received")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/tickets")
def create_ticket(ticket: dict):
    tickets_counter.inc()
    return {
        "message": "Ticket received",
        "ticket": ticket
    }

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")
