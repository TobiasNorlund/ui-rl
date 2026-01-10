import fastapi
import asyncio
from contextlib import asynccontextmanager
from io import BytesIO
from datetime import datetime
from fastapi import Response
from session import SimpleDataEntrySession
import computer


# Create session
session = SimpleDataEntrySession()


@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    await session.start()
    yield
    

app = fastapi.FastAPI(lifespan=lifespan)
start_time = datetime.now()


@app.get("/")
async def root():
    return Response(content=f"""UI Verifiers Session Server
Session Type: Simple Data Entry
Start time: {start_time}
Duration: {datetime.now() - start_time}
""", media_type="text/plain")


@app.get("/progress")
async def get_progress():
    """Get progress information"""
    progress = session.get_progress()
    return progress


@app.get("/act")
async def act(
    action_type: computer.ActionType,
    x: int | None = None,
    y: int | None = None,
    text: str | None = None,
    keys: str | None = None,
    direction: str | None = None,
    delay: float = 1.0,
):
    """Execute a computer action and return a screenshot"""
    # Prepare kwargs based on action type
    kwargs = {}
    if x is not None:
        kwargs["x"] = x
    if y is not None:
        kwargs["y"] = y
    if text is not None:
        kwargs["text"] = text
    if keys is not None:
        kwargs["keys"] = keys
    if direction is not None:
        kwargs["direction"] = direction

    res = await computer.act(type=action_type, **kwargs)
    if action_type == computer.ActionType.Screenshot:
        buf = BytesIO()
        res.save(buf, format="PNG")
        return Response(buf.getvalue(), media_type="image/png")
    
    await asyncio.sleep(delay)

    img = computer.screenshot()
    buf = BytesIO()
    img.save(buf, format="PNG")
    return Response(buf.getvalue(), media_type="image/png")
