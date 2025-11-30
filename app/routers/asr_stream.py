import asyncio
import base64
import io
import json
import logging
import os
import uuid
from typing import Optional

from websockets.client import connect
from fastapi import APIRouter, File, UploadFile, WebSocket, WebSocketDisconnect
from pydub import AudioSegment
from starlette.websockets import WebSocketState

logger = logging.getLogger(__name__)

DASHSCOPE_WS = "wss://dashscope.aliyuncs.com/api-ws/v1/inference"
API_KEY = os.getenv("DASHSCOPE_API_KEY")
MODEL = "paraformer-realtime-v2"


async def _run_dashscope_session(pcm_bytes: bytes) -> str:
    if not API_KEY:
        raise RuntimeError("DASHSCOPE_API_KEY missing")

    headers = {"Authorization": f"Bearer {API_KEY}", "user-agent": "one-gateway-asr"}
    task_id = uuid.uuid4().hex
    collected = []

    async with connect(DASHSCOPE_WS, extra_headers=headers) as ds:
        run_task = {
            "header": {"action": "run-task", "task_id": task_id, "streaming": "duplex"},
            "payload": {
                "task_group": "audio",
                "task": "asr",
                "function": "recognition",
                "model": MODEL,
                "parameters": {
                    "format": "pcm",
                    "sample_rate": 16000,
                    "disfluency_removal_enabled": False,
                    "punctuation_prediction_enabled": True,
                },
                "input": {},
            },
        }
        await ds.send(json.dumps(run_task))

        # 等待 task-started
        while True:
            msg = await ds.recv()
            payload = json.loads(msg)
            event = payload.get("header", {}).get("event")
            if event == "task-started":
                break

        # 发送音频（binary）
        await ds.send(pcm_bytes)

        # 发送 finish-task
        finish = {
            "header": {"action": "finish-task", "task_id": task_id, "streaming": "duplex"},
            "payload": {"input": {}},
        }
        await ds.send(json.dumps(finish))

        # 收集结果
        async for msg in ds:
            payload = json.loads(msg)
            header = payload.get("header", {})
            event = header.get("event")
            if event == "result-generated":
                sentence = payload.get("payload", {}).get("output", {}).get("sentence", {})
                if sentence:
                    collected.append(sentence.get("text", ""))
            if event == "task-finished":
                break

    return "".join(collected).strip()


async def relay_stream(ws: WebSocket):
    if not API_KEY:
        await ws.send_json({"type": "error", "message": "DASHSCOPE_API_KEY missing"})
        return

    headers = {"Authorization": f"Bearer {API_KEY}", "user-agent": "one-gateway-asr"}
    task_id = uuid.uuid4().hex

    async with connect(DASHSCOPE_WS, extra_headers=headers) as ds:
        run_task = {
            "header": {"action": "run-task", "task_id": task_id, "streaming": "duplex"},
            "payload": {
                "task_group": "audio",
                "task": "asr",
                "function": "recognition",
                "model": MODEL,
                "parameters": {"format": "pcm", "sample_rate": 16000},
                "input": {},
            },
        }
        await ds.send(json.dumps(run_task))

        # 等待 task-started
        while True:
            msg = await ds.recv()
            payload = json.loads(msg)
            if payload.get("header", {}).get("event") == "task-started":
                await ws.send_json(payload)
                break

        async def from_client():
            try:
                while True:
                    msg = await ws.receive_text()
                    data = json.loads(msg)
                    if data.get("type") == "stop":
                        finish = {
                            "header": {"action": "finish-task", "task_id": task_id, "streaming": "duplex"},
                            "payload": {"input": {}},
                        }
                        await ds.send(json.dumps(finish))
                        break
                    audio_b64 = data.get("audio")
                    if audio_b64:
                        await ds.send(base64.b64decode(audio_b64))
            except WebSocketDisconnect:
                pass

        async def from_dashscope():
            async for resp in ds:
                try:
                    payload = json.loads(resp)
                except Exception:
                    continue
                await ws.send_json(payload)
                if payload.get("header", {}).get("event") == "task-finished":
                    break

        await asyncio.gather(from_client(), from_dashscope())


router = APIRouter(prefix="/cn/v1", tags=["asr"])


@router.post("/asr_once")
async def asr_once(file: UploadFile = File(...)):
    try:
        raw = await file.read()
        audio = AudioSegment.from_file(io.BytesIO(raw))
        pcm = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2).raw_data
        text = await _run_dashscope_session(pcm)
        return {"text": text}
    except Exception as e:
        logger.exception("asr_once error")
        return {"text": "", "error": str(e)}


@router.websocket("/asr_stream")
async def asr_stream(ws: WebSocket):
    await ws.accept()
    try:
        await relay_stream(ws)
    except Exception as e:
        logger.exception("asr_stream error")
        if ws.application_state == WebSocketState.CONNECTED:
            await ws.send_json({"type": "error", "message": str(e)})
    finally:
        if ws.application_state == WebSocketState.CONNECTED:
            await ws.close()
