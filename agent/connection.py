"""WebSocket connection to the Qwen Training Engine server."""

import asyncio
import json
import logging
import uuid

import websockets

logger = logging.getLogger("agent.connection")


class ServerConnection:
    def __init__(self, server_url: str, agent_name: str, api_key: str = ""):
        self.server_url = server_url
        self.agent_name = agent_name
        self.api_key = api_key
        self.agent_id = f"{agent_name}_{uuid.uuid4().hex[:8]}"
        self._ws = None
        self._connected = False
        self.on_job_received = None  # async callback(job_data)

    @property
    def connected(self) -> bool:
        return self._connected and self._ws is not None

    async def connect(self):
        """Connect with exponential backoff."""
        delay = 5
        while True:
            try:
                logger.info(f"Connecting to {self.server_url}")
                self._ws = await websockets.connect(
                    self.server_url,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5,
                )

                # Register
                await self._ws.send(json.dumps({
                    "type": "register",
                    "agent_id": self.agent_id,
                    "agent_name": self.agent_name,
                    "api_key": self.api_key,
                }))

                raw = await self._ws.recv()
                data = json.loads(raw)

                if data.get("type") == "registered":
                    self._connected = True
                    logger.info(f"Connected. Agent ID: {self.agent_id}")
                    return
                else:
                    logger.error(f"Registration failed: {data}")
                    await self._ws.close()
                    self._ws = None

            except Exception as e:
                logger.warning(f"Connection failed: {e}. Retrying in {delay}s...")
                self._ws = None
                self._connected = False
                await asyncio.sleep(delay)
                delay = min(delay * 2, 60)

    async def send_heartbeat(self, gpu_info: dict | None):
        """Send heartbeat. Server may respond with a job assignment."""
        if not self.connected:
            return False
        try:
            await self._ws.send(json.dumps({
                "type": "heartbeat",
                "agent_id": self.agent_id,
                "state": "AVAILABLE",
                "gpu_info": gpu_info,
            }))

            raw = await self._ws.recv()
            data = json.loads(raw)

            if data.get("type") == "job_assign":
                logger.info(f"Received job: {data.get('job_id', '')[:8]}")
                if self.on_job_received:
                    await self.on_job_received(data)
            return True

        except Exception as e:
            logger.warning(f"Heartbeat failed: {e}")
            self._connected = False
            self._ws = None
            return False

    async def send_job_started(self, job_id: str):
        await self._send({"type": "job_started", "job_id": job_id})

    async def send_job_progress(self, job_id: str, progress: str, **extra):
        msg = {"type": "job_progress", "job_id": job_id, "progress": progress}
        msg.update(extra)
        await self._send(msg)

    async def send_job_completed(self, job_id: str, result: dict):
        await self._send({"type": "job_completed", "job_id": job_id, "result": json.dumps(result)})

    async def send_job_failed(self, job_id: str, error: str):
        await self._send({"type": "job_failed", "job_id": job_id, "error": error})

    async def _send(self, data: dict):
        if not self.connected:
            return
        try:
            await self._ws.send(json.dumps(data))
        except Exception as e:
            logger.warning(f"Send failed: {e}")
            self._connected = False
            self._ws = None

    async def disconnect(self):
        if self._ws:
            await self._ws.close()
            self._ws = None
            self._connected = False
            logger.info("Disconnected")
