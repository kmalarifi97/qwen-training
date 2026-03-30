"""Qwen Training Agent — runs on Colab/any GPU machine, connects to server via WebSocket."""

import asyncio
import logging
import signal
import sys
from pathlib import Path

from config import load_config
from connection import ServerConnection
from gpu_monitor import get_gpu_info
from job_runner import JobRunner


def setup_logging(log_dir: str):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    file_handler = logging.FileHandler(Path(log_dir) / "agent.log")
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(file_handler)
    root.addHandler(console_handler)


logger = logging.getLogger("agent")


class Agent:
    def __init__(self):
        self.config = load_config()
        setup_logging(self.config["log_dir"])

        self.connection = ServerConnection(
            server_url=self.config["server_url"],
            agent_name=self.config["agent_name"],
            api_key=self.config["api_key"],
        )
        self.job_runner = JobRunner(model_cache_dir=self.config["model_cache_dir"])
        self.heartbeat_interval = self.config["heartbeat_interval_seconds"]
        self._running = False
        self._working = False

        self.connection.on_job_received = self._on_job_received

    async def _on_job_received(self, job_data: dict):
        if self._working:
            logger.warning("Already working, rejecting job")
            return

        self._working = True
        job_id = job_data["job_id"]

        await self.connection.send_job_started(job_id)
        logger.info(f"Starting job {job_id[:8]}")

        async def progress_cb(jid, progress, **extra):
            await self.connection.send_job_progress(jid, progress, **extra)

        result = await self.job_runner.run_job(job_data, progress_cb)

        if result["status"] == "completed":
            await self.connection.send_job_completed(job_id, result.get("result", {}))
            logger.info(f"Job {job_id[:8]} completed")
        elif result["status"] == "failed":
            await self.connection.send_job_failed(job_id, result.get("error", ""))
            logger.error(f"Job {job_id[:8]} failed: {result.get('error')}")

        self._working = False

    async def run(self):
        logger.info("=" * 50)
        logger.info("Qwen Training Agent starting")
        logger.info(f"Agent: {self.config['agent_name']}")
        logger.info(f"Server: {self.config['server_url']}")

        gpu = get_gpu_info()
        if gpu:
            logger.info(f"GPU: {gpu['name']} ({gpu['vram_total_mb']}MB VRAM)")
        else:
            logger.warning("No GPU detected")
        logger.info("=" * 50)

        self._running = True
        await self.connection.connect()

        try:
            while self._running:
                gpu = get_gpu_info()
                if not self._working:
                    await self.connection.send_heartbeat(gpu)

                if not self.connection.connected:
                    logger.warning("Lost connection. Reconnecting...")
                    await self.connection.connect()

                await asyncio.sleep(self.heartbeat_interval)
        except asyncio.CancelledError:
            pass
        finally:
            await self.connection.disconnect()
            logger.info("Agent stopped")

    def stop(self):
        self._running = False


def main():
    agent = Agent()

    def handler(sig, frame):
        logger.info("Shutting down...")
        agent.stop()

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)

    try:
        asyncio.run(agent.run())
    except KeyboardInterrupt:
        agent.stop()


if __name__ == "__main__":
    main()
