"""Runs Slurm commands on the cluster login node over one reused SSH connection."""

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import asyncssh


@dataclass
class CommandResult:
    exit_status: int
    stdout: str
    stderr: str


class Runner(Protocol):
    async def run(self, command: str, input: str | None = None) -> CommandResult: ...

    def stream(self, command: str) -> AsyncIterator[str]: ...


class SSHRunner:
    """Needs key-based login (no password or Duo prompt), since it runs unattended."""

    def __init__(self, host: str, user: str, key_path: str, known_hosts_path: str):
        self.host = host
        self.user = user
        self.key_path = str(Path(key_path).expanduser())
        self.known_hosts_path = str(Path(known_hosts_path).expanduser())
        self._conn: asyncssh.SSHClientConnection | None = None
        self._lock = asyncio.Lock()

    async def _connection(self) -> asyncssh.SSHClientConnection:
        async with self._lock:
            if self._conn is None:
                self._conn = await asyncssh.connect(
                    self.host,
                    username=self.user,
                    client_keys=[self.key_path],
                    known_hosts=self.known_hosts_path,
                    keepalive_interval=30,
                )
            return self._conn

    async def run(self, command: str, input: str | None = None) -> CommandResult:
        for attempt in (1, 2):  # one reconnect if the cached connection has dropped
            conn = await self._connection()
            try:
                result = await conn.run(command, input=input, check=False)
            except (asyncssh.Error, OSError):
                self._conn = None
                if attempt == 2:
                    raise
                continue
            exit_status = result.exit_status if result.exit_status is not None else -1
            return CommandResult(exit_status, str(result.stdout or ""), str(result.stderr or ""))
        raise AssertionError("unreachable")

    async def stream(self, command: str) -> AsyncIterator[str]:
        conn = await self._connection()
        async with conn.create_process(command) as process:
            async for line in process.stdout:
                yield line

    async def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
