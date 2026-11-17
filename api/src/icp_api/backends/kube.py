"""Async wrapper over the official Kubernetes client, which is synchronous."""

import asyncio
import contextlib
from collections.abc import AsyncIterator
from typing import Any

from kubernetes import client, config
from kubernetes.client.rest import ApiException

_KINDS = {
    "Deployment": ("apps", "namespaced_deployment"),
    "Service": ("core", "namespaced_service"),
}


class KubeOps:
    def __init__(self, namespace: str, context: str = ""):
        self.namespace = namespace
        self.context = context
        self._api: client.ApiClient | None = None

    def _client(self) -> client.ApiClient:
        # Loaded lazily so the API can start (and tests can run) without a cluster.
        if self._api is None:
            try:
                config.load_incluster_config()
                self._api = client.ApiClient()
            except config.ConfigException:
                self._api = config.new_client_from_config(context=self.context or None)
        return self._api

    def _apps(self) -> client.AppsV1Api:
        return client.AppsV1Api(self._client())

    def _core(self) -> client.CoreV1Api:
        return client.CoreV1Api(self._client())

    async def apply(self, manifest: dict[str, Any]) -> None:
        """Create the object, or patch it if it already exists."""
        await asyncio.to_thread(self._apply, manifest)

    def _apply(self, manifest: dict[str, Any]) -> None:
        group, suffix = _KINDS[manifest["kind"]]
        api = self._apps() if group == "apps" else self._core()
        try:
            getattr(api, f"create_{suffix}")(self.namespace, manifest)
        except ApiException as e:
            if e.status != 409:
                raise
            getattr(api, f"patch_{suffix}")(manifest["metadata"]["name"], self.namespace, manifest)

    async def read(self, name: str, selector: str) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        """The named Deployment and the pods matching `selector`, as camelCase dicts."""
        return await asyncio.to_thread(self._read, name, selector)

    def _read(self, name: str, selector: str) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        to_dict = self._client().sanitize_for_serialization
        try:
            deployment = to_dict(self._apps().read_namespaced_deployment(name, self.namespace))
        except ApiException as e:
            if e.status != 404:
                raise
            deployment = None
        pods = self._core().list_namespaced_pod(self.namespace, label_selector=selector).items
        return deployment, [to_dict(p) for p in pods]

    async def delete_selected(self, selector: str) -> None:
        """Delete the Deployments and Services matching `selector` (their pods follow)."""
        await asyncio.to_thread(self._delete_selected, selector)

    def _delete_selected(self, selector: str) -> None:
        self._apps().delete_collection_namespaced_deployment(
            self.namespace, label_selector=selector, propagation_policy="Background"
        )
        core = self._core()
        for svc in core.list_namespaced_service(self.namespace, label_selector=selector).items:
            self._ignore_missing(core.delete_namespaced_service, svc.metadata.name, self.namespace)

    async def delete_service(self, name: str) -> None:
        await asyncio.to_thread(self._ignore_missing, self._core().delete_namespaced_service, name, self.namespace)

    @staticmethod
    def _ignore_missing(fn, *args) -> None:
        try:
            fn(*args)
        except ApiException as e:
            if e.status != 404:
                raise

    async def pod_log(self, pod: str, previous: bool = False, tail: int = 200) -> str:
        def read() -> str:
            try:
                return self._core().read_namespaced_pod_log(pod, self.namespace, previous=previous, tail_lines=tail)
            except ApiException:
                return ""  # e.g. no previous container yet

        return await asyncio.to_thread(read)

    async def stream_logs(self, selector: str, follow: bool, tail: int) -> AsyncIterator[str]:
        core = self._core()
        pods = await asyncio.to_thread(lambda: core.list_namespaced_pod(self.namespace, label_selector=selector).items)
        if not pods:
            yield "[icp] no pods for this revision\n"
            return
        pod = pods[0].metadata.name
        if not follow:
            yield await self.pod_log(pod, tail=tail)
            return

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[str | None] = asyncio.Queue()
        resp = await asyncio.to_thread(
            core.read_namespaced_pod_log,
            pod,
            self.namespace,
            follow=True,
            tail_lines=tail,
            _preload_content=False,
        )

        def pump() -> None:
            try:
                for chunk in resp.stream(decode_content=True):
                    loop.call_soon_threadsafe(queue.put_nowait, chunk.decode(errors="replace"))
            except Exception:  # noqa: BLE001  (closed from the other side, or the pod went away)
                pass
            finally:
                with contextlib.suppress(RuntimeError):  # loop already closed
                    loop.call_soon_threadsafe(queue.put_nowait, None)

        loop.run_in_executor(None, pump)
        try:
            while (chunk := await queue.get()) is not None:
                yield chunk
        finally:
            resp.close()  # client went away: stop the reader thread
