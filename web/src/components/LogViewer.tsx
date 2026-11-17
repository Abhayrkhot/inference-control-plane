import { useEffect, useRef, useState } from "react";

const MAX_CHARS = 200_000;

export default function LogViewer({ name }: { name: string }) {
  const [text, setText] = useState("");
  const [following, setFollowing] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const boxRef = useRef<HTMLPreElement>(null);

  async function load(follow: boolean) {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    setText("");
    setFollowing(follow);
    try {
      const response = await fetch(`/api/deployments/${name}/logs?tail=300&follow=${follow}`, {
        signal: controller.signal,
      });
      if (!response.ok || !response.body) {
        setText(`error ${response.status}: ${await response.text()}`);
        return;
      }
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        setText((t) => (t + chunk).slice(-MAX_CHARS));
      }
    } catch (e) {
      if ((e as Error).name !== "AbortError") setText((t) => `${t}\n[error] ${(e as Error).message}`);
    } finally {
      if (abortRef.current === controller) setFollowing(false);
    }
  }

  useEffect(() => () => abortRef.current?.abort(), []);
  useEffect(() => {
    const box = boxRef.current;
    if (box) box.scrollTop = box.scrollHeight;
  }, [text]);

  return (
    <section className="panel">
      <div className="panel-head">
        <h2>Logs</h2>
        <div className="actions">
          <button onClick={() => load(false)}>Load</button>
          {following ? (
            <button onClick={() => abortRef.current?.abort()}>Stop</button>
          ) : (
            <button onClick={() => load(true)}>Follow</button>
          )}
        </div>
      </div>
      <pre ref={boxRef} className="logs">
        {text || "Logs of the serving revision (or the latest one if nothing is serving)."}
      </pre>
    </section>
  );
}
