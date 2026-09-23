"""Send a few real requests through the FlagCX PD router and save a receipt."""

import concurrent.futures
import json
from pathlib import Path
import time
from urllib import error, request


URL = "http://127.0.0.1:31821/v1/chat/completions"
OPENER = request.build_opener(request.ProxyHandler({}))
PROMPTS = (
    "Write a Python function that returns the first ten Fibonacci numbers.\n",
    "Explain in three sentences why a hash table lookup is usually fast.\n",
    "Calculate 17 * 23 and show one short verification step.\n",
    "Name the capital of China and the capital of the United States.\n",
)


def run_one(prompt):
    payload = json.dumps({
        "model": "deepseek-v4.1-flash",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 128,
        "chat_template_kwargs": {"thinking": False},
    }).encode()
    started = time.perf_counter()
    req = request.Request(URL, payload, {"Content-Type": "application/json"})
    try:
        with OPENER.open(req, timeout=180) as response:
            status = response.status
            data = json.load(response)
    except error.HTTPError as exc:
        raise RuntimeError(
            f"PD router returned HTTP {exc.code}: {exc.read(2048).decode(errors='replace')}"
        ) from exc
    assert status == 200, status
    assert data.get("choices"), data
    content = data["choices"][0].get("message", {}).get("content")
    assert content, data
    return {
        "http_status": status,
        "latency_s": round(time.perf_counter() - started, 3),
        "id": data.get("id"),
        "finish_reason": data["choices"][0].get("finish_reason"),
        "usage": data.get("usage"),
        "text_prefix": content[:160],
    }


def main():
    warmup = run_one(PROMPTS[0])
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        parallel_results = list(pool.map(run_one, PROMPTS[1:]))
    result = {"url": URL, "warmup": warmup, "concurrent": parallel_results}
    target = Path("/work/results/pd-smoke.json")
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
