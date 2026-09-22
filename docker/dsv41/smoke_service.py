"""Small HTTP correctness and DSpark activity gate for a running service.

Graph replay must additionally be confirmed from the scheduler log. This is
not a throughput benchmark or a full model quality evaluation.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:31818")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(path, payload=None):
        data = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(
            args.url.rstrip("/") + path,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with opener.open(req, timeout=600) as response:
            return json.load(response)

    def generate(index):
        expected = str(13001 + index)
        prompt = (
            f"The verification code for request {index} is {expected}. "
            "Reply with only that five-digit verification code."
        )
        # Unequal prompt lengths exercise changing request metadata at replay.
        prompt = ("This is an independent verification request. " * index) + prompt
        result = request(
            "/v1/chat/completions",
            {
                "model": "deepseek-v4.1-flash",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": 128,
                "chat_template_kwargs": {"thinking": False},
            },
        )
        choice = result["choices"][0]
        content = choice["message"].get("content") or ""
        assert expected in re.findall(r"\b\d{5}\b", content), result
        assert choice["finish_reason"] == "stop", result
        return {"request": index, "expected": expected, "response": result}

    before = request("/server_info")
    states = before["internal_states"]
    assert states, before
    for state in states:
        assert state["speculative_algorithm"] == "DSPARK", state
        assert not state["disable_cuda_graph"], state
        assert state["disaggregation_mode"] == "null", state

    responses = [generate(0)]
    for count in (4, 8):
        with ThreadPoolExecutor(max_workers=count) as pool:
            responses.extend(pool.map(generate, range(1, count + 1)))
    # A longer response exercises several draft/verify graph replays and makes
    # the periodic scheduler graph/acceptance log observable.
    sequence = request(
        "/v1/chat/completions",
        {
            "model": "deepseek-v4.1-flash",
            "messages": [
                {
                    "role": "user",
                    "content": "List every integer from 1 to 64 inclusive, in order, "
                    "separated by commas. Output only the list.",
                }
            ],
            "temperature": 0,
            "max_tokens": 512,
            "chat_template_kwargs": {"thinking": False},
        },
    )
    sequence_choice = sequence["choices"][0]
    numbers = re.findall(r"\d+", sequence_choice["message"].get("content") or "")
    assert list(map(int, numbers)) == list(range(1, 65)), sequence
    assert sequence_choice["finish_reason"] == "stop", sequence
    responses.append({"request": "sequence_1_to_64", "response": sequence})
    after = request("/server_info")
    accepts = [s.get("avg_spec_accept_length", 0) for s in after["internal_states"]]
    assert max(accepts) > 1, "No accepted DSpark draft tokens observed"
    report = {
        "requests_passed": len(responses),
        "concurrency": [1, 4, 8],
        "avg_spec_accept_length": accepts,
        "server_info": after,
        "responses": responses,
        "graph_replay_requires_scheduler_log_confirmation": True,
    }
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("server_info", "responses")}
        )
    )


if __name__ == "__main__":
    main()
