#!/usr/bin/env python3
"""Reproduce the V11 32K shared-prefix burst shape and measure decode TPS.

The generated coding reference is deterministic, but is not the private V11
prompt. Record both per-request streamed-content generation rates and pooled
burst output throughput so the two distinct metrics cannot be confused.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import statistics
import threading
import time
import urllib.request


def post(opener, url, body, timeout):
    request = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    return opener.open(request, timeout=timeout)


def make_prompt(repetitions):
    reference = (
        "# Reference: weighted interval scheduling\n"
        "# Jobs have start, end, and value fields. Sort by end. For each job i, "
        "binary-search the last compatible job p(i). The recurrence is "
        "dp[i] = max(dp[i-1], value[i] + dp[p(i)]). Prefer the lexicographically "
        "smaller sequence of original indices when values tie. An empty "
        "schedule has value zero. Include input validation for invalid intervals.\n"
        "def predecessor(ends, start):\n"
        "    lo, hi = 0, len(ends)\n"
        "    while lo < hi:\n"
        "        mid = (lo + hi) // 2\n"
        "        if ends[mid] <= start: lo = mid + 1\n"
        "        else: hi = mid\n"
        "    return lo - 1\n"
    )
    return (
        "Use this reference material when answering the final coding task.\n"
        + reference * repetitions
        + "\nWrite a Python 3 implementation of weighted interval scheduling. "
        "Include the function, type hints, and a brief example.\n"
    )


def tokenize_count(opener, base_url, prompt):
    with post(
        opener,
        base_url + "/tokenize",
        {"model": "deepseek-v4.1-flash", "messages": [{"role": "user", "content": prompt}]},
        60,
    ) as response:
        return json.load(response)["count"]


def run_one(opener, url, prefix, index, barrier, timeout):
    prompt = prefix + f"\nRequest variant {index}: return the implementation now."
    body = {
        "model": "deepseek-v4.1-flash",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 512,
        "ignore_eos": True,
        "chat_template_kwargs": {"thinking": False},
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    barrier.wait(timeout=60)
    start = time.monotonic()
    first = last = None
    content = []
    usage = None
    finish_reason = None
    done = False
    with post(opener, url + "/v1/chat/completions", body, timeout) as response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                done = True
                break
            event = json.loads(payload)
            if event.get("error"):
                raise RuntimeError(str(event["error"]))
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                chunk = choice.get("delta", {}).get("content")
                if chunk:
                    now = time.monotonic()
                    first = now if first is None else first
                    last = now
                    content.append(chunk)
                if choice.get("finish_reason") is not None:
                    finish_reason = choice["finish_reason"]
    end = time.monotonic()
    if not done or not usage or not content or finish_reason != "length":
        raise RuntimeError(f"Incomplete request {index}: {done=} {usage=} {finish_reason=}")
    tokens = usage["completion_tokens"]
    if tokens != 512 or last <= first:
        raise RuntimeError(f"Unexpected output {index}: {tokens=}, {first=}, {last=}")
    return {
        "index": index,
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": tokens,
        "finish_reason": finish_reason,
        "content_sha256": hashlib.sha256("".join(content).encode()).hexdigest(),
        "first_content_from_start_s": first - start,
        "generation_s": last - first,
        "decode_tps": (tokens - 1) / (last - first),
        "start_offset_s": start,
        "first_offset_s": first,
        "last_offset_s": last,
        "end_offset_s": end,
    }


def run_burst(opener, url, prefix, concurrency, timeout):
    barrier = threading.Barrier(concurrency + 1)
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(run_one, opener, url, prefix, i, barrier, timeout)
            for i in range(concurrency)
        ]
        barrier.wait(timeout=60)
        results = [future.result() for future in futures]
    first = min(item["first_offset_s"] for item in results)
    last = max(item["last_offset_s"] for item in results)
    start = min(item["start_offset_s"] for item in results)
    end = max(item["end_offset_s"] for item in results)
    speeds = [item["decode_tps"] for item in results]
    for item in results:
        for field in ("start_offset_s", "first_offset_s", "last_offset_s", "end_offset_s"):
            item[field] -= start
    return {
        "concurrency": concurrency,
        "requests": results,
        "min_request_decode_tps": min(speeds),
        "median_request_decode_tps": statistics.median(speeds),
        "aggregate_decode_tps": sum(item["completion_tokens"] - 1 for item in results) / (last - first),
        "pooled_burst_output_tps": sum(item["completion_tokens"] for item in results) / (end - start),
        "decode_window_s": last - first,
        "batch_wall_s": end - start,
        "prompt_token_counts": sorted(set(item["prompt_tokens"] for item in results)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://10.8.2.1:31818")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()
    url = args.url.rstrip("/")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    # Calibrate using the running model tokenizer, keeping the 32K shared-prefix
    # shape of the mainline V11 experiment without depending on a private prompt.
    lo, hi = 1, 256
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if tokenize_count(opener, url, make_prompt(mid)) <= 32768:
            lo = mid
        else:
            hi = mid - 1
    prefix = make_prompt(lo)
    prefix_tokens = tokenize_count(opener, url, prefix)
    metadata = {
        "workload": "V11-shape 32K shared-prefix weighted-interval coding, fixed 512 output",
        "mainline_prompt_identical": False,
        "single_node_no_pd": True,
        "prompt_sha256": hashlib.sha256(prefix.encode()).hexdigest(),
        "reference_repetitions": lo,
        "prefix_token_count": prefix_tokens,
        "warmups_per_concurrency": args.warmups,
        "measured_rounds_per_concurrency": args.rounds,
        "concurrency": args.concurrency,
        "decode_tps_definition": "(completion_tokens - 1)/(last_content - first_content)",
        "aggregate_decode_tps_definition": "sum(completion_tokens - 1)/(last_content_any - first_content_any)",
    }
    report = {"metadata": metadata, "rounds": [], "summary": {}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for concurrency in args.concurrency:
        measured = []
        for round_index in range(-args.warmups, args.rounds):
            result = run_burst(opener, url, prefix, concurrency, args.timeout)
            result["round"] = round_index
            result["warmup"] = round_index < 0
            report["rounds"].append(result)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({
                "concurrency": concurrency,
                "round": round_index,
                "aggregate_decode_tps": result["aggregate_decode_tps"],
                "min_request_decode_tps": result["min_request_decode_tps"],
                "median_request_decode_tps": result["median_request_decode_tps"],
                "prompt_token_counts": result["prompt_token_counts"],
            }), flush=True)
            if round_index >= 0:
                measured.append(result)
        all_requests = [item for result in measured for item in result["requests"]]
        total_tokens = sum(sum(item["completion_tokens"] for item in result["requests"]) for result in measured)
        total_wall = sum(result["batch_wall_s"] for result in measured)
        report["summary"][str(concurrency)] = {
            "requests": len(all_requests),
            "min_request_decode_tps": min(item["decode_tps"] for item in all_requests),
            "median_request_decode_tps": statistics.median(item["decode_tps"] for item in all_requests),
            "median_aggregate_decode_tps": statistics.median(result["aggregate_decode_tps"] for result in measured),
            "pooled_burst_output_tps": total_tokens / total_wall,
            "prompt_token_counts": sorted(set(item["prompt_tokens"] for item in all_requests)),
        }
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"metadata": metadata, "summary": report["summary"]}), flush=True)


if __name__ == "__main__":
    main()
