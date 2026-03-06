import requests
import pandas as pd
import plotly.express as px
import time
import re
import signal
import os
from datetime import datetime

# --- 配置 ---
METRICS_URL = "http://127.0.0.1:8080/metrics"
SAMPLE_INTERVAL = 0.5 

# 数据存储
system_history = []    # 随时间变化 (Throughput, Queue)
request_history = []   # 随序号变化 (TTFT, ITL, TPOT)

last_snapshot = {
    "tokens": 0, "time": time.time(),
    "ttft_sum": 0, "ttft_cnt": 0,
    "itl_sum": 0, "itl_cnt": 0,
    "tpot_sum": 0, "tpot_cnt": 0
}
global_request_idx = 0
is_running = True

def parse_metrics(text):
    results = {}
    for line in text.split('\n'):
        if line.startswith('#') or not line: continue
        match = re.search(r'^([a-zA-Z0-9_:]+)(\{.*?\})?\s+([0-9.eE+-]+)', line)
        if match:
            name, _, value = match.groups()
            results[name] = float(value)
    return results

def signal_handler(sig, frame):
    global is_running
    is_running = False

signal.signal(signal.SIGINT, signal_handler)

def collect_step():
    global last_snapshot, global_request_idx
    try:
        r = requests.get(METRICS_URL, timeout=1)
        m = parse_metrics(r.text)
        curr_time = time.time()
        time_str = datetime.now().strftime("%H:%M:%S")

        # 1. 系统级数据 (基于时间)
        dt = curr_time - last_snapshot["time"]
        curr_tokens = m.get("vllm:generation_tokens_total", 0)
        tput = max(0, (curr_tokens - last_snapshot["tokens"]) / dt) if dt > 0 else 0
        
        system_history.append({
            "time": time_str,
            "Running": m.get("vllm:num_requests_running", 0),
            "Waiting": m.get("vllm:num_requests_waiting", 0),
            "Throughput": tput
        })

        # 2. 请求级数据 (基于序号)
        curr_cnt = int(m.get("vllm:time_to_first_token_seconds_count", 0))
        new_reqs = curr_cnt - last_snapshot["ttft_cnt"]

        if new_reqs > 0:
            delta_ttft = (m.get("vllm:time_to_first_token_seconds_sum", 0) - last_snapshot["ttft_sum"]) / new_reqs
            delta_tpot = (m.get("vllm:request_time_per_output_token_seconds_sum", 0) - last_snapshot["tpot_sum"]) / new_reqs
            delta_itl = (m.get("vllm:inter_token_latency_seconds_sum", 0) - last_snapshot["itl_sum"]) / new_reqs

            for _ in range(new_reqs):
                global_request_idx += 1
                request_history.append({
                    "Request_Index": global_request_idx,
                    "TTFT_ms": delta_ttft * 1000,
                    "TPOT_ms": delta_tpot * 1000,
                    "ITL_ms": delta_itl * 1000
                })

        last_snapshot.update({
            "tokens": curr_tokens, "time": curr_time,
            "ttft_sum": m.get("vllm:time_to_first_token_seconds_sum", 0),
            "ttft_cnt": curr_cnt,
            "itl_sum": m.get("vllm:inter_token_latency_seconds_sum", 0),
            "tpot_sum": m.get("vllm:request_time_per_output_token_seconds_sum", 0)
        })
    except: pass

def save_report():
    if not system_history:
        print("无数据记录。")
        return

    # 生成带时间戳的文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"vllm_report_{timestamp}.html"

    df_sys = pd.DataFrame(system_history)
    df_req = pd.DataFrame(request_history) if request_history else pd.DataFrame()
    
    html = f"<html><head><title>vLLM Report {timestamp}</title><style>body{{background:#121212;color:#eee;font-family:sans-serif;padding:20px;}}.card{{background:#1e1e1e;padding:15px;margin-bottom:20px;border-radius:10px;}}</style></head><body>"
    html += f"<h1 style='text-align:center;'>vLLM Hybrid Performance Report</h1>"
    html += f"<p style='text-align:center; color:#888;'>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Total Requests: {global_request_idx}</p>"

    # 绘制系统级曲线
    fig1 = px.line(df_sys, x="time", y=["Running", "Waiting"], title="System Load (Queues) over Time", template="plotly_dark")
    fig2 = px.area(df_sys, x="time", y="Throughput", title="Token Throughput (tok/s)", template="plotly_dark")
    
    html += f"<div class='card'>{fig1.to_html(full_html=False, include_plotlyjs='cdn')}</div>"
    html += f"<div class='card'>{fig2.to_html(full_html=False, include_plotlyjs='cdn')}</div>"

    # 绘制请求级散点图
    if not df_req.empty:
        fig3 = px.scatter(df_req, x="Request_Index", y="TTFT_ms", title="TTFT per Request ID", 
                          color="TTFT_ms", color_continuous_scale="Reds", template="plotly_dark")
        fig4 = px.scatter(df_req, x="Request_Index", y="TPOT_ms", title="TPOT per Request ID", template="plotly_dark")
        fig5 = px.scatter(df_req, x="Request_Index", y="ITL_ms", title="ITL per Request ID", template="plotly_dark")
        
        for f in [fig3, fig4, fig5]:
            f.update_traces(marker=dict(size=6, opacity=0.7))
            html += f"<div class='card'>{f.to_html(full_html=False, include_plotlyjs='cdn')}</div>"
    
    html += "</body></html>"
    with open(filename, "w") as f: f.write(html)
    print(f"\n[✔] 报告已保存至: {os.path.abspath(filename)}")

print(">>> 监控已启动 (混合模式 + 自动归档)")
print(">>> 1. 运行你的 Benchmark 命令")
print(">>> 2. 结束后在此窗口按 Ctrl+C 导出结果")

while is_running:
    collect_step()
    time.sleep(SAMPLE_INTERVAL)
save_report()