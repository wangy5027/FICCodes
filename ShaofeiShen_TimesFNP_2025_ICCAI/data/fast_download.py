import os
import sys
import time
import threading
import urllib.request

URL = "https://dachxiu.chicagobooth.edu/download/datashare.zip"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "raw", "gkx_datashare.zip")   # 放置于本脚本同级的 raw/ 下
PROXY = None   # 如需代理, 设为 "http://host:port"; 默认不使用
N_THREADS = 24
CHUNK = 256 * 1024

def make_opener():
    if PROXY:
        return urllib.request.build_opener(
            urllib.request.ProxyHandler({"http": PROXY, "https": PROXY}))
    return urllib.request.build_opener()

def get_size():
    req = urllib.request.Request(URL, method="HEAD")
    with make_opener().open(req, timeout=60) as r:
        return int(r.headers["Content-Length"])

def download_part(idx, start, end, part_path, done_flag):
    have = os.path.getsize(part_path) if os.path.exists(part_path) else 0
    want = end - start + 1
    if have >= want:
        done_flag[idx] = True
        return
    opener = make_opener()
    while have < want:
        try:
            req = urllib.request.Request(URL)
            req.add_header("Range", f"bytes={start + have}-{end}")
            with opener.open(req, timeout=120) as r, open(part_path, "ab") as f:
                while True:
                    buf = r.read(CHUNK)
                    if not buf:
                        break
                    f.write(buf)
                    have += len(buf)
                    if have >= want:
                        break
        except Exception as e:
            print(f"[part {idx}] retry after error: {e}", flush=True)
            time.sleep(5)
    done_flag[idx] = True

def main():
    total = get_size()
    print("total bytes:", total, flush=True)
    tmpdir = OUT + ".parts"
    os.makedirs(tmpdir, exist_ok=True)
    seg = (total + N_THREADS - 1) // N_THREADS
    threads, done = [], [False] * N_THREADS
    t0 = time.time()
    for i in range(N_THREADS):
        s, e = i * seg, min((i + 1) * seg - 1, total - 1)
        if s > e:
            done[i] = True
            continue
        t = threading.Thread(target=download_part,
                             args=(i, s, e, os.path.join(tmpdir, f"part{i:03d}"), done),
                             daemon=True)
        t.start()
        threads.append(t)
    while not all(done):
        time.sleep(30)
        got = sum(os.path.getsize(os.path.join(tmpdir, f))
                  for f in os.listdir(tmpdir))
        rate = got / max(time.time() - t0, 1) / 1024
        print(f"progress: {got/1e6:.0f}/{total/1e6:.0f} MB  avg {rate:.0f} KB/s", flush=True)
    for t in threads:
        t.join()
    with open(OUT, "wb") as out:
        for i in range(N_THREADS):
            p = os.path.join(tmpdir, f"part{i:03d}")
            if os.path.exists(p):
                with open(p, "rb") as f:
                    while True:
                        b = f.read(8 * 1024 * 1024)
                        if not b:
                            break
                        out.write(b)
    assert os.path.getsize(OUT) == total, "size mismatch!"
    print("DONE:", OUT, os.path.getsize(OUT), flush=True)

if __name__ == "__main__":
    main()
