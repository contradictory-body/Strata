#!/usr/bin/env python3
"""
ws_test.py — WebSocket 功能测试脚本
=====================================
测试：
  1. 连接建立与 connected 事件
  2. 发送消息，接收流式 token
  3. 多端同步（同一 session_id 两个连接）
  4. Ping/Pong 心跳
  5. 工具调用事件（tool_start / tool_end）

依赖：pip install websockets requests
运行：python3 ws_test.py
"""

import asyncio
import json
import sys
import requests
import websockets

BASE_HTTP = "http://localhost:8000"
BASE_WS   = "ws://localhost:8000"

# ── 颜色 ─────────────────────────────────────────────────────
GREEN  = "\033[32m"; RED  = "\033[31m"; CYAN = "\033[36m"
YELLOW = "\033[33m"; BOLD = "\033[1m";  RESET = "\033[0m"

def ok(msg):   print(f"  {GREEN}✓{RESET}  {msg}")
def fail(msg): print(f"  {RED}✗{RESET}  {msg}")
def info(msg): print(f"  {CYAN}→{RESET}  {msg}")
def head(msg): print(f"\n{BOLD}{CYAN}【{msg}】{RESET}")


def register_and_login(username: str, password: str) -> tuple[str, str]:
    """注册用户并登录，返回 (token, user_id)"""
    # 尝试注册，忽略已存在错误
    requests.post(f"{BASE_HTTP}/api/auth/register", json={
        "username": username, "email": f"{username}@test.dev", "password": password
    })
    # 登录
    r = requests.post(f"{BASE_HTTP}/api/auth/login", json={
        "username": username, "password": password
    })
    d = r.json()
    return d["access_token"], str(d["user_id"])


def create_session(token: str) -> str:
    """创建新会话，返回 session_id"""
    r = requests.post(f"{BASE_HTTP}/api/sessions",
        headers={"Authorization": f"Bearer {token}"},
        json={}
    )
    return r.json()["id"]


# ── 测试 1：基础连接 ──────────────────────────────────────────
async def test_basic_connection():
    head("测试 1：WebSocket 基础连接")
    token = register_and_login("ws_test_user", "Test1234!")[0]
    session_id = create_session(token)
    info(f"Session: {session_id[:8]}...")

    uri = f"{BASE_WS}/ws/{session_id}?token={token}"
    try:
        async with websockets.connect(uri, open_timeout=10) as ws:
            # 期望收到 connected 事件
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            if msg.get("type") == "connected":
                ok("收到 connected 事件")
                ok(f"session_id 正确: {msg['data']['session_id'][:8]}...")
            else:
                fail(f"期望 connected，实际收到: {msg}")
    except Exception as e:
        fail(f"连接失败: {e}")


# ── 测试 2：发送消息接收流式 token ───────────────────────────
async def test_streaming():
    head("测试 2：发送消息，接收流式 token")
    token = register_and_login("ws_test_user", "Test1234!")[0]
    session_id = create_session(token)

    uri = f"{BASE_WS}/ws/{session_id}?token={token}"
    try:
        async with websockets.connect(uri, open_timeout=10) as ws:
            # 等待 connected
            await asyncio.wait_for(ws.recv(), timeout=5)

            # 发送消息
            await ws.send(json.dumps({
                "type": "message",
                "content": "你好，请用一句话介绍你自己"
            }))
            info("消息已发送，等待响应...")

            events_received = []
            total_tokens = 0
            assistant_text = ""

            # 收集事件（最多等待 30 秒）
            try:
                while True:
                    raw = await asyncio.wait_for(ws.recv(), timeout=30)
                    event = json.loads(raw)
                    t = event.get("type")
                    events_received.append(t)

                    if t == "user_message":
                        ok("收到 user_message 广播")
                    elif t == "token":
                        total_tokens += 1
                        assistant_text += event.get("data", "")
                        if total_tokens == 1:
                            ok("收到第一个 token（流式开始）")
                        print(f"    {event['data']}", end="", flush=True)
                    elif t == "tool_start":
                        ok(f"工具开始: {event['data']['name']}")
                    elif t == "tool_end":
                        ok(f"工具完成: {event['data']['name']}")
                    elif t == "clarify":
                        ok(f"收到澄清反问: {event['data'][:40]}...")
                    elif t == "done":
                        print()  # 换行
                        ok(f"收到 done 事件（共 {total_tokens} 个 token）")
                        break
                    elif t == "error":
                        fail(f"收到错误: {event['data']}")
                        break

            except asyncio.TimeoutError:
                fail("等待响应超时（30s）")

            if total_tokens > 0:
                ok(f"流式输出正常，总 token 数: {total_tokens}")
                info(f"回复预览: {assistant_text[:80]}...")

    except Exception as e:
        fail(f"流式测试失败: {e}")
        import traceback; traceback.print_exc()


# ── 测试 3：多端同步 ──────────────────────────────────────────
async def test_multi_device_sync():
    head("测试 3：多端同步（两个 WebSocket 连接同一 session）")
    token = register_and_login("ws_test_user", "Test1234!")[0]
    session_id = create_session(token)
    uri = f"{BASE_WS}/ws/{session_id}?token={token}"

    try:
        async with websockets.connect(uri, open_timeout=10) as ws1, \
                   websockets.connect(uri, open_timeout=10) as ws2:

            # 两个连接都等待 connected 事件
            c1 = json.loads(await asyncio.wait_for(ws1.recv(), timeout=5))
            c2 = json.loads(await asyncio.wait_for(ws2.recv(), timeout=5))

            if c1.get("type") == "connected" and c2.get("type") == "connected":
                ok("设备 1 连接成功")
                ok("设备 2 连接成功")
            else:
                fail("连接事件未收到")
                return

            # 从设备 1 发送消息
            await ws1.send(json.dumps({
                "type": "message",
                "content": "这是多端同步测试消息"
            }))
            info("设备 1 发送消息...")

            # 设备 2 应该收到 user_message 广播
            try:
                while True:
                    raw2 = await asyncio.wait_for(ws2.recv(), timeout=15)
                    event2 = json.loads(raw2)
                    if event2.get("type") == "user_message":
                        ok("设备 2 收到设备 1 发送的 user_message 广播 ✓ 多端同步正常")
                        break
                    elif event2.get("type") in ("token", "done", "error"):
                        ok(f"设备 2 同步收到 {event2['type']} 事件")
                        if event2.get("type") == "done":
                            break
            except asyncio.TimeoutError:
                fail("设备 2 未收到广播（超时）")

    except Exception as e:
        fail(f"多端同步测试失败: {e}")
        import traceback; traceback.print_exc()


# ── 测试 4：Ping/Pong 心跳 ────────────────────────────────────
async def test_ping_pong():
    head("测试 4：Ping/Pong 心跳")
    token = register_and_login("ws_test_user", "Test1234!")[0]
    session_id = create_session(token)
    uri = f"{BASE_WS}/ws/{session_id}?token={token}"

    try:
        async with websockets.connect(uri, open_timeout=10) as ws:
            await asyncio.wait_for(ws.recv(), timeout=5)  # connected

            await ws.send(json.dumps({"type": "ping"}))
            info("发送 ping...")

            response = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            if response.get("type") == "pong":
                ok("收到 pong 响应（心跳正常）")
            else:
                fail(f"期望 pong，收到: {response}")

    except Exception as e:
        fail(f"心跳测试失败: {e}")


# ── 测试 5：鉴权失败 ──────────────────────────────────────────
async def test_auth_failure():
    head("测试 5：WebSocket 鉴权失败")

    token = register_and_login("ws_test_user", "Test1234!")[0]
    session_id = create_session(token)

    # 错误 Token
    try:
        async with websockets.connect(
            f"{BASE_WS}/ws/{session_id}?token=invalid_token",
            open_timeout=5
        ) as ws:
            try:
                await asyncio.wait_for(ws.recv(), timeout=3)
                fail("应该被拒绝但连接成功了")
            except Exception:
                ok("错误 Token 连接被拒绝")
    except websockets.exceptions.ConnectionClosedError as e:
        if e.code == 4001:
            ok(f"错误 Token 被正确拒绝（code=4001）")
        else:
            ok(f"连接被拒绝（code={e.code}）")
    except Exception as e:
        ok(f"连接被拒绝: {type(e).__name__}")


# ── 主程序 ────────────────────────────────────────────────────
async def main():
    print(f"\n{BOLD}Strata v7 WebSocket 功能测试{RESET}")
    print(f"后端地址: {BASE_HTTP}\n")

    # 先检查后端是否在线
    try:
        r = requests.get(f"{BASE_HTTP}/api/health", timeout=3)
        if r.status_code == 200:
            print(f"{GREEN}✓{RESET} 后端在线\n")
        else:
            print(f"{RED}✗ 后端返回 {r.status_code}，请确认服务已启动{RESET}")
            sys.exit(1)
    except Exception:
        print(f"{RED}✗ 无法连接到后端 {BASE_HTTP}，请先启动后端服务{RESET}")
        sys.exit(1)

    await test_basic_connection()
    await test_ping_pong()
    await test_auth_failure()
    await test_multi_device_sync()

    # 流式测试放最后（耗时最长）
    print(f"\n{YELLOW}注意：下一项测试会触发 LLM，需要有效的 API Key，耗时约 5-15 秒{RESET}")
    choice = input("是否运行流式消息测试？[y/N] ").strip().lower()
    if choice == "y":
        await test_streaming()

    print(f"\n{BOLD}测试完成{RESET}\n")


if __name__ == "__main__":
    asyncio.run(main())
