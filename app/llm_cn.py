import json
import os
from pathlib import Path
from typing import List, Optional

import requests
from pydantic import BaseModel


# ==== 从 .env 加载环境变量 ====


def _load_env_from_dotenv():
    """
    简单版 .env 加载器：
    - 假设 .env 在项目根目录（app 上级目录）
    - 形如 KEY=VALUE 的行会被写入 os.environ（如果原来没有）
    """
    try:
        root_dir = Path(__file__).resolve().parent.parent
        env_path = root_dir / ".env"
        if not env_path.is_file():
            return
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    except Exception:
        # 失败就静默忽略，不影响后续逻辑
        pass


_load_env_from_dotenv()


# ==== 配置 ====


QWEN_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
QWEN_BASE_URL = os.getenv(
    "QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
)
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen-plus")


SYSTEM_PROMPT = """
你是一个中文旅行规划助手，类似“本地朋友 + 旅行教练”。

- 你会基于用户的对话历史和最新一句话，判断现在应该：
  - 继续和用户澄清需求；还是
  - 触发行程规划接口 `/v1/trip_plan`。

- 你只需要返回一个 JSON 对象，格式必须是：

{
  "reply": "<给用户看的中文回复>",
  "should_call_trip_plan": false,
  "confirm_real_world": false,
  "slots_json": null
}

字段含义：
- reply：给用户看的自然语言回复（中文）。
- should_call_trip_plan：如果已经收集到足够信息，可以帮用户直接出行程草案，请设为 true。
- confirm_real_world：是否需要向用户确认某些现实世界约束（例如签证/天气/节假日），暂时可以一直返回 false。
- slots_json：当 should_call_trip_plan 为 true 时，这里应是 TripPlanRequest 对应的 JSON 对象（而不是字符串）；否则为 null。
  例如：
  {
    "city": "杭州",
    "days": 2,
    "start_date": "2025-11-29",
    ...
  }

注意：
- 一定要返回合法 JSON，不能有注释，不能有多余的文本。
- 如果暂时信息不够，不要强行凑 slots_json，should_call_trip_plan 设为 false 即可。
""".strip()


# ==== 数据模型 ====


class ChatMessage(BaseModel):
    role: str
    content: str


class TripChatLLMInput(BaseModel):
    user_id: str
    history: List[ChatMessage]
    new_user_message: str
    location_lat: Optional[float] = None
    location_lng: Optional[float] = None


class TripChatLLMDecision(BaseModel):
    reply: str
    should_call_trip_plan: bool = False
    confirm_real_world: bool = False
    # 注意：这里用字符串保存 slots_json，方便上层按需解析
    slots_json: Optional[str] = None


# ==== 调用千问 ====


def _build_messages(llm_input: TripChatLLMInput):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for m in llm_input.history:
        role = m.role
        if role not in ("system", "user", "assistant"):
            role = "user"
        messages.append({"role": role, "content": m.content})

    user_content = llm_input.new_user_message
    if llm_input.location_lat is not None and llm_input.location_lng is not None:
        user_content += f"\n\n[用户位置坐标] lat={llm_input.location_lat}, lng={llm_input.location_lng}"

    messages.append({"role": "user", "content": user_content})
    return messages


def call_qwen_for_trip_chat(llm_input: TripChatLLMInput) -> TripChatLLMDecision:
    """
    调用通义千问（OpenAI 兼容接口），返回 TripChatLLMDecision。
    如果 API Key 缺失或网络异常，则优雅降级，返回一个简单文本回复。
    """
    # 没配置 key：直接降级
    if not QWEN_API_KEY:
        return TripChatLLMDecision(
            reply="（网关未配置 DASHSCOPE_API_KEY，目前先用本地占位回复。）\n你刚才说的是："
            + llm_input.new_user_message,
            should_call_trip_plan=False,
            confirm_real_world=False,
            slots_json=None,
        )

    messages = _build_messages(llm_input)

    try:
        resp = requests.post(
            f"{QWEN_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {QWEN_API_KEY}"},
            json={
                "model": QWEN_MODEL,
                "messages": messages,
                "temperature": 0.7,
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return TripChatLLMDecision(
            reply=f"（调用千问接口失败:{e}）\n你刚才说的是："
            + llm_input.new_user_message,
            should_call_trip_plan=False,
            confirm_real_world=False,
            slots_json=None,
        )

    # 期望 content 是一个 JSON 字符串
    try:
        obj = json.loads(content)
        reply = obj.get("reply") or "我已经收到你的需求了。"
        should_call_trip_plan = bool(obj.get("should_call_trip_plan", False))
        confirm_real_world = bool(obj.get("confirm_real_world", False))
        slots = obj.get("slots_json", None)

        if slots is None:
            slots_json_str: Optional[str] = None
        else:
            slots_json_str = json.dumps(slots, ensure_ascii=False)

        return TripChatLLMDecision(
            reply=reply,
            should_call_trip_plan=should_call_trip_plan,
            confirm_real_world=confirm_real_world,
            slots_json=slots_json_str,
        )
    except Exception:
        return TripChatLLMDecision(
            reply=content,
            should_call_trip_plan=False,
            confirm_real_world=False,
            slots_json=None,
        )
