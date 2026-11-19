from openai import OpenAI
from openai import AuthenticationError
from openai.types.chat import ChatCompletionMessageParam
from dotenv import load_dotenv
import os

load_dotenv()

SUB2API_BASE = os.getenv("SUB2API_BASE")
SUB2API_KEY = os.getenv("SUB2API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")

if not SUB2API_BASE:
    raise ValueError("缺少 SUB2API_BASE")

if not SUB2API_KEY:
    raise ValueError("缺少 SUB2API_KEY")

if not MODEL_NAME:
    raise ValueError("缺少 MODEL_NAME")

client = OpenAI(
    base_url=SUB2API_BASE,
    api_key=SUB2API_KEY,
    timeout=120
)

messages: list[ChatCompletionMessageParam] = [
    {
        "role": "system",
        "content": (
            "你是一个简洁、严谨的技术助手。回复性格：专业、直接、克制，不使用夸张或过度热情的表达。"
            "回答规范：1. 使用简体中文。2. 先给结论，再给解释。3. 不重复用户的问题。"
            "4. 不输出无关背景。5. 信息不足时明确说明，不要猜测。6. 默认使用普通纯文本。"
            "7. 只有确实要时才使用 Markdown、列表或代码块。8. 不使用表情符号。"
        ),
    }
]

while True:
    try:
        user_input = input("\n你：").strip()
        if user_input.lower() in ("exit", "quit"):
            print("对话结束。")
            break
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})
        stream = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.25,
            stream=True,
        )

        print("助手：", end="", flush=True)
        reply_parts: list[str] = []
        for chunk in stream:
            if chunk.choices:
                content = chunk.choices[0].delta.content
                if content:
                    print(content, end="", flush=True)
                    reply_parts.append(content)
        print()

        assistant_reply = "".join(reply_parts)
        if assistant_reply:
            messages.append({"role": "assistant", "content": assistant_reply})
        else:
            messages.pop()
            print("本轮没有收到文本回复，请重试。")
    except AuthenticationError as e:
        print("\n密钥认证失败 401：检查 API_KEY 是否正确", e)
        break
    except (KeyboardInterrupt, EOFError):
        print("\n对话结束。")
        break
    except Exception as e:
        if messages and messages[-1]["role"] == "user":
            messages.pop()
        print("\n本轮请求失败：", e)
