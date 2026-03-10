"""
restaurant_bot.py
==================
Restaurant Bot with 4 agents connected via handoffs:
  1. Triage Agent   — routes customer to the right specialist
  2. Menu Agent     — menu items, ingredients, allergies
  3. Order Agent    — takes and confirms orders
  4. Reservation Agent — handles table bookings

Run:
    pip install openai-agents streamlit python-dotenv
    streamlit run restaurant_bot.py
"""

import asyncio
import dotenv
import streamlit as st
from agents import Agent, Runner, SQLiteSession, function_tool, handoff

dotenv.load_dotenv()

# ============================================================
# Page config
# ============================================================
st.set_page_config(page_title="🍽️ Restaurant Bot", page_icon="🍽️", layout="centered")

# ============================================================
# Restaurant data — mock database
# ============================================================
MENU = {
    "appetizers": [
        {"name": "시저 샐러드", "price": 12000, "ingredients": "로메인, 파마산, 크루통, 시저드레싱", "vegetarian": True, "allergens": "유제품, 글루텐"},
        {"name": "새우 튀김", "price": 15000, "ingredients": "새우, 밀가루, 빵가루, 레몬", "vegetarian": False, "allergens": "갑각류, 글루텐"},
        {"name": "버섯 크림수프", "price": 10000, "ingredients": "양송이, 크림, 양파, 버터", "vegetarian": True, "allergens": "유제품"},
    ],
    "main_courses": [
        {"name": "안심 스테이크", "price": 45000, "ingredients": "소안심, 감자, 아스파라거스, 레드와인소스", "vegetarian": False, "allergens": "없음"},
        {"name": "연어 구이", "price": 35000, "ingredients": "연어, 레몬버터, 계절채소", "vegetarian": False, "allergens": "유제품, 생선"},
        {"name": "채식 파스타", "price": 22000, "ingredients": "펜네, 토마토, 올리브, 바질, 마늘", "vegetarian": True, "allergens": "글루텐"},
        {"name": "두부 스테이크", "price": 20000, "ingredients": "두부, 버섯소스, 구운채소", "vegetarian": True, "allergens": "대두"},
    ],
    "desserts": [
        {"name": "티라미수", "price": 12000, "ingredients": "마스카포네, 에스프레소, 코코아", "vegetarian": True, "allergens": "유제품, 글루텐, 카페인"},
        {"name": "과일 셔벗", "price": 9000, "ingredients": "제철 과일, 설탕, 레몬즙", "vegetarian": True, "allergens": "없음"},
    ],
    "drinks": [
        {"name": "아메리카노", "price": 5000, "ingredients": "에스프레소, 물", "vegetarian": True, "allergens": "카페인"},
        {"name": "생과일주스", "price": 8000, "ingredients": "제철 과일", "vegetarian": True, "allergens": "없음"},
        {"name": "하우스 와인 (글라스)", "price": 15000, "ingredients": "포도", "vegetarian": True, "allergens": "아황산염"},
    ],
}

RESERVATIONS_DB = []  # in-memory store for demo
ORDERS_DB = []


# ============================================================
# Function tools — actions the agents can take
# ============================================================
@function_tool
def get_full_menu() -> str:
    """Get the complete restaurant menu with prices and details."""
    result = []
    for category, items in MENU.items():
        category_kr = {
            "appetizers": "🥗 에피타이저",
            "main_courses": "🥩 메인 요리",
            "desserts": "🍰 디저트",
            "drinks": "🥤 음료",
        }.get(category, category)
        result.append(f"\n{category_kr}")
        for item in items:
            veg = " (채식)" if item["vegetarian"] else ""
            result.append(
                f"  • {item['name']}{veg} — {item['price']:,}원\n"
                f"    재료: {item['ingredients']}\n"
                f"    알레르기: {item['allergens']}"
            )
    return "\n".join(result)


@function_tool
def check_vegetarian_options() -> str:
    """Get all vegetarian menu items."""
    result = ["🌿 채식 메뉴 목록:"]
    for category, items in MENU.items():
        for item in items:
            if item["vegetarian"]:
                result.append(f"  • {item['name']} — {item['price']:,}원")
    return "\n".join(result)


@function_tool
def check_allergens(allergen: str) -> str:
    """Check which menu items contain a specific allergen.

    Args:
        allergen: The allergen to check for (e.g., '유제품', '글루텐', '갑각류')
    """
    safe = []
    contains = []
    for category, items in MENU.items():
        for item in items:
            if allergen in item["allergens"]:
                contains.append(item["name"])
            else:
                safe.append(item["name"])
    return (
        f"⚠️ '{allergen}' 포함 메뉴: {', '.join(contains)}\n"
        f"✅ '{allergen}' 없는 메뉴: {', '.join(safe)}"
    )


@function_tool
def place_order(items: str, special_requests: str = "") -> str:
    """Place an order with the specified menu items.

    Args:
        items: Comma-separated list of menu item names to order
        special_requests: Any special requests or modifications
    """
    order_items = [i.strip() for i in items.split(",")]
    total = 0
    confirmed = []

    all_menu_items = {}
    for category in MENU.values():
        for item in category:
            all_menu_items[item["name"]] = item

    for name in order_items:
        if name in all_menu_items:
            confirmed.append(name)
            total += all_menu_items[name]["price"]

    if not confirmed:
        return "❌ 주문하신 메뉴를 찾을 수 없습니다. 메뉴명을 다시 확인해주세요."

    order_id = len(ORDERS_DB) + 1
    order = {
        "id": order_id,
        "items": confirmed,
        "total": total,
        "special_requests": special_requests,
    }
    ORDERS_DB.append(order)

    receipt = [
        f"✅ 주문 확인 (주문번호: #{order_id})",
        "─" * 30,
    ]
    for name in confirmed:
        price = all_menu_items[name]["price"]
        receipt.append(f"  {name} — {price:,}원")
    receipt.append("─" * 30)
    receipt.append(f"  합계: {total:,}원")
    if special_requests:
        receipt.append(f"  요청사항: {special_requests}")
    receipt.append("\n조리 시간은 약 20~30분 예상됩니다.")

    return "\n".join(receipt)


@function_tool
def make_reservation(party_size: int, date: str, time: str, name: str) -> str:
    """Make a table reservation.

    Args:
        party_size: Number of guests
        date: Desired date (e.g., '2025-03-15' or '이번 토요일')
        time: Desired time (e.g., '19:00' or '저녁 7시')
        name: Name for the reservation
    """
    reservation_id = len(RESERVATIONS_DB) + 1
    reservation = {
        "id": reservation_id,
        "name": name,
        "party_size": party_size,
        "date": date,
        "time": time,
    }
    RESERVATIONS_DB.append(reservation)

    return (
        f"✅ 예약이 완료되었습니다!\n"
        f"─────────────────────\n"
        f"  예약번호: #{reservation_id}\n"
        f"  예약자: {name}\n"
        f"  인원: {party_size}명\n"
        f"  날짜: {date}\n"
        f"  시간: {time}\n"
        f"─────────────────────\n"
        f"변경이나 취소는 언제든 말씀해주세요."
    )


@function_tool
def check_availability(date: str, time: str, party_size: int) -> str:
    """Check table availability for a given date, time and party size.

    Args:
        date: Date to check
        time: Time to check
        party_size: Number of guests
    """
    # Mock — always available for demo, but show capacity limits
    if party_size > 8:
        return f"⚠️ {party_size}명은 단체석이 필요합니다. 전화로 문의 부탁드립니다 (02-1234-5678)."
    return f"✅ {date} {time}에 {party_size}명 예약 가능합니다!"


# ============================================================
# Specialist agents (defined first, triage references them)
# ============================================================

menu_agent = Agent(
    name="Menu Agent",
    instructions="""당신은 레스토랑의 메뉴 전문가입니다.

역할:
- 메뉴 항목, 가격, 재료를 안내합니다.
- 채식 옵션을 추천합니다.
- 알레르기 정보를 정확히 제공합니다.
- 음식 추천을 해줍니다.

규칙:
- 항상 get_full_menu 또는 관련 도구를 사용하여 정확한 정보를 제공하세요.
- 알레르기 질문에는 반드시 check_allergens 도구를 사용하세요.
- 친절하고 전문적인 톤을 유지하세요.
- 대화가 주문이나 예약으로 넘어가면 Triage Agent에게 다시 전달하세요.

한국어로 응답하세요.""",
    tools=[get_full_menu, check_vegetarian_options, check_allergens],
    model="gpt-4.1-mini",
)

order_agent = Agent(
    name="Order Agent",
    instructions="""당신은 레스토랑의 주문 담당자입니다.

역할:
- 고객의 주문을 정확하게 접수합니다.
- 주문 전에 항목을 확인합니다.
- 특별 요청사항을 기록합니다.
- 주문 완료 후 영수증을 보여줍니다.

주문 프로세스:
1. 고객이 원하는 메뉴를 확인합니다.
2. 주문 내역을 한번 더 확인합니다 ("안심 스테이크 1개, 시저 샐러드 1개 맞으시죠?")
3. place_order 도구로 주문을 접수합니다.
4. 영수증을 보여주고 예상 시간을 안내합니다.

규칙:
- 주문 전 반드시 고객에게 확인을 받으세요.
- 메뉴에 없는 항목은 주문할 수 없습니다.
- 대화가 메뉴 질문이나 예약으로 넘어가면 Triage Agent에게 다시 전달하세요.

한국어로 응답하세요.""",
    tools=[get_full_menu, place_order],
    model="gpt-4.1-mini",
)

reservation_agent = Agent(
    name="Reservation Agent",
    instructions="""당신은 레스토랑의 예약 담당자입니다.

역할:
- 테이블 예약을 접수합니다.
- 예약 가능 여부를 확인합니다.
- 예약 확인서를 제공합니다.

예약 프로세스:
1. 필요한 정보를 수집합니다: 인원수, 날짜, 시간, 예약자 이름
2. check_availability로 가능 여부를 확인합니다.
3. make_reservation으로 예약을 완료합니다.
4. 예약 확인서를 보여줍니다.

규칙:
- 누락된 정보는 하나씩 친절하게 물어보세요.
- 8명 초과 단체는 전화 예약을 안내하세요.
- 대화가 메뉴 질문이나 주문으로 넘어가면 Triage Agent에게 다시 전달하세요.

한국어로 응답하세요.""",
    tools=[check_availability, make_reservation],
    model="gpt-4.1-mini",
)

# ============================================================
# Triage agent — routes to specialists via handoffs
# ============================================================

# Enable specialists to hand back to triage when topic changes
menu_agent.handoffs = [
    handoff(order_agent, tool_description_override="주문을 하고 싶어하는 고객을 주문 담당에게 연결"),
    handoff(reservation_agent, tool_description_override="예약을 하고 싶어하는 고객을 예약 담당에게 연결"),
]
order_agent.handoffs = [
    handoff(menu_agent, tool_description_override="메뉴에 대해 질문하는 고객을 메뉴 전문가에게 연결"),
    handoff(reservation_agent, tool_description_override="예약을 하고 싶어하는 고객을 예약 담당에게 연결"),
]
reservation_agent.handoffs = [
    handoff(menu_agent, tool_description_override="메뉴에 대해 질문하는 고객을 메뉴 전문가에게 연결"),
    handoff(order_agent, tool_description_override="주문을 하고 싶어하는 고객을 주문 담당에게 연결"),
]

triage_agent = Agent(
    name="Triage Agent",
    instructions="""당신은 레스토랑의 안내 데스크 직원입니다.

역할:
- 고객을 환영하고 무엇을 도와드릴지 파악합니다.
- 적절한 전문 담당자에게 연결합니다.

라우팅 규칙:
- 메뉴, 음식, 재료, 알레르기, 채식 → Menu Agent에게 전달
- 주문, 음식 시키기 → Order Agent에게 전달
- 예약, 테이블, 날짜/시간 → Reservation Agent에게 전달
- 인사나 일반 대화 → 직접 응답하고 도움을 제안

중요:
- 항상 먼저 고객에게 "~담당에게 연결해 드릴게요!" 라고 안내한 후 handoff하세요.
- 빠르고 친절하게 라우팅하세요. 불필요한 질문을 하지 마세요.
- 직접 메뉴 정보를 답하거나 주문을 받지 마세요 — 반드시 전문 에이전트에게 넘기세요.

한국어로 응답하세요.""",
    handoffs=[
        handoff(menu_agent, tool_description_override="메뉴, 음식, 재료, 알레르기에 대해 질문하는 고객을 메뉴 전문가에게 연결"),
        handoff(order_agent, tool_description_override="음식을 주문하고 싶어하는 고객을 주문 담당에게 연결"),
        handoff(reservation_agent, tool_description_override="테이블 예약을 하고 싶어하는 고객을 예약 담당에게 연결"),
    ],
    model="gpt-4.1-mini",
)

# ============================================================
# Session — persistent memory
# ============================================================
SESSION_DB = "restaurant_bot_memory.db"
SESSION_ID = "restaurant-session"

if "session" not in st.session_state:
    st.session_state["session"] = SQLiteSession(SESSION_ID, SESSION_DB)

memory = st.session_state["session"]

# ============================================================
# Agent name → emoji mapping for UI
# ============================================================
AGENT_DISPLAY = {
    "Triage Agent": ("🏠 안내 데스크", "triage"),
    "Menu Agent": ("📋 메뉴 전문가", "menu"),
    "Order Agent": ("🛒 주문 담당", "order"),
    "Reservation Agent": ("📅 예약 담당", "reservation"),
}

# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.markdown("## 🍽️ Restaurant Bot")
    st.caption("AI 레스토랑 어시스턴트")
    st.divider()

    st.markdown("### 👥 에이전트 팀")
    st.markdown(
        """
| 에이전트 | 역할 |
|----------|------|
| 🏠 안내 데스크 | 고객 요청 파악 & 라우팅 |
| 📋 메뉴 전문가 | 메뉴, 재료, 알레르기 |
| 🛒 주문 담당 | 주문 접수 & 확인 |
| 📅 예약 담당 | 테이블 예약 처리 |
"""
    )
    st.divider()

    st.markdown("### 💡 이런 걸 물어보세요")
    demo_prompts = {
        "📋 메뉴 보기": "메뉴 좀 보여줘",
        "🌿 채식 메뉴": "채식 메뉴 있어?",
        "⚠️ 알레르기": "유제품 알레르기가 있는데 먹을 수 있는 메뉴가 뭐야?",
        "🛒 주문하기": "안심 스테이크랑 시저 샐러드 주문할게",
        "📅 예약하기": "이번 토요일 저녁 7시에 4명 예약하고 싶어",
    }
    for label, prompt_text in demo_prompts.items():
        if st.button(label, key=f"btn_{label}", use_container_width=True):
            st.session_state["sidebar_prompt"] = prompt_text

    st.divider()
    if st.button("🗑️ 대화 초기화", use_container_width=True):
        asyncio.run(memory.clear_session())
        st.rerun()


# ============================================================
# Render saved chat history
# ============================================================
async def render_history():
    items = await memory.get_items()
    for item in items:
        # User messages
        if item.get("role") == "user":
            with st.chat_message("user"):
                content = item.get("content", "")
                if isinstance(content, str):
                    st.write(content)
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "input_text":
                            st.write(part.get("text", ""))

        # Assistant text messages
        elif item.get("role") == "assistant" and item.get("type") == "message":
            with st.chat_message("assistant"):
                for part in item.get("content", []):
                    if isinstance(part, dict) and "text" in part:
                        st.write(part["text"].replace("$", r"\$"))

        # Tool calls (show which tools were used)
        elif item.get("type") == "function_call":
            fn = item.get("name", "")
            if fn.startswith("transfer_to_"):
                agent_key = fn.replace("transfer_to_", "").replace("_", " ").title()
                display_name = AGENT_DISPLAY.get(agent_key, (f"🔄 {agent_key}", ""))[0]
                with st.chat_message("assistant"):
                    st.info(f"🔄 {display_name}에게 연결합니다...")


asyncio.run(render_history())


# ============================================================
# Stream status labels
# ============================================================
STREAM_LABELS = {
    "response.output_text.delta": None,  # handled separately
    "response.completed": ("✅ 응답 완료", "complete"),
}


# ============================================================
# Stream agent response with handoff visualization
# ============================================================
async def stream_response(user_message: str):
    with st.chat_message("assistant"):
        status_box = st.status("🏠 안내 데스크에서 확인 중...", expanded=False)
        text_area = st.empty()
        accumulated_text = ""
        current_agent_name = "Triage Agent"

        stream = Runner.run_streamed(
            triage_agent,
            user_message,
            session=memory,
        )

        async for event in stream.stream_events():
            if event.type == "raw_response_event":
                event_name = event.data.type

                # Update status
                if event_name in STREAM_LABELS and STREAM_LABELS[event_name]:
                    label, state = STREAM_LABELS[event_name]
                    status_box.update(label=label, state=state)

                # Stream text
                if event_name == "response.output_text.delta":
                    accumulated_text += event.data.delta
                    text_area.write(accumulated_text.replace("$", r"\$"))

            # Detect handoffs via agent_updated events
            elif event.type == "agent_updated_stream_event":
                new_agent = event.new_agent
                if new_agent.name != current_agent_name:
                    current_agent_name = new_agent.name
                    display = AGENT_DISPLAY.get(
                        current_agent_name,
                        (f"🔄 {current_agent_name}", ""),
                    )[0]
                    status_box.update(
                        label=f"🔄 {display}에게 연결되었습니다",
                        state="running",
                    )

        # Final status — show which agent answered
        final_display = AGENT_DISPLAY.get(
            stream.result.last_agent.name,
            (stream.result.last_agent.name, ""),
        )[0]
        status_box.update(
            label=f"✅ {final_display}가 답변 완료",
            state="complete",
        )


# ============================================================
# Title
# ============================================================
st.title("🍽️ Restaurant Bot")
st.caption("메뉴 안내 · 주문 · 예약까지 — AI 레스토랑 어시스턴트")

# ============================================================
# Chat input
# ============================================================
sidebar_prompt = st.session_state.pop("sidebar_prompt", None)
typed_input = st.chat_input("무엇을 도와드릴까요?")
active_message = sidebar_prompt or typed_input

if active_message:
    with st.chat_message("user"):
        st.write(active_message)
    asyncio.run(stream_response(active_message))