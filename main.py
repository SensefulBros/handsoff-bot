"""
restaurant_bot.py
==================
Restaurant Bot with 5 agents + Guardrails:
  1. Triage Agent       — routes customer to the right specialist
  2. Menu Agent         — menu items, ingredients, allergies
  3. Order Agent        — takes and confirms orders
  4. Reservation Agent  — handles table bookings
  5. Complaints Agent   — handles dissatisfied customers with empathy

Guardrails:
  - Input:  blocks off-topic & inappropriate messages
  - Output: ensures professional, no-internal-info responses

Run:
    pip install openai-agents streamlit python-dotenv
    streamlit run restaurant_bot.py
"""

import os
import asyncio
import streamlit as st
from pydantic import BaseModel
from agents import (
    Agent,
    Runner,
    SQLiteSession,
    function_tool,
    handoff,
    GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered,
    OutputGuardrailTripwireTriggered,
    RunContextWrapper,
    TResponseInputItem,
    input_guardrail,
    output_guardrail,
)

# ============================================================
# API Key — works on both Streamlit Cloud and local
# ============================================================
# Priority: st.secrets (Cloud) → env var (local) → .env file (fallback)
try:
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except FileNotFoundError:
    pass  # No secrets.toml — that's fine for local dev

if not os.environ.get("OPENAI_API_KEY"):
    try:
        import dotenv
        dotenv.load_dotenv()
    except ImportError:
        pass

if not os.environ.get("OPENAI_API_KEY"):
    st.error("⚠️ OPENAI_API_KEY가 설정되지 않았습니다. Streamlit secrets 또는 환경 변수를 확인하세요.")
    st.stop()

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

RESERVATIONS_DB = []
ORDERS_DB = []
COMPLAINTS_DB = []


# ============================================================
# Function tools
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
    order = {"id": order_id, "items": confirmed, "total": total, "special_requests": special_requests}
    ORDERS_DB.append(order)

    receipt = [f"✅ 주문 확인 (주문번호: #{order_id})", "─" * 30]
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
    RESERVATIONS_DB.append({"id": reservation_id, "name": name, "party_size": party_size, "date": date, "time": time})
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
    if party_size > 8:
        return f"⚠️ {party_size}명은 단체석이 필요합니다. 전화로 문의 부탁드립니다 (02-1234-5678)."
    return f"✅ {date} {time}에 {party_size}명 예약 가능합니다!"


# ── Complaints Agent tools ─────────────────────────────────
@function_tool
def log_complaint(issue: str, severity: str = "medium") -> str:
    """Log a customer complaint for internal tracking.

    Args:
        issue: Description of the complaint
        severity: low, medium, or high
    """
    complaint_id = len(COMPLAINTS_DB) + 1
    COMPLAINTS_DB.append({"id": complaint_id, "issue": issue, "severity": severity, "status": "접수됨"})
    return f"불만 접수 완료 (접수번호: C-{complaint_id:03d}, 심각도: {severity})"


@function_tool
def offer_compensation(compensation_type: str, details: str = "") -> str:
    """Offer compensation to a dissatisfied customer.

    Args:
        compensation_type: Type of compensation — 'discount', 'refund', 'manager_callback', or 'free_item'
        details: Additional details about the compensation
    """
    offers = {
        "discount": "다음 방문 시 50% 할인 쿠폰을 발행해 드리겠습니다.",
        "refund": "해당 메뉴의 전액 환불을 진행해 드리겠습니다.",
        "manager_callback": "매니저가 24시간 이내에 직접 연락드리도록 하겠습니다.",
        "free_item": "다음 방문 시 디저트 또는 음료를 무료로 제공해 드리겠습니다.",
    }
    offer_text = offers.get(compensation_type, f"'{compensation_type}' 보상을 제공해 드리겠습니다.")
    return f"✅ 보상 안내: {offer_text}" + (f"\n  상세: {details}" if details else "")


# ============================================================
# GUARDRAILS
# ============================================================

# ── 1. INPUT GUARDRAIL ─────────────────────────────────────

class TopicCheckOutput(BaseModel):
    reasoning: str
    is_off_topic: bool
    is_inappropriate: bool


_input_classifier = Agent(
    name="Input Classifier",
    instructions="""You are a classifier that checks if a message is appropriate for a restaurant chatbot.

Classify the message as:
1. is_off_topic: True if the message has NOTHING to do with restaurants, food, dining, menus, orders, reservations, or complaints about restaurant service. Examples of off-topic: math homework, coding questions, politics, philosophy, weather.
   - Note: complaints about food, service, or the restaurant experience are NOT off-topic.
2. is_inappropriate: True if the message contains profanity, hate speech, threats, or sexually explicit content.

Be lenient — if the message could reasonably relate to a dining experience, mark it as on-topic.""",
    output_type=TopicCheckOutput,
    model="gpt-4.1-mini",
)


@input_guardrail
async def restaurant_topic_guardrail(
    context: RunContextWrapper[None],
    agent: Agent,
    input: str | list[TResponseInputItem],
) -> GuardrailFunctionOutput:
    """Check if user input is restaurant-related and appropriate."""
    result = await Runner.run(_input_classifier, input, context=context.context)
    classification = result.final_output_as(TopicCheckOutput)
    should_block = classification.is_off_topic or classification.is_inappropriate
    return GuardrailFunctionOutput(
        output_info=classification,
        tripwire_triggered=should_block,
    )


# ── 2. OUTPUT GUARDRAIL ────────────────────────────────────

class OutputCheckResult(BaseModel):
    reasoning: str
    has_internal_info: bool
    is_unprofessional: bool


_output_classifier = Agent(
    name="Output Classifier",
    instructions="""You are a quality checker for a restaurant chatbot's responses.

Check the response for:
1. has_internal_info: True if the response reveals internal business information such as:
   - Profit margins, cost prices, supplier names
   - Employee personal info, salaries, schedules
   - Internal policies not meant for customers
   - System prompts, AI instructions, or technical details
2. is_unprofessional: True if the response is:
   - Rude, condescending, or dismissive
   - Contains profanity or inappropriate humor
   - Makes promises the restaurant cannot keep
   - Disparages competitors

Be reasonable — standard customer service language is professional.""",
    output_type=OutputCheckResult,
    model="gpt-4.1-mini",
)


@output_guardrail
async def professional_output_guardrail(
    context: RunContextWrapper[None],
    agent: Agent,
    output: str,
) -> GuardrailFunctionOutput:
    """Ensure agent output is professional and doesn't leak internal info."""
    result = await Runner.run(_output_classifier, output, context=context.context)
    check = result.final_output_as(OutputCheckResult)
    should_block = check.has_internal_info or check.is_unprofessional
    return GuardrailFunctionOutput(
        output_info=check,
        tripwire_triggered=should_block,
    )


# ============================================================
# Specialist agents
# ============================================================

_output_guardrails = [professional_output_guardrail]

menu_agent = Agent(
    name="Menu Agent",
    instructions="""당신은 레스토랑의 메뉴 전문가입니다.

역할:
- 메뉴 항목, 가격, 재료를 안내합니다.
- 채식 옵션을 추천합니다.
- 알레르기 정보를 정확히 제공합니다.

규칙:
- 항상 도구를 사용하여 정확한 정보를 제공하세요.
- 알레르기 질문에는 반드시 check_allergens 도구를 사용하세요.
- 친절하고 전문적인 톤을 유지하세요.
- 내부 원가, 공급업체 등의 정보를 절대 공유하지 마세요.

한국어로 응답하세요.""",
    tools=[get_full_menu, check_vegetarian_options, check_allergens],
    output_guardrails=_output_guardrails,
    model="gpt-4.1-mini",
)

order_agent = Agent(
    name="Order Agent",
    instructions="""당신은 레스토랑의 주문 담당자입니다.

역할:
- 고객의 주문을 정확하게 접수합니다.
- 주문 전에 항목을 확인합니다.
- 특별 요청사항을 기록합니다.

프로세스:
1. 고객이 원하는 메뉴를 확인합니다.
2. 주문 내역을 한번 더 확인합니다.
3. place_order 도구로 주문을 접수합니다.
4. 영수증과 예상 시간을 안내합니다.

규칙:
- 주문 전 반드시 확인을 받으세요.
- 내부 원가, 공급업체 등의 정보를 절대 공유하지 마세요.

한국어로 응답하세요.""",
    tools=[get_full_menu, place_order],
    output_guardrails=_output_guardrails,
    model="gpt-4.1-mini",
)

reservation_agent = Agent(
    name="Reservation Agent",
    instructions="""당신은 레스토랑의 예약 담당자입니다.

역할:
- 테이블 예약을 접수합니다.
- 예약 가능 여부를 확인합니다.

프로세스:
1. 인원수, 날짜, 시간, 이름을 수집합니다.
2. check_availability로 확인합니다.
3. make_reservation으로 예약을 완료합니다.

규칙:
- 누락된 정보는 친절하게 물어보세요.
- 8명 초과는 전화 예약을 안내하세요.
- 내부 정보를 절대 공유하지 마세요.

한국어로 응답하세요.""",
    tools=[check_availability, make_reservation],
    output_guardrails=_output_guardrails,
    model="gpt-4.1-mini",
)

complaints_agent = Agent(
    name="Complaints Agent",
    instructions="""당신은 레스토랑의 고객 불만 처리 전문가입니다.

역할:
- 불만족한 고객의 이야기를 경청하고 공감합니다.
- 진심 어린 사과를 합니다.
- 구체적인 해결책을 제시합니다.

불만 처리 프로세스:
1. 먼저 고객의 불만을 인정하고 공감을 표현합니다.
   "정말 불쾌하셨겠어요. 저희가 그런 경험을 드려 진심으로 죄송합니다."
2. log_complaint 도구로 불만을 기록합니다.
3. 상황에 맞는 보상을 offer_compensation 도구로 제안합니다:
   - 음식 품질 문제 → 'refund' 또는 'free_item'
   - 서비스 문제 → 'discount'
   - 심각한 문제 (위생, 안전) → 'manager_callback' + 'refund'
4. 고객이 원하는 해결 방법을 물어봅니다.

톤:
- 절대 방어적이거나 변명하지 마세요.
- 항상 "저희 잘못입니다"라는 자세를 유지하세요.
- 구체적인 보상 옵션을 최소 2가지 제시하세요.
- 내부 원가, 공급업체, 직원 개인정보를 절대 공유하지 마세요.

한국어로 응답하세요.""",
    tools=[log_complaint, offer_compensation],
    output_guardrails=_output_guardrails,
    model="gpt-4.1-mini",
)


# ============================================================
# Cross-agent handoffs
# ============================================================
menu_agent.handoffs = [
    handoff(order_agent, tool_description_override="주문을 하고 싶어하는 고객을 주문 담당에게 연결"),
    handoff(reservation_agent, tool_description_override="예약을 하고 싶어하는 고객을 예약 담당에게 연결"),
    handoff(complaints_agent, tool_description_override="불만이 있는 고객을 불만 처리 담당에게 연결"),
]
order_agent.handoffs = [
    handoff(menu_agent, tool_description_override="메뉴에 대해 질문하는 고객을 메뉴 전문가에게 연결"),
    handoff(reservation_agent, tool_description_override="예약을 하고 싶어하는 고객을 예약 담당에게 연결"),
    handoff(complaints_agent, tool_description_override="불만이 있는 고객을 불만 처리 담당에게 연결"),
]
reservation_agent.handoffs = [
    handoff(menu_agent, tool_description_override="메뉴에 대해 질문하는 고객을 메뉴 전문가에게 연결"),
    handoff(order_agent, tool_description_override="주문을 하고 싶어하는 고객을 주문 담당에게 연결"),
    handoff(complaints_agent, tool_description_override="불만이 있는 고객을 불만 처리 담당에게 연결"),
]
complaints_agent.handoffs = [
    handoff(menu_agent, tool_description_override="메뉴에 대해 질문하는 고객을 메뉴 전문가에게 연결"),
    handoff(order_agent, tool_description_override="주문을 하고 싶어하는 고객을 주문 담당에게 연결"),
    handoff(reservation_agent, tool_description_override="예약을 하고 싶어하는 고객을 예약 담당에게 연결"),
]


# ============================================================
# Triage agent — INPUT guardrail attached here
# ============================================================
triage_agent = Agent(
    name="Triage Agent",
    instructions="""당신은 레스토랑의 안내 데스크 직원입니다.

역할:
- 고객을 환영하고 무엇을 도와드릴지 파악합니다.
- 적절한 전문 담당자에게 연결합니다.

라우팅 규칙:
- 메뉴, 음식, 재료, 알레르기, 채식 → Menu Agent
- 주문, 음식 시키기 → Order Agent
- 예약, 테이블, 날짜/시간 → Reservation Agent
- 불만, 불쾌, 서비스 문제, 환불, 컴플레인 → Complaints Agent
- 인사나 일반 대화 → 직접 응답하고 도움을 제안

중요:
- 항상 먼저 "~담당에게 연결해 드릴게요!" 라고 안내한 후 handoff하세요.
- 빠르고 친절하게 라우팅하세요.
- 직접 메뉴 정보를 답하거나 주문을 받지 마세요.

한국어로 응답하세요.""",
    handoffs=[
        handoff(menu_agent, tool_description_override="메뉴, 음식, 재료, 알레르기에 대해 질문하는 고객을 메뉴 전문가에게 연결"),
        handoff(order_agent, tool_description_override="음식을 주문하고 싶어하는 고객을 주문 담당에게 연결"),
        handoff(reservation_agent, tool_description_override="테이블 예약을 하고 싶어하는 고객을 예약 담당에게 연결"),
        handoff(complaints_agent, tool_description_override="불만족하거나 문제가 있는 고객을 불만 처리 담당에게 연결"),
    ],
    input_guardrails=[restaurant_topic_guardrail],
    output_guardrails=_output_guardrails,
    model="gpt-4.1-mini",
)


# ============================================================
# Session
# ============================================================
SESSION_DB = "restaurant_bot_memory.db"
SESSION_ID = "restaurant-session"

if "session" not in st.session_state:
    st.session_state["session"] = SQLiteSession(SESSION_ID, SESSION_DB)

memory = st.session_state["session"]

# ============================================================
# Agent display mapping
# ============================================================
AGENT_DISPLAY = {
    "Triage Agent": ("🏠 안내 데스크", "triage"),
    "Menu Agent": ("📋 메뉴 전문가", "menu"),
    "Order Agent": ("🛒 주문 담당", "order"),
    "Reservation Agent": ("📅 예약 담당", "reservation"),
    "Complaints Agent": ("😔 불만 처리 담당", "complaints"),
}

# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.markdown("## 🍽️ Restaurant Bot")
    st.caption("AI 레스토랑 어시스턴트 (Guardrails 적용)")
    st.divider()

    st.markdown("### 👥 에이전트 팀")
    st.markdown(
        """
| 에이전트 | 역할 |
|----------|------|
| 🏠 안내 데스크 | 요청 파악 & 라우팅 |
| 📋 메뉴 전문가 | 메뉴, 재료, 알레르기 |
| 🛒 주문 담당 | 주문 접수 & 확인 |
| 📅 예약 담당 | 테이블 예약 처리 |
| 😔 불만 처리 | 공감, 사과, 보상 |
"""
    )

    st.divider()
    st.markdown("### 🛡️ Guardrails")
    st.markdown(
        """
| 가드레일 | 보호 대상 |
|----------|----------|
| 🚫 Input | 주제 이탈 & 부적절한 언어 차단 |
| ✅ Output | 전문성 보장 & 내부정보 차단 |
"""
    )

    st.divider()
    st.markdown("### 💡 테스트 프롬프트")
    demo_prompts = {
        "📋 메뉴 보기": "메뉴 좀 보여줘",
        "🌿 채식 메뉴": "채식 메뉴 있어?",
        "🛒 주문하기": "안심 스테이크랑 시저 샐러드 주문할게",
        "📅 예약하기": "이번 토요일 저녁 7시에 4명 예약하고 싶어",
        "😔 불만 접수": "음식이 너무 별로였고 직원도 불친절했어",
        "🚫 주제 이탈": "인생의 의미가 뭘까?",
    }
    for label, prompt_text in demo_prompts.items():
        if st.button(label, key=f"btn_{label}", use_container_width=True):
            st.session_state["sidebar_prompt"] = prompt_text

    st.divider()
    if st.button("🗑️ 대화 초기화", use_container_width=True):
        asyncio.run(memory.clear_session())
        st.rerun()


# ============================================================
# Render chat history
# ============================================================
async def render_history():
    items = await memory.get_items()
    for item in items:
        if item.get("role") == "user":
            with st.chat_message("user"):
                content = item.get("content", "")
                if isinstance(content, str):
                    st.write(content)
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "input_text":
                            st.write(part.get("text", ""))

        elif item.get("role") == "assistant" and item.get("type") == "message":
            with st.chat_message("assistant"):
                for part in item.get("content", []):
                    if isinstance(part, dict) and "text" in part:
                        st.write(part["text"].replace("$", r"\$"))

        elif item.get("type") == "function_call":
            fn = item.get("name", "")
            if fn.startswith("transfer_to_"):
                agent_key = fn.replace("transfer_to_", "").replace("_", " ").title()
                display_name = AGENT_DISPLAY.get(agent_key, (f"🔄 {agent_key}", ""))[0]
                with st.chat_message("assistant"):
                    st.info(f"🔄 {display_name}에게 연결합니다...")


asyncio.run(render_history())


# ============================================================
# Guardrail rejection messages
# ============================================================
INPUT_BLOCK_MSG = (
    "🚫 저는 레스토랑 관련 질문에 대해서만 도와드리고 있어요.\n\n"
    "다음과 같은 것들을 도와드릴 수 있습니다:\n"
    "• 📋 메뉴 확인 및 추천\n"
    "• 🛒 음식 주문\n"
    "• 📅 테이블 예약\n"
    "• 😔 서비스 관련 불만 접수\n\n"
    "무엇을 도와드릴까요?"
)

OUTPUT_BLOCK_MSG = (
    "⚠️ 죄송합니다. 응답을 생성하는 중 문제가 발생했습니다.\n"
    "다시 질문해 주시거나, 다른 도움이 필요하시면 말씀해 주세요."
)


# ============================================================
# Stream response with guardrail exception handling
# ============================================================
STREAM_LABELS = {
    "response.output_text.delta": None,
    "response.completed": ("✅ 응답 완료", "complete"),
}


async def stream_response(user_message: str):
    """Run triage agent with streaming + guardrail exception handling."""
    try:
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

                    if event_name in STREAM_LABELS and STREAM_LABELS[event_name]:
                        label, state = STREAM_LABELS[event_name]
                        status_box.update(label=label, state=state)

                    if event_name == "response.output_text.delta":
                        accumulated_text += event.data.delta
                        text_area.write(accumulated_text.replace("$", r"\$"))

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

            final_display = AGENT_DISPLAY.get(
                stream.result.last_agent.name,
                (stream.result.last_agent.name, ""),
            )[0]
            status_box.update(label=f"✅ {final_display}가 답변 완료", state="complete")

    except InputGuardrailTripwireTriggered:
        with st.chat_message("assistant"):
            st.warning(INPUT_BLOCK_MSG)

    except OutputGuardrailTripwireTriggered:
        with st.chat_message("assistant"):
            st.error(OUTPUT_BLOCK_MSG)


# ============================================================
# Title
# ============================================================
st.title("🍽️ Restaurant Bot")
st.caption("메뉴 · 주문 · 예약 · 불만 처리 — AI 레스토랑 어시스턴트 (Guardrails 적용)")

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