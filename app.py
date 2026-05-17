# app.py
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
from dotenv import load_dotenv
from agent import classify_customer
import hashlib

load_dotenv()

# ── Page Config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="🏦 Churn Prediction Agent",
    page_icon="🏦",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1a1a2e, #16213e, #0f3460);
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 25px;
    }
    .metric-card {
        background: #16213e;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #0f3460;
    }
    .step-box {
        background-color: #1e1e2e;
        border-left: 4px solid #0f3460;
        padding: 10px;
        margin: 5px 0;
        border-radius: 4px;
        font-family: monospace;
        font-size: 13px;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ System Info")

    st.success("🟢 Groq — Llama 3.3 70B")
    st.info("🤖 ReAct Agent Active")

    st.markdown("---")
    st.markdown("### 🧠 Models Loaded")
    st.markdown("✅ SVM (Accuracy: 85.6%)")
    st.markdown("✅ Decision Tree (Accuracy: 85.9%)")
    st.markdown("✅ Neural Network (Accuracy: 86.1%)")

    st.markdown("---")
    st.markdown("### 📊 Dataset")
    st.markdown("🏦 Bank Customer Churn")
    st.markdown("📋 10,000 records")
    st.markdown("🔢 11 features")

    st.markdown("---")
    st.markdown("### 🔄 How It Works")
    st.markdown("""
    1️⃣ You describe a customer  
    2️⃣ LLM Agent thinks & reasons  
    3️⃣ Picks the best ML model  
    4️⃣ Returns prediction + explanation  
    """)

    st.markdown("---")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown("### 📈 Session Stats")
    total = len([m for m in st.session_state.get("messages", [])
                 if m["role"] == "user"])
    churns = len([m for m in st.session_state.get("messages", [])
                  if m["role"] == "assistant" and "CHURN" in m["content"].upper()])
    stays = len([m for m in st.session_state.get("messages", [])
                 if m["role"] == "assistant" and "STAY" in m["content"].upper()])

    st.metric("🔮 Total Predictions", total)
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("🔴 Churns", churns)
    with col_b:
        st.metric("🟢 Stays", stays)

    st.markdown("---")
    st.markdown("### 📈 Caching Status")
    st.success("📄 Predictions cached (1 hour TTL)")
    st.caption("Same predictions use cache to prevent rate limits")

    st.markdown("---")
    st.caption("COMSATS University · Spring 2026")
    st.caption("ML Project By Awais Manzoor")

# ── Header ────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1 style="color: white; margin: 0;">
        🏦 Agentic Customer Churn Prediction System
    </h1>
    <p style="color: #aaaaaa; margin: 8px 0 0 0;">
        LangGraph · ReAct Framework · LangChain · Groq Llama 3.3 70B · Scikit-Learn
    </p>
</div>
""", unsafe_allow_html=True)

# ── Stats Row ─────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("🤖 LLM Backend", "Llama 3.3 70B")
with col2:
    st.metric("🧠 Framework", "ReAct Agent")
with col3:
    st.metric("⚡ ML Models", "3 Classifiers")
with col4:
    st.metric("💰 API Cost", "$0 Free")

st.markdown("---")

# ── Example Inputs ────────────────────────────────────────────────────
with st.expander("💡 Example Customer Descriptions — Click to Try These"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**🔴 Likely to CHURN:**")
        st.info("""
A 45-year-old female customer
from Germany. Credit score 650,
balance 130000, tenure 2 years,
1 product, has credit card,
NOT an active member,
salary 85000.
        """)

    with col2:
        st.markdown("**🟢 Likely to STAY:**")
        st.info("""
Male customer aged 32 from
France. Credit score 750,
balance 0, tenure 8 years,
2 products, has credit card,
active member,
salary 120000.
        """)

    with col3:
        st.markdown("**🟡 Borderline Case:**")
        st.info("""
A 38-year-old male from Spain.
Credit score 580, balance 75000,
tenure 5 years, 1 product,
no credit card, active member,
salary 60000.
        """)

st.markdown("---")

# ── Manual Input Form ─────────────────────────────────────────────────
with st.expander("📝 Fill Customer Details Manually — Then Click Predict"):
    st.markdown("Enter customer information below:")

    c1, c2, c3 = st.columns(3)
    with c1:
        credit_score = st.number_input(
            "💳 Credit Score", min_value=300, max_value=850, value=650)
        age = st.number_input(
            "🎂 Age", min_value=18, max_value=92, value=35)
        tenure = st.number_input(
            "📅 Tenure (years)", min_value=0, max_value=10, value=5)
        balance = st.number_input(
            "💰 Balance", min_value=0, max_value=300000, value=75000)

    with c2:
        num_products = st.selectbox("📦 Num of Products", [1, 2, 3, 4])
        has_cr_card = st.selectbox(
            "💳 Has Credit Card", [1, 0],
            format_func=lambda x: "Yes ✅" if x == 1 else "No ❌")
        is_active = st.selectbox(
            "⚡ Is Active Member", [1, 0],
            format_func=lambda x: "Yes ✅" if x == 1 else "No ❌")
        salary = st.number_input(
            "💵 Estimated Salary", min_value=0, max_value=200000, value=90000)

    with c3:
        geography = st.selectbox("🌍 Geography", ["France", "Germany", "Spain"])
        gender = st.selectbox("👤 Gender", ["Male", "Female"])
        st.markdown("&nbsp;", unsafe_allow_html=True)
        st.markdown("&nbsp;", unsafe_allow_html=True)
        predict_btn = st.button(
            "🔮 Predict This Customer",
            use_container_width=True,
            type="primary"
        )

    if predict_btn:
        auto_query = f"""
        Customer details:
        - CreditScore: {credit_score}
        - Age: {age}
        - Tenure: {tenure}
        - Balance: {balance}
        - NumOfProducts: {num_products}
        - HasCrCard: {has_cr_card}
        - IsActiveMember: {is_active}
        - EstimatedSalary: {salary}
        - Geography: {geography}
        - Gender: {gender}
        """
        st.session_state.auto_query = auto_query
        st.rerun()

st.markdown("---")
st.markdown("### 💬 Chat with the Agent")

# ── Chat History ──────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "auto_query" not in st.session_state:
    st.session_state.auto_query = None

# ── Welcome Message ───────────────────────────────────────────────────
if len(st.session_state.messages) == 0:
    with st.chat_message("assistant"):
        st.markdown("""
👋 **Welcome to the Agentic Churn Prediction System!**

I'm powered by **Groq Llama 3.3 70B** with **3 ML models** ready:

| Model | Best For | Accuracy |
|-------|----------|----------|
| ⚡ SVM | Clear numerical boundaries | 85.6% |
| 🌲 Decision Tree | Interpretable rules | 85.9% |
| 🧠 Neural Network | Complex patterns | 86.1% |

**Try typing something like:**
> *"45-year-old female from Germany, credit score 650, balance 130000, not active member..."*

I'll reason about which model fits best and explain the prediction! 🎯

**📄 Note:** Predictions are cached to prevent rate limits. Same requests will be instant!
        """)

# Display existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ── Helper to Run Agent & Display Result ──────────────────────────────
@st.cache_data(ttl=3600)
def get_cached_prediction(prompt_hash: str, prompt: str):
    """Cache predictions to avoid rate limits. TTL = 1 hour"""
    return classify_customer(prompt)


def run_agent_and_display(prompt: str):
    """Runs agent and displays result in chat."""

    # Create hash of prompt for caching
    prompt_hash = hashlib.md5(prompt.encode()).hexdigest()

    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Run agent and show result (CACHED if same prompt)
    with st.chat_message("assistant"):
        with st.spinner("🧠 Agent is reasoning... please wait"):
            result = get_cached_prediction(prompt_hash, prompt)

        # Show reasoning trace
        if result["steps"]:
            with st.expander(
                "🧠 Agent Reasoning Trace — Click to Expand",
                expanded=False
            ):
                for i, step in enumerate(result["steps"], 1):
                    st.markdown(f"**Step {i}:**")
                    st.code(step, language="text")

        # ── Show as CARDS LAYOUT ──────────────────────────────────────
        try:
            data = result.get("structured", {})
            
            # Prediction Card
            pred = data.get("prediction", "N/A")
            if "CHURN" in pred.upper():
                st.error(f"⚠️ **PREDICTION: {pred}**\n\nImmediate action needed!")
            else:
                st.success(f"✅ **PREDICTION: {pred}**\n\nRelationship is healthy!")
            
            st.markdown("---")
            
            # Model & Confidence Card
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"🤖 **Model Used**\n\n{data.get('model_used', 'N/A')}")
            with col2:
                st.info(f"📊 **Confidence**\n\n{data.get('confidence', 'N/A')}")
            
            # Why Chosen Card
            st.markdown(f"💡 **Why This Model?**\n\n{data.get('why_chosen', 'N/A')}")
            st.markdown("---")
            
            # Key Factors Card
            st.markdown("🔑 **Key Factors Influencing Decision**")
            factors = data.get("key_factors", [])
            if factors:
                for factor in factors:
                    st.markdown(f"- {factor}")
            else:
                st.markdown("- *No factors extracted*")
            
            st.markdown("---")
            
            # Recommendation Card
            st.markdown(f"🏦 **Bank Recommendation**\n\n{data.get('recommendation', 'N/A')}")
            st.markdown("---")
            
        except Exception as e:
            st.error(f"❌ Error displaying results: {str(e)}")
            st.markdown("### Raw Output")
            st.markdown(result.get("output", "No output"))

        # Save to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["output"]
        })


# ── Handle Auto Query from Form ───────────────────────────────────────
if st.session_state.auto_query:
    prompt = st.session_state.auto_query
    st.session_state.auto_query = None
    run_agent_and_display(prompt)

# ── Chat Input Box ────────────────────────────────────────────────────
if prompt := st.chat_input(
    "💬 Describe a bank customer... e.g. '45-year-old female from Germany, balance 130000, not active...'"
):
    run_agent_and_display(prompt)