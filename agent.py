# agent.py
import os
import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from ml_tools import ALL_TOOLS

# ── 1. Load Environment Variables ─────────────────────────────────────
load_dotenv()

# ── 2. Configure Groq LLM ─────────────────────────────────────────────
print("🤖 Initializing Groq LLM...")
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY"),
)
print("✅ LLM ready!")

# ── 3. Build LangGraph ReAct Agent ────────────────────────────────────
print("⚙️  Building ReAct agent...")
graph = create_react_agent(
    model=llm,
    tools=ALL_TOOLS,
)
print("✅ Agent ready!")


# ── 4. Main Classification Function ───────────────────────────────────
def classify_customer(user_query: str) -> dict:
    """
    Takes natural language customer description
    and returns prediction + reasoning trace.
    """

    full_prompt = f"""
You are an expert ML orchestrator for a bank customer churn prediction system.

You have access to 3 ML models as tools:
- classify_with_svm: Good for structured numerical data with clear boundaries
- classify_with_decision_tree: Good for interpretable rule-based predictions
- classify_with_neural_network: Best overall accuracy for complex patterns

Your job:
1. Read the customer description carefully
2. Extract all features from it
3. Choose the best ML model tool
4. Call the tool with the extracted features as individual arguments
5. After getting the result give a structured final answer

Customer Description:
{user_query}

After calling the model, provide analysis in this format:
PREDICTION: [CHURN or STAY]
CONFIDENCE: [the confidence percentage from model]
MODEL USED: [name of model you chose]
WHY THIS MODEL: [brief explanation]
KEY FACTORS: [list the main factors that led to this prediction]
RECOMMENDATION: [what the bank should do]
"""

    # ── Run the graph ─────────────────────────────────────────────────
    result = graph.invoke({
        "messages": [HumanMessage(content=full_prompt)]
    })

    # ── Extract final answer ──────────────────────────────────────────
    all_messages = result["messages"]
    final_answer = all_messages[-1].content

    # ── Extract reasoning steps ───────────────────────────────────────
    steps = []
    for msg in all_messages:
        msg_type = type(msg).__name__

        if msg_type == "AIMessage":
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    steps.append(
                        f"🔧 Action: {tc['name']}\n"
                        f"   Input: {tc['args']}"
                    )

        elif msg_type == "ToolMessage":
            steps.append(
                f"📊 Observation from {msg.name}:\n"
                f"   {msg.content}"
            )

    # ── Parse structured output ────────────────────────────────────────
    def parse_output(text):
        """Extract structured data from LLM response with flexible parsing."""
        data = {
            "prediction": "N/A",
            "confidence": "N/A",
            "model_used": "N/A",
            "why_chosen": "N/A",
            "key_factors": [],
            "recommendation": "N/A",
            "raw": text
        }
        
        text_upper = text.upper()
        lines = text.split('\n')
        
        for line in lines:
            line_upper = line.upper()
            
            # Parse prediction (look for CHURN or STAY)
            if "PREDICTION" in line_upper or "PREDICT" in line_upper:
                if "CHURN" in line_upper:
                    data["prediction"] = "CHURN"
                elif "STAY" in line_upper:
                    data["prediction"] = "STAY"
                else:
                    data["prediction"] = line.split(":")[-1].strip() if ":" in line else "N/A"
            
            # Parse confidence
            if "CONFIDENCE" in line_upper:
                data["confidence"] = line.split(":")[-1].strip() if ":" in line else "N/A"
            
            # Parse model used
            if "MODEL" in line_upper and "USED" in line_upper:
                data["model_used"] = line.split(":")[-1].strip() if ":" in line else "N/A"
            elif "MODEL" in line_upper and ("SVM" in line_upper or "TREE" in line_upper or "NETWORK" in line_upper or "NEURAL" in line_upper):
                data["model_used"] = line.split(":")[-1].strip() if ":" in line else "N/A"
            
            # Parse why chosen
            if "WHY" in line_upper or "REASON" in line_upper:
                data["why_chosen"] = line.split(":")[-1].strip() if ":" in line else "N/A"
            
            # Parse key factors
            if "FACTOR" in line_upper or "DROVE" in line_upper or "INFLUENCED" in line_upper:
                if ":" in line:
                    factors_str = line.split(":")[-1].strip()
                    data["key_factors"] = [f.strip() for f in factors_str.split(',') if f.strip()]
            
            # Parse recommendation
            if "RECOMMENDATION" in line_upper or "RECOMMEND" in line_upper:
                data["recommendation"] = line.split(":")[-1].strip() if ":" in line else "N/A"
        
        # If no prediction found but text contains CHURN/STAY
        if data["prediction"] == "N/A":
            if "CHURN" in text_upper:
                data["prediction"] = "CHURN"
            elif "STAY" in text_upper:
                data["prediction"] = "STAY"
        
        return data

    parsed = parse_output(final_answer)

    return {
        "output": final_answer,
        "steps": steps,
        "structured": parsed
    }


# ── 5. Terminal Test ───────────────────────────────────────────────────
if __name__ == "__main__":
    test_query = """
    A 45-year-old female customer from Germany.
    Her credit score is 650, balance is 130000,
    tenure is 2 years, has 1 product, has a credit card,
    is NOT an active member, estimated salary is 85000.
    """

    print("\n🧪 Testing agent with sample customer...\n")
    print("=" * 50)

    result = classify_customer(test_query)

    print("\n🧠 AGENT REASONING TRACE:")
    print("-" * 50)
    if result["steps"]:
        for i, step in enumerate(result["steps"], 1):
            print(f"Step {i}:")
            print(step)
            print()
    else:
        print("No steps captured")

    print("\n📤 FINAL ANSWER:")
    print("=" * 50)
    print(result["output"])
    print("=" * 50)