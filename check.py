import json
import llm

# Invariant: 'parsed' is always a validated dict









EVALUATOR_PROMPT = {
    "role": "system",
    "content": (
        "You are an evaluator AI. "
        "Given an assistant answer, judge whether it is:\n"
        "- correct\n"
        "- clear\n"
        "- safe\n\n"
        "Respond ONLY in valid JSON with fields:\n"
        "{ \"score\": float between 0 and 1, \"issues\": [string]}"
    )
}





def evaluate_answer(answer: str, persona: str) -> dict:
    messages = [
        EVALUATOR_PROMPT,
        {
            "role": "user",
            "content": f"Answer to evaluate:\n{answer}"
        }
    ]

    raw = llm.llm_call(messages, persona)

    try:
        data = json.loads(raw)
    except Exception:
        return {
            "score": 0.0,
            "issues": ["Invalid evaluator JSON"]
        }
    
    if "score" not in data:
        data["score"] = 0.0
    if "issues" not in data:
        data["issues"] = []

    return data








def basic_confidence_check(parsed: dict) -> list:
    issues = []

    confidence = parsed.get("confidence")

    # Missing or Low confidence
    if confidence is None:
        issues.append("Missing confidence field")
    elif not isinstance(confidence, (int, float)):
        issues.append("Confidence is not numeric")
    elif confidence < 0.4:
        issues.append("Low model confidence")

    # Weak or empty answer
    answer = parsed.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        issues.append("Empty answer")
    elif len(answer.strip()) < 15:
        issues.append("Answer too short")

    return issues










LOW_SCORE_THRESHOLD = 0.5
MAX_ISSUES_ALLOWED = 2

def decide_response_action(parsed: dict) -> str:
    issues = parsed.get("warnings", [])
    evaluation = parsed.get("evaluation") or {}
    confidence = parsed.get("confidence", 0)
    score = evaluation.get("score", 0)

    if confidence < 0.4:
        return "low_confidence"

    if score < LOW_SCORE_THRESHOLD:
        return "warn"
    
    if len(issues) > MAX_ISSUES_ALLOWED:
        return "warn"
    
    return "ok"









# Off by default
def should_refuse(parsed: dict) -> bool:
    evaluation = parsed.get("evaluation", {})
    score = evaluation.get("score", 1.0)

    return score < 0.2