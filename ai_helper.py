# ai_helper.py
import json
import re
from typing import Any, Dict, Optional

import streamlit as st
from google import genai


SYSTEM_MESSAGE = """
You are an AI assistant embedded in FluoroSelect, a Streamlit app for fluorophore
selection in multiplexed fluorescence microscopy.

Important rules:
- Do not invent fluorophores, probes, lasers, metrics, or experimental results.
- Do not directly choose the final panel unless the optimizer result is provided.
- For input parsing, return only valid JSON.
- For result explanation, base your answer only on the provided structured data.
- Make clear that the optimization result is computed by FluoroSelect, not by AI.
- Use concise scientific language.
"""


def get_gemini_client() -> Optional[genai.Client]:
    api_key = st.secrets.get("GEMINI_API_KEY", None)
    if not api_key:
        return None
    return genai.Client(api_key=api_key)


def get_model_name() -> str:
    return st.secrets.get("GEMINI_MODEL", "gemini-2.5-flash")


def call_gemini(prompt: str) -> str:
    client = get_gemini_client()
    if client is None:
        return (
            "Gemini API key is not configured. "
            "Please add GEMINI_API_KEY to Streamlit secrets."
        )

    response = client.models.generate_content(
        model=get_model_name(),
        contents=f"{SYSTEM_MESSAGE}\n\n{prompt}",
    )
    return response.text or ""


def extract_json(text: str) -> Dict[str, Any]:
    """
    Robustly extract JSON from model output.
    Gemini should return JSON only, but this protects against accidental markdown.
    """
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        raise ValueError(f"No JSON object found in Gemini output:\n{text}")

    return json.loads(match.group(0))


def parse_user_request(user_text: str, app_context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert natural-language user request into a structured FluoroSelect request.

    The output is only a suggested UI state. app.py must validate all fields.
    """
    prompt = f"""
Convert the user's natural-language request into a structured JSON object for FluoroSelect.

Available app context:
{json.dumps(app_context, indent=2, ensure_ascii=False)}

User request:
{user_text}

Return only JSON with this schema:
{{
  "mode": "Emission spectra" or "Predicted spectra" or null,
  "laser_strategy": "Simultaneous" or "Separate" or null,
  "spectral_resolution": "1 nm (general)" or "33 detection channels (Valm lab)" or null,
  "lasers": [405, 488, 561, 639] or [],
  "selection_source": "By probes" or "From readout pool" or "All fluorophores" or "EUB338 only" or null,

  "fixed_probe_fluorophore_pairs": [
    {{"probe": "probe name from context", "fluorophore": "fluorophore name from context"}}
  ],

  "additional_probes": ["probe name from context"],
  "fixed_fluorophores": ["fluorophore name from context"],
  "candidate_fluorophores": ["fluorophore name from context"],
  "n_additional_fluorophores": null or integer,

  "warnings": ["anything ambiguous or unsupported"]
}}

Rules:
- Use only probe and fluorophore names that appear in the provided app context.
- If the user asks to fix a specific probe with a specific fluorophore, put it in fixed_probe_fluorophore_pairs.
- If the user asks to choose other probes, put them in additional_probes if explicitly named.
- If the user only gives a number of additional probes but no names, put a warning.
- If the user asks for pool selection, use fixed_fluorophores, candidate_fluorophores, and n_additional_fluorophores.
- If ambiguous, set the field to null or [] and add a warning.
"""
    raw = call_gemini(prompt)
    return extract_json(raw)


def explain_result(result_context: Dict[str, Any]) -> str:
    prompt = f"""
Explain this FluoroSelect result.

Structured result:
{json.dumps(result_context, indent=2, ensure_ascii=False)}

Please explain:
1. Why this selected panel is good or risky.
2. Which fluorophore pairs are most concerning.
3. Whether the current laser/spectral-resolution setting seems appropriate.
4. Any limitations in the interpretation.

Do not invent additional data.
"""
    return call_gemini(prompt)


def suggest_improvements(result_context: Dict[str, Any]) -> str:
    prompt = f"""
Suggest practical improvements for this FluoroSelect result.

Structured result:
{json.dumps(result_context, indent=2, ensure_ascii=False)}

Please suggest:
1. Which selected fluorophore pairs may need replacement.
2. Whether changing laser settings may help.
3. Whether reducing or increasing panel size may help.
4. What to test next experimentally.

Do not invent fluorophore properties that are not supported by the result context.
"""
    return call_gemini(prompt)


def answer_light_question(question: str, app_context: Dict[str, Any], result_context: Optional[Dict[str, Any]] = None) -> str:
    prompt = f"""
Answer the user's question about FluoroSelect.

Available app context:
{json.dumps(app_context, indent=2, ensure_ascii=False)}

Current result context:
{json.dumps(result_context or {}, indent=2, ensure_ascii=False)}

User question:
{question}

Answer concisely. If the answer requires data not provided, say what is missing.
"""
    return call_gemini(prompt)


def generate_methods_text(result_context: Dict[str, Any]) -> str:
    prompt = f"""
Generate a concise Methods-style paragraph for this FluoroSelect run.

Structured result:
{json.dumps(result_context, indent=2, ensure_ascii=False)}

Requirements:
- Scientific writing style.
- Do not overclaim.
- State that fluorophore selection was performed by the FluoroSelect optimization algorithm.
- Mention fixed constraints if present.
- Mention AI only if relevant as an interface assistant, not as the selection algorithm.
"""
    return call_gemini(prompt)
