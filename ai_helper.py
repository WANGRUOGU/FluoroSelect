# ai_helper.py
import json
import re
from typing import Any, Dict, Optional

import streamlit as st

try:
    from google import genai
except Exception:
    genai = None


SYSTEM_MESSAGE = """
You are an AI assistant embedded in FluoroSelect, a Streamlit app for fluorophore selection in multiplexed fluorescence microscopy.

Rules:
- Do not invent fluorophores, probes, lasers, similarity metrics, soft-penalty settings, or experimental results.
- Do not directly choose the final panel unless an optimizer result is provided.
- For input parsing, return only valid JSON.
- For result explanation, base your answer only on the provided structured data.
- Make clear that the optimization result is computed by FluoroSelect, not by AI.
- Do not assume that the pairwise score is cosine similarity. Use the provided similarity_metric.
- Use concise scientific language.
"""


def get_gemini_client() -> Optional[Any]:
    if genai is None:
        return None

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
  "spectral_resolution": "1 nm" or "9.8 nm" or null,
  "similarity_metric": "Cosine similarity" or "Spectral overlap" or "Pearson correlation" or "Spectral angle similarity" or null,
  "low_priority_fluorophores": ["fluorophore name from context"],
  "soft_penalty_strength": "Low" or "Medium" or "High" or null,
  "lasers": [488, 561, 639] or [],
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

Important interpretation rules:
- Use only probe and fluorophore names that appear in the provided app context.
- Default mode is "Emission spectra" unless the user explicitly says predicted/effective spectra or mentions laser-based spectra.
- Default lasers are [488, 561, 639].
- Default laser_strategy is "Simultaneous".
- Default spectral_resolution is "1 nm".
- Default similarity_metric is "Cosine similarity".
- Default soft_penalty_strength is "Medium" when low_priority_fluorophores is non-empty.

Similarity metric rules:
- If the user asks for cosine similarity, use "Cosine similarity".
- If the user asks for overlap or area overlap, use "Spectral overlap".
- If the user asks for Pearson, correlation, or shape correlation, use "Pearson correlation".
- If the user asks for spectral angle, angle, or hyperspectral angle, use "Spectral angle similarity".
- If the user does not mention a metric, use "Cosine similarity".

Soft-penalty rules:
- If the user says to avoid a fluorophore if possible, lower its priority, deprioritize it, penalize it, or mark it as less preferred, put that fluorophore in low_priority_fluorophores.
- Soft penalty is not a hard exclusion. Do not put these fluorophores in candidate_fluorophores or remove them from candidates.
- If the user says strongly avoid, high penalty, or avoid as much as possible, use soft_penalty_strength = "High".
- If the user says slightly avoid, mild penalty, or low penalty, use soft_penalty_strength = "Low".
- Otherwise, use soft_penalty_strength = "Medium".
- Example: "avoid AF633 and Pacific Orange if possible" means low_priority_fluorophores = ["AF633", "Pacific Orange"], soft_penalty_strength = "Medium".

Pool-selection rules:
- If the user asks to select/choose/pick N fluorophores, use selection_source = "All fluorophores".
- If the user says "fix PROBE with FLUOROPHORE and choose another N fluorophores", use selection_source = "All fluorophores", fixed_fluorophores = [FLUOROPHORE], n_additional_fluorophores = N.
- If the user says "fix PROBE with FLUOROPHORE" without a number, use selection_source = "All fluorophores", fixed_fluorophores = [FLUOROPHORE], n_additional_fluorophores = 4.
- In pool mode, do not put the fixed probe-fluorophore pair into fixed_probe_fluorophore_pairs. Put only the fluorophore into fixed_fluorophores.
- If candidate_fluorophores is not specified by the user, return [].

By-probes rules:
- If the user says they want to use specific probes, such as "use EUB338 and ACT476", use selection_source = "By probes" and put those probes in additional_probes.
- In By probes mode, the user does NOT need to specify fluorophores.
- In By probes mode, FluoroSelect optimizer will choose one fluorophore for each probe.
- Do not ask the user to specify fluorophores for each probe.
- If the user gives only probe names and no fluorophore names, put the probe names in additional_probes.
- Only use fixed_probe_fluorophore_pairs when the user explicitly says a specific probe must be fixed with a specific fluorophore as a pair.

Default number rule:
- If selection_source is "All fluorophores" and the user does not specify a number, use n_additional_fluorophores = 4.
- If the user says "choose N fluorophores" and there are no fixed fluorophores, use n_additional_fluorophores = N.
- If the user says "choose another N fluorophores" and fixed_fluorophores is non-empty, use n_additional_fluorophores = N.

Warnings:
- Add a warning only when a requested probe/fluorophore/metric/soft-penalty item is not found or the request is truly unsupported.
- Do not warn merely because default settings were used.
- Do not answer with explanatory text. Return JSON only.
"""

    raw = call_gemini(prompt)
    return extract_json(raw)


def explain_result(result_context: Dict[str, Any]) -> str:
    prompt = f"""
Explain this FluoroSelect result very briefly.

Structured result:
{json.dumps(result_context, indent=2, ensure_ascii=False)}

Rules:
- Maximum 4 bullet points.
- Base the explanation only on optimizer outputs, selected labels, top pairwise scores, similarity_metric, low_priority_fluorophores, lasers, and metrics.
- Do not assume the pairwise score is cosine similarity. Use the provided similarity_metric.
- Do not repeat all selected fluorophores.
- Do not give generic microscopy background.
- Do not say "validation is needed" unless a specific metric or score suggests risk.
- If the top pairwise score is not high, say the panel looks reasonably separated under the chosen metric.
- If lower-priority fluorophores were selected, briefly mention that they were used despite the soft penalty.
- If the top pairwise score is high, identify the riskiest pair and say why it may be difficult.

Output format:
- Overall:
- Main risk:
- Most concerning pair:
- Practical note:
"""

    return call_gemini(prompt)


def suggest_improvements(result_context: Dict[str, Any]) -> str:
    prompt = f"""
Give concise next-step suggestions for this FluoroSelect result.

Structured result:
{json.dumps(result_context, indent=2, ensure_ascii=False)}

Rules:
- Maximum 4 bullet points.
- Base suggestions only on selected labels, top pairwise scores, similarity_metric, low_priority_fluorophores, lasers, mode, and metrics.
- Do not assume the pairwise score is cosine similarity. Use the provided similarity_metric.
- If a lower-priority fluorophore was selected, say it was probably needed to preserve the optimized spectral score or constraints.
- If a pair has a high score under the chosen metric, suggest replacing or constraining one member of that pair.
- If metrics are weak, suggest reducing panel size or changing candidate constraints.
- If the result looks good, say no major change is needed and suggest testing the panel experimentally.
- No generic advice.
- No long explanation.

Output format:
- Keep:
- Check:
- Try next:
- Avoid:
"""

    return call_gemini(prompt)


def answer_light_question(
    question: str,
    app_context: Dict[str, Any],
    result_context: Optional[Dict[str, Any]] = None,
) -> str:
    prompt = f"""
Answer the user's question about FluoroSelect briefly.

Available app context:
{json.dumps(app_context, indent=2, ensure_ascii=False)}

Current result context:
{json.dumps(result_context or {}, indent=2, ensure_ascii=False)}

User question:
{question}

Rules:
- Maximum 5 sentences.
- Answer based on FluoroSelect settings/results only.
- Do not assume that the similarity score is cosine similarity. Use the provided similarity_metric if present.
- If information is missing, say exactly what is missing.
- Do not give generic background.
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
- Mention the spectral mode, spectral resolution, and similarity_metric.
- Mention fixed constraints and soft-penalty fluorophores if present.
- Mention AI only if relevant as an interface assistant, not as the selection algorithm.
"""

    return call_gemini(prompt)
