from __future__ import annotations

import json
import os
import re
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DIMENSIONS = [
    "Demand Strength",
    "Pricing Power",
    "Execution",
    "Capex Pressure",
    "Balance-Sheet Risk",
    "Governance",
    "Competitive Position",
    "Disruption Risk",
]
POSITIVE_DIMENSIONS = ["Demand Strength", "Pricing Power", "Execution", "Governance", "Competitive Position"]
RISK_DIMENSIONS = ["Capex Pressure", "Balance-Sheet Risk", "Disruption Risk"]

KEYWORDS = {
    "Demand Strength": {
        "positive": ["demand", "volume", "sales growth", "occupancy", "leasing", "organic growth", "销量", "需求", "增长", "出租率"],
        "negative": ["weak demand", "declining sales", "lower traffic", "vacancy", "需求疲软", "销量下降"],
    },
    "Pricing Power": {
        "positive": ["pricing power", "price", "margin", "rent spread", "cash rent", "提价", "利润率", "租差"],
        "negative": ["price pressure", "margin compression", "discount", "降价", "利润率压力"],
    },
    "Execution": {
        "positive": ["execution", "raised guide", "guidance", "record", "delivery", "执行", "指引", "上调"],
        "negative": ["missed guidance", "execution risk", "delay", "下调指引", "执行风险", "延期"],
    },
    "Capex Pressure": {
        "positive": ["low capex", "capital light", "低资本开支"],
        "negative": ["capex", "capital expenditure", "development", "investment plan", "资本开支", "开发", "投资计划"],
    },
    "Balance-Sheet Risk": {
        "positive": ["strong balance sheet", "low debt", "liquidity", "低负债", "流动性"],
        "negative": ["debt", "leverage", "refinancing", "balance-sheet risk", "负债", "杠杆", "再融资"],
    },
    "Governance": {
        "positive": ["governance", "incentive", "discipline", "management", "治理", "激励", "管理层"],
        "negative": ["governance risk", "conflict", "governance concern", "治理风险", "利益冲突"],
    },
    "Competitive Position": {
        "positive": ["moat", "brand", "barrier", "distribution", "competitive", "护城河", "品牌", "壁垒", "分销"],
        "negative": ["competition", "disruption", "竞争", "替代"],
    },
    "Disruption Risk": {
        "positive": ["resilient", "defensive", "inelastic", "防御", "韧性"],
        "negative": ["disruption", "obsolescence", "technology risk", "颠覆", "淘汰", "技术风险"],
    },
}


def load_documents(path: str | Path) -> pd.DataFrame:
    documents = pd.read_csv(path, encoding="utf-8-sig")
    required = {"ticker", "document_id", "document_date", "document_type", "text", "source_url"}
    missing = required.difference(documents.columns)
    if missing:
        raise ValueError(f"Missing narrative document columns: {sorted(missing)}")
    if documents["document_id"].duplicated().any():
        raise ValueError("document_id must be unique")
    if documents["text"].isna().any():
        raise ValueError("Narrative documents cannot have empty text")
    if "is_demo" not in documents:
        documents["is_demo"] = False
    documents["is_demo"] = documents["is_demo"].map(
        lambda value: value.strip().lower() in {"true", "1", "yes", "y"} if isinstance(value, str) else bool(value)
    )
    return documents


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?。！？])\s+|[\n\r]+", str(text))
    return [part.strip() for part in parts if part and part.strip()]


def _role_overall(role: str, dimension_scores: dict[str, float]) -> float:
    positive = float(np.mean([dimension_scores[name] for name in POSITIVE_DIMENSIONS]))
    risk = float(np.mean([dimension_scores[name] for name in RISK_DIMENSIONS]))
    defensive = 6.0 - risk
    if role == "Risk Analyst":
        value = 0.60 * positive + 0.40 * defensive - 0.10
    elif role == "Macro Analyst":
        value = 0.80 * positive + 0.20 * 3.2
    elif role == "Devil's Advocate":
        value = 0.75 * positive + 0.25 * defensive - 0.35
    elif role == "Client-Liability Analyst":
        value = 0.55 * positive + 0.45 * defensive - 0.10
    else:
        value = positive
    return float(np.clip(value, 1, 5))


class ExtractiveNarrativeProvider:
    """Deterministic offline fallback. It is not an LLM and always triggers review."""

    name = "extractive_fallback"

    def analyze(self, document: pd.Series, role: str) -> dict[str, Any]:
        sentences = _sentences(document["text"])
        dimensions = []
        for name in DIMENSIONS:
            keywords = KEYWORDS[name]
            evidence = [
                sentence for sentence in sentences
                if any(word.lower() in sentence.lower() for word in keywords["positive"] + keywords["negative"])
            ]
            positive = sum(
                sentence.lower().count(word.lower()) for sentence in evidence for word in keywords["positive"]
            )
            negative = sum(
                sentence.lower().count(word.lower()) for sentence in evidence for word in keywords["negative"]
            )
            score = float(np.clip(3 + 0.45 * (positive - negative), 1, 5))
            direction = int(np.sign(positive - negative))
            confidence = 0.25 if not evidence else min(0.65, 0.35 + 0.08 * len(evidence))
            dimensions.append({
                "dimension": name,
                "score": score,
                "confidence": confidence,
                "direction_of_change": direction,
                "evidence": evidence[0][:700] if evidence else "No direct evidence extracted from the supplied document.",
                "evidence_found": bool(evidence),
            })
        scores = {item["dimension"]: item["score"] for item in dimensions}
        overall = _role_overall(role, scores)
        confidence = float(np.mean([item["confidence"] for item in dimensions]))
        return {
            "overall_score": overall,
            "overall_confidence": confidence,
            "rationale": "Offline keyword/sentence extraction; mandatory human verification required.",
            "dimensions": dimensions,
        }


class OpenAICompatibleProvider:
    """LLM adapter for an OpenAI-compatible /chat/completions endpoint."""

    name = "openai_compatible_llm"

    def __init__(self, base_url: str, api_key: str, model: str, timeout: int = 90):
        if not base_url or not api_key or not model:
            raise ValueError("GAIA_LLM_BASE_URL, GAIA_LLM_API_KEY, and GAIA_LLM_MODEL are required")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    @classmethod
    def from_env(cls) -> "OpenAICompatibleProvider":
        return cls(
            os.getenv("GAIA_LLM_BASE_URL", ""),
            os.getenv("GAIA_LLM_API_KEY", ""),
            os.getenv("GAIA_LLM_MODEL", ""),
            int(os.getenv("GAIA_LLM_TIMEOUT", "90")),
        )

    def analyze(self, document: pd.Series, role: str) -> dict[str, Any]:
        schema = {
            "overall_score": "number 1-5",
            "overall_confidence": "number 0-1",
            "rationale": "short string",
            "dimensions": [{
                "dimension": "one of " + ", ".join(DIMENSIONS),
                "score": "number 1-5",
                "confidence": "number 0-1",
                "direction_of_change": "-1, 0, or 1",
                "evidence": "exact short quote from the document",
            }],
        }
        prompt = f"""
You are the {role} on an investment committee. Analyze only the supplied document.
Return strict JSON with every required dimension. Do not invent facts. Score investment
attractiveness from 1 (very poor) to 5 (very strong). Confidence is 0-1. Direction
means change in the information set: -1 deteriorating, 0 unchanged/unclear, 1 improving.
Every evidence field must be an exact quote from the document or 'No direct evidence'.

Required JSON shape: {json.dumps(schema)}
Document type: {document['document_type']}
Document date: {document['document_date']}
Document text:
{document['text']}
"""
        request = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps({
                "model": self.model,
                "temperature": 0,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": "You produce evidence-grounded investment analysis as strict JSON."},
                    {"role": "user", "content": prompt},
                ],
            }).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        content = payload["choices"][0]["message"]["content"]
        content = re.sub(r"^```(?:json)?|```$", "", content.strip(), flags=re.MULTILINE).strip()
        return json.loads(content)


def build_provider(name: str):
    if name == "extractive":
        return ExtractiveNarrativeProvider()
    if name == "openai":
        return OpenAICompatibleProvider.from_env()
    raise ValueError("Narrative provider must be 'extractive' or 'openai'")


class NarrativeEngine:
    def __init__(self, roles: list[str], min_confidence: float, max_ads: float):
        self.roles = roles
        self.min_confidence = min_confidence
        self.max_ads = max_ads

    def run(self, documents: pd.DataFrame, provider) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        dimension_rows: list[dict] = []
        agent_rows: list[dict] = []
        for _, document in documents.iterrows():
            for role in self.roles:
                result = provider.analyze(document, role)
                by_name = {item["dimension"]: item for item in result.get("dimensions", [])}
                missing = set(DIMENSIONS).difference(by_name)
                if missing:
                    raise ValueError(f"Narrative provider omitted dimensions: {sorted(missing)}")
                for name in DIMENSIONS:
                    item = by_name[name]
                    score = float(item["score"])
                    confidence = float(item["confidence"])
                    direction = int(item["direction_of_change"])
                    if not 1 <= score <= 5 or not 0 <= confidence <= 1 or direction not in {-1, 0, 1}:
                        raise ValueError("Invalid narrative score, confidence, or direction")
                    evidence_text = str(item.get("evidence", ""))
                    if provider.name == "openai_compatible_llm" and evidence_text != "No direct evidence" and evidence_text not in str(document["text"]):
                        raise ValueError("LLM evidence must be an exact quote from the supplied document")
                    dimension_rows.append({
                        "ticker": document["ticker"],
                        "document_id": document["document_id"],
                        "document_date": document["document_date"],
                        "agent": role,
                        "dimension": name,
                        "score": score,
                        "confidence": confidence,
                        "direction_of_change": direction,
                        "evidence": evidence_text,
                        "source_url": document["source_url"],
                        "provider": provider.name,
                        "is_demo": bool(document.get("is_demo", False)),
                    })
                overall_score = float(result["overall_score"])
                overall_confidence = float(result["overall_confidence"])
                if not 1 <= overall_score <= 5 or not 0 <= overall_confidence <= 1:
                    raise ValueError("Invalid agent overall assessment")
                agent_rows.append({
                    "ticker": document["ticker"],
                    "document_id": document["document_id"],
                    "agent": role,
                    "overall_score": overall_score,
                    "overall_confidence": overall_confidence,
                    "rationale": str(result.get("rationale", "")),
                    "provider": provider.name,
                    "is_demo": bool(document.get("is_demo", False)),
                })

        dimensions = pd.DataFrame(dimension_rows)
        agents = pd.DataFrame(agent_rows)
        summaries = []
        for ticker, group in agents.groupby("ticker"):
            dimension_group = dimensions[dimensions["ticker"] == ticker]
            weights = dimension_group["confidence"].clip(lower=0.01)
            delta = float(np.average(dimension_group["direction_of_change"], weights=weights))
            ads = float(group["overall_score"].std(ddof=0))
            confidence = float(group["overall_confidence"].mean())
            provider_name = str(group["provider"].iloc[0])
            reasons = []
            if provider_name != "openai_compatible_llm":
                reasons.append("OFFLINE_EXTRACTIVE_NOT_LLM")
            if confidence < self.min_confidence:
                reasons.append("LOW_AI_CONFIDENCE")
            if ads > self.max_ads:
                reasons.append("HIGH_AGENT_DISAGREEMENT")
            if bool(group["is_demo"].any()):
                reasons.append("DEMO_DOCUMENTS")
            slow = dimension_group[dimension_group["dimension"].isin(POSITIVE_DIMENSIONS)]
            summaries.append({
                "ticker": ticker,
                "ai_consensus": float(group["overall_score"].mean() / 5.0),
                "ai_confidence": confidence,
                "ads": ads,
                "narrative_delta": delta,
                "slow_structural_signal": float((slow["score"].mean() - 1) / 4.0),
                "fast_information_delta": delta,
                "provider": provider_name,
                "ai_gate": "PROGRESS_TO_HUMAN_IC" if not reasons else "REVIEW_REQUIRED",
                "gate_reasons": "|".join(reasons),
                "demo_only": bool(group["is_demo"].any()),
            })
        return dimensions, agents, pd.DataFrame(summaries)
