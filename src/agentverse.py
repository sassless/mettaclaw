import asyncio
import json
import os
import re
import time
from typing import Any, Optional

import requests
from uagents import Model
from uagents.communication import send_message_raw
from uagents.query import send_sync_message

AGENTVERSE_API_URL = os.environ.get("AGENTVERSE_API_URL", "https://agentverse.ai")
TECHNICAL_ANALYSIS_AGENT_ADDRESS = os.environ.get(
    "TECHNICAL_ANALYSIS_AGENT_ADDRESS",
    "agent1q085746wlr3u2uh4fmwqplude8e0w6fhrmqgsnlp49weawef3ahlutypvu6",
)
TAVILY_SEARCH_AGENT_ADDRESS = os.environ.get(
    "TAVILY_SEARCH_AGENT_ADDRESS",
    "agent1qt5uffgp0l3h9mqed8zh8vy5vs374jl2f8y0mjjvqm44axqseejqzmzx9v8",
)
DEFAULT_AGENT_SEARCH_LIMIT = 5
MAX_AGENT_SEARCH_LIMIT = 10
MAX_AGENT_SEARCH_SCAN = 30
DEFAULT_AGENT_SEARCH_PER_QUERY = 10
COMMON_PROTOCOL_NAMES = {
    "",
    "<unnamed>",
    "agentchatprotocol",
    "default",
    "healthprotocol",
}
SEARCH_STOPWORDS = {
    "a",
    "an",
    "and",
    "agent",
    "ai",
    "by",
    "for",
    "in",
    "language",
    "of",
    "on",
    "service",
    "the",
    "to",
    "tool",
    "with",
}
QUERY_EXPANSIONS = {
    "translate": ["translation", "translator"],
    "translation": ["translate", "translator"],
    "translator": ["translate", "translation"],
    "voice": ["speech", "audio", "tts"],
    "speech": ["voice", "audio", "tts"],
    "tts": ["voice", "speech", "audio"],
}


class WebSearchRequest(Model):
    query: str


class TechAnalysisRequest(Model):
    ticker: str


def _json_dumps(value: Any, indent: Optional[int] = None) -> str:
    return json.dumps(value, ensure_ascii=False, indent=indent, sort_keys=True)


def _format_json_text(text: str) -> str:
    try:
        return _json_dumps(json.loads(text), indent=2)
    except json.JSONDecodeError:
        return text


def _strip_digest_prefix(value: str, prefix: str) -> str:
    expected_prefix = f"{prefix}:"
    if value.startswith(expected_prefix):
        return value[len(expected_prefix) :]
    return value


def _normalize_digest(value: str, prefix: str) -> str:
    return f"{prefix}:{_strip_digest_prefix(value.strip(), prefix)}"


def _digest_suffix(value: str, prefix: str) -> str:
    return _strip_digest_prefix(value.strip(), prefix)


def _truncate_text(value: Any, limit: int) -> str:
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _format_tavily_results(response: str, max_results: int = 5) -> str:
    try:
        data = json.loads(response)
    except json.JSONDecodeError:
        return response

    if not isinstance(data, dict):
        return response

    results = data.get("results")
    if not isinstance(results, list):
        return response

    formatted = []
    for result in results[:max_results]:
        if not isinstance(result, dict):
            continue

        title = _truncate_text(result.get("title", ""), 160)
        url = _truncate_text(result.get("url", ""), 240)
        snippet = _truncate_text(result.get("content", ""), 400)

        parts = []
        if title:
            parts.append(f"TITLE: {title}")
        if url:
            parts.append(f"URL: {url}")
        if snippet:
            parts.append(f"SNIPPET: {snippet}")

        if parts:
            formatted.append(f"({' '.join(parts)})")

    return f"({' '.join(formatted)})" if formatted else response


def _agentverse_get(path: str, timeout: int = 10) -> Any:
    response = requests.get(f"{AGENTVERSE_API_URL}{path}", timeout=timeout)
    response.raise_for_status()
    return response.json()


def _agentverse_post(path: str, payload: dict[str, Any], timeout: int = 10) -> Any:
    response = requests.post(
        f"{AGENTVERSE_API_URL}{path}",
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def _fetch_agent_record(address: str, timeout: int = 10) -> dict[str, Any]:
    data = _agentverse_get(f"/v1/almanac/agents/{address}", timeout=timeout)
    if isinstance(data, list):
        if not data:
            raise ValueError(f"agent not found: {address}")
        data = data[0]
    if not isinstance(data, dict):
        raise ValueError(f"unexpected agent response for {address}: {type(data).__name__}")
    return data


def _fetch_protocol_manifest(protocol_digest: str, timeout: int = 10) -> dict[str, Any]:
    protocol_suffix = _digest_suffix(protocol_digest, "proto")
    data = _agentverse_get(
        f"/v1/almanac/manifests/protocols/{protocol_suffix}",
        timeout=timeout,
    )
    if isinstance(data, list):
        if not data:
            raise ValueError(f"protocol manifest not found: {protocol_digest}")
        data = data[0]
    if not isinstance(data, dict):
        raise ValueError(
            f"unexpected manifest response for {protocol_digest}: {type(data).__name__}"
        )
    return data


def _manifest_metadata(manifest: dict[str, Any]) -> dict[str, Any]:
    metadata = manifest.get("metadata") or {}
    if isinstance(metadata, list):
        for item in metadata:
            if isinstance(item, dict):
                return item
        return {}
    return metadata if isinstance(metadata, dict) else {}


def _manifest_models(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    models: dict[str, dict[str, Any]] = {}
    for model in manifest.get("models", []):
        if not isinstance(model, dict):
            continue
        digest = model.get("digest")
        schema = model.get("schema")
        if isinstance(digest, str) and isinstance(schema, dict):
            models[digest] = schema
    return models


def _collect_request_models(address: str, timeout: int = 10) -> list[dict[str, Any]]:
    agent = _fetch_agent_record(address, timeout=timeout)
    request_models: list[dict[str, Any]] = []

    for protocol_digest in agent.get("protocols", []):
        if not isinstance(protocol_digest, str):
            continue

        manifest = _fetch_protocol_manifest(protocol_digest, timeout=timeout)
        metadata = _manifest_metadata(manifest)
        model_index = _manifest_models(manifest)

        for interaction in manifest.get("interactions", []):
            if not isinstance(interaction, dict):
                continue

            request_digest = interaction.get("request")
            if not isinstance(request_digest, str):
                continue

            request_schema = model_index.get(request_digest)
            if request_schema is None:
                continue

            request_models.append(
                {
                    "protocol_name": metadata.get("name", ""),
                    "protocol_version": metadata.get("version", ""),
                    "protocol_digest": _normalize_digest(
                        str(metadata.get("digest", protocol_digest)),
                        "proto",
                    ),
                    "interaction_type": interaction.get("type", ""),
                    "request_digest": _normalize_digest(request_digest, "model"),
                    "request_schema": request_schema,
                    "response_digests": [
                        _normalize_digest(response_digest, "model")
                        for response_digest in interaction.get("responses", [])
                        if isinstance(response_digest, str)
                    ],
                }
            )

    return request_models


def _coerce_search_limit(
    limit: Any,
    default: int = DEFAULT_AGENT_SEARCH_LIMIT,
    max_limit: int = MAX_AGENT_SEARCH_LIMIT,
) -> int:
    try:
        value = int(limit)
    except (TypeError, ValueError):
        value = default
    return max(1, min(value, max_limit))


def _coerce_timeout_seconds(timeout: Any, default: float) -> float:
    try:
        value = float(timeout)
    except (TypeError, ValueError):
        value = default
    return max(0.0, value)


def _remaining_timeout(deadline: float) -> float:
    return max(0.0, deadline - time.monotonic())


def _search_agents(
    search_text: str,
    limit: int = DEFAULT_AGENT_SEARCH_LIMIT,
    timeout: int = 10,
    sort: str = "relevancy",
    direction: Optional[str] = None,
) -> list[dict[str, Any]]:
    normalized_search_text = " ".join(search_text.split())
    if not normalized_search_text:
        raise ValueError("search query must not be empty")
    if direction is None:
        direction = "asc" if sort == "relevancy" else "desc"

    data = _agentverse_post(
        "/v1/search/agents",
        {
            "filters": {"state": ["active"]},
            "sort": sort,
            "direction": direction,
            "search_text": normalized_search_text,
            "offset": 0,
            "limit": _coerce_search_limit(
                limit,
                default=DEFAULT_AGENT_SEARCH_LIMIT,
                max_limit=MAX_AGENT_SEARCH_SCAN,
            ),
        },
        timeout=timeout,
    )
    if not isinstance(data, dict):
        raise ValueError(f"unexpected search response: {type(data).__name__}")

    agents = data.get("agents")
    if not isinstance(agents, list):
        raise ValueError("unexpected search response: missing agents list")

    return [agent for agent in agents if isinstance(agent, dict)]


def _tokenize_search_text(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", value.casefold())


def _search_query_variants(search_text: str) -> list[str]:
    normalized_search_text = " ".join(search_text.split())
    tokens = _tokenize_search_text(normalized_search_text)
    variants: list[str] = []

    def add_variant(value: str) -> None:
        cleaned = " ".join(value.split())
        if cleaned and cleaned not in variants:
            variants.append(cleaned)

    add_variant(normalized_search_text)

    significant_tokens = [
        token
        for token in tokens
        if len(token) >= 4 and token not in SEARCH_STOPWORDS
    ]

    for token in significant_tokens:
        add_variant(token)
        for expanded in QUERY_EXPANSIONS.get(token, []):
            add_variant(expanded)

    if significant_tokens:
        add_variant(" ".join(significant_tokens))

    return variants[:6]


def _protocol_names(protocols: Any) -> list[str]:
    if not isinstance(protocols, list):
        return []
    names = []
    for protocol in protocols:
        if not isinstance(protocol, dict):
            continue
        name = str(protocol.get("name") or "").strip()
        if name:
            names.append(name)
    return names


def _searchable_agent_text(agent: dict[str, Any]) -> str:
    parts = [
        agent.get("name") or "",
        agent.get("description") or "",
        agent.get("readme") or "",
        agent.get("domain") or "",
    ]
    parts.extend(_protocol_names(agent.get("protocols")))
    return " ".join(str(part) for part in parts if part).casefold()


def _query_match_score(query_variants: list[str], agent: dict[str, Any]) -> int:
    haystack = _searchable_agent_text(agent)
    if not haystack:
        return 0

    original_tokens = _tokenize_search_text(query_variants[0]) if query_variants else []
    significant_tokens = [
        token
        for token in original_tokens
        if len(token) >= 4 and token not in SEARCH_STOPWORDS
    ]

    score = 0
    if query_variants:
        original_query = query_variants[0].casefold()
        if original_query in haystack:
            score += 40

    name = str(agent.get("name") or "").casefold()
    description = str(agent.get("description") or "").casefold()
    readme = str(agent.get("readme") or "").casefold()
    protocol_names = " ".join(_protocol_names(agent.get("protocols"))).casefold()

    for variant in query_variants[1:]:
        lowered_variant = variant.casefold()
        if lowered_variant in haystack:
            score += 20
        if lowered_variant and lowered_variant in name:
            score += 15
        elif lowered_variant and lowered_variant in protocol_names:
            score += 12
        elif lowered_variant and lowered_variant in description:
            score += 10
        elif lowered_variant and lowered_variant in readme:
            score += 8

    matched_tokens = 0
    for token in significant_tokens:
        if token in haystack:
            matched_tokens += 1
            score += 5
            if token in name:
                score += 6
            elif token in protocol_names:
                score += 5
            elif token in description:
                score += 4

    if significant_tokens and matched_tokens == len(significant_tokens):
        score += 15

    return score


def _specialized_protocol_count(protocols: Any) -> int:
    count = 0
    for protocol_name in _protocol_names(protocols):
        if protocol_name.casefold() not in COMMON_PROTOCOL_NAMES:
            count += 1
    return count


def _numeric(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _search_mode_bonus(sort: str) -> int:
    return {
        "relevancy": 8,
        "interactions": 12,
        "last-modified": 6,
    }.get(sort, 0)


def _candidate_pre_score(candidate: dict[str, Any], query_variants: list[str]) -> int:
    score = _query_match_score(query_variants, candidate)
    if score <= 0:
        return 0

    score += _search_mode_bonus(str(candidate.get("_best_sort") or ""))
    score += min(int(_numeric(candidate.get("recent_interactions")) // 100), 20)
    score += min(int(_numeric(candidate.get("total_interactions")) // 1000), 20)
    score += min(_specialized_protocol_count(candidate.get("protocols")) * 6, 18)

    if candidate.get("featured"):
        score += 5
    if str(candidate.get("category") or "").casefold() == "verified":
        score += 5

    return score


def _final_candidate_score(candidate: dict[str, Any], query_variants: list[str]) -> int:
    score = _candidate_pre_score(candidate, query_variants)
    if score <= 0:
        return 0
    score += min(int(_numeric(candidate.get("rating")) * 4), 20)
    score += min(_specialized_protocol_count(candidate.get("protocols")) * 4, 12)
    return score


def _merge_search_results(
    merged_candidates: dict[str, dict[str, Any]],
    raw_agents: list[dict[str, Any]],
    query_variants: list[str],
    sort: str,
) -> None:
    for position, raw_agent in enumerate(raw_agents):
        address = raw_agent.get("address")
        if not isinstance(address, str) or not address.strip():
            continue

        enriched_agent = dict(raw_agent)
        enriched_agent["_best_sort"] = sort
        enriched_agent["_positions"] = [f"{sort}:{position}"]

        existing = merged_candidates.get(address)
        if existing is None:
            merged_candidates[address] = enriched_agent
            continue

        existing["_positions"].append(f"{sort}:{position}")

        if not existing.get("description") and enriched_agent.get("description"):
            existing["description"] = enriched_agent["description"]
        if not existing.get("readme") and enriched_agent.get("readme"):
            existing["readme"] = enriched_agent["readme"]
        if not existing.get("protocols") and enriched_agent.get("protocols"):
            existing["protocols"] = enriched_agent["protocols"]

        existing_score = _candidate_pre_score(existing, query_variants)
        incoming_score = _candidate_pre_score(enriched_agent, query_variants)
        if incoming_score > existing_score:
            existing["_best_sort"] = sort


def _verify_search_results(
    raw_agents: list[dict[str, Any]],
    limit: int,
    query_variants: list[str],
    deadline: float,
) -> tuple[list[dict[str, Any]], bool]:
    verified_agents: list[dict[str, Any]] = []
    timed_out = False

    for raw_agent in raw_agents:
        address = raw_agent.get("address")
        if not isinstance(address, str) or not address.strip():
            continue

        enriched_agent = dict(raw_agent)
        if not enriched_agent.get("protocols"):
            remaining_timeout = _remaining_timeout(deadline)
            if remaining_timeout <= 0:
                timed_out = True
                break

            try:
                agent_record = _fetch_agent_record(address, timeout=remaining_timeout)
            except requests.Timeout:
                timed_out = True
                break
            except Exception:
                continue

            record_protocols = agent_record.get("protocols")
            if isinstance(record_protocols, list):
                enriched_agent["protocols"] = [
                    {"name": "", "version": "", "digest": protocol_digest}
                    for protocol_digest in record_protocols
                    if isinstance(protocol_digest, str)
                ]

            domain_name = agent_record.get("domain_name")
            if isinstance(domain_name, str) and domain_name.strip():
                enriched_agent["domain"] = domain_name

        if not enriched_agent.get("protocols"):
            continue

        if _final_candidate_score(enriched_agent, query_variants) <= 0:
            continue

        verified_agents.append(enriched_agent)
        if len(verified_agents) >= limit:
            break

    return verified_agents, timed_out


def _schema_required_fields(schema: dict[str, Any]) -> set[str]:
    return {
        field
        for field in schema.get("required", [])
        if isinstance(field, str)
    }


def _schema_properties(schema: dict[str, Any]) -> set[str]:
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return set()
    return {field for field in properties if isinstance(field, str)}


def _matches_payload_shape(schema: dict[str, Any], payload_obj: dict[str, Any]) -> bool:
    payload_keys = set(payload_obj.keys())
    required_fields = _schema_required_fields(schema)
    if not required_fields.issubset(payload_keys):
        return False

    properties = _schema_properties(schema)
    if not properties:
        return not payload_keys

    if properties and not payload_keys.issubset(properties):
        return False

    return True


def _format_protocols(protocols: Any, max_protocols: int = 3) -> str:
    if not isinstance(protocols, list):
        return "none"

    formatted_protocols = []
    for protocol in protocols:
        if not isinstance(protocol, dict):
            continue

        name = str(protocol.get("name") or "<unnamed>")
        version = str(protocol.get("version") or "").strip()
        digest = protocol.get("digest")

        label = name
        if version:
            label += f" v{version}"
        if isinstance(digest, str) and digest.strip():
            label += f" ({_normalize_digest(digest, 'proto')})"

        formatted_protocols.append(label)

    if len(formatted_protocols) > max_protocols:
        remaining = len(formatted_protocols) - max_protocols
        formatted_protocols = formatted_protocols[:max_protocols] + [
            f"+{remaining} more"
        ]

    return ", ".join(formatted_protocols) or "none"


def _format_rating(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.1f}"
    return "n/a"


def _find_request_model(
    request_models: list[dict[str, Any]],
    payload_obj: dict[str, Any],
    selector: Optional[str],
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    if selector:
        selector = selector.strip()
        normalized_model_digest = _normalize_digest(selector, "model")
        normalized_protocol_digest = _normalize_digest(selector, "proto")
        lowered_selector = selector.casefold()

        model_matches = [
            request_model
            for request_model in request_models
            if request_model["request_digest"] == normalized_model_digest
        ]
        if len(model_matches) == 1:
            return model_matches[0], None
        if len(model_matches) > 1:
            return None, f"selector matched multiple request digests: {selector}"

        protocol_matches = [
            request_model
            for request_model in request_models
            if request_model["protocol_digest"] == normalized_protocol_digest
        ]
        if len(protocol_matches) == 1:
            return protocol_matches[0], None
        if len(protocol_matches) > 1:
            return None, (
                f"selector {selector} matched protocol "
                f"{protocol_matches[0]['protocol_name']} with multiple request models"
            )

        name_matches = [
            request_model
            for request_model in request_models
            if str(request_model["protocol_name"]).casefold() == lowered_selector
        ]
        if len(name_matches) == 1:
            return name_matches[0], None
        if len(name_matches) > 1:
            return None, f"selector matched multiple protocols named: {selector}"

        return None, f"selector did not match any request model or protocol: {selector}"

    compatible_matches = [
        request_model
        for request_model in request_models
        if _matches_payload_shape(request_model["request_schema"], payload_obj)
    ]
    if len(compatible_matches) == 1:
        return compatible_matches[0], None
    if len(compatible_matches) > 1:
        return None, "multiple request models match the payload shape"
    if len(request_models) == 1:
        return request_models[0], None
    return None, "could not infer which request model to use"


def _format_request_models(address: str, request_models: list[dict[str, Any]]) -> str:
    if not request_models:
        return f"address: {address}\nrequest-models: none published"

    lines = [f"address: {address}", "request-models:"]
    for request_model in request_models:
        lines.append(
            (
                f"- protocol: {request_model['protocol_name'] or '<unnamed>'} "
                f"({request_model['protocol_digest']})"
            )
        )
        lines.append(f"  request-digest: {request_model['request_digest']}")
        lines.append(f"  interaction-type: {request_model['interaction_type']}")
        lines.append(
            f"  response-digests: {', '.join(request_model['response_digests']) or 'none'}"
        )
        lines.append(
            f"  schema: {_json_dumps(request_model['request_schema'], indent=2)}"
        )
    return "\n".join(lines)


async def _ask_agent_raw(
    destination: str,
    payload_obj: dict[str, Any],
    request_digest: str,
    timeout: int = 60,
) -> Any:
    return await send_message_raw(
        destination=destination,
        message_schema_digest=request_digest,
        message_body=_json_dumps(payload_obj),
        timeout=timeout,
        sync=True,
    )


async def _ask_agent(destination: str, request: Model, timeout: int = 60) -> str:
    envelope_or_status = await send_sync_message(
        destination=destination,
        message=request,
        timeout=timeout,
    )
    return str(envelope_or_status)


def technical_analysis(ticker: str, timeout: int = 60) -> str:
    try:
        request = TechAnalysisRequest(ticker=ticker)
        return asyncio.run(
            _ask_agent(TECHNICAL_ANALYSIS_AGENT_ADDRESS, request, int(timeout))
        )
    except Exception as e:
        return f"error: {e}"


def tavily_search(search_query: str, timeout: int = 60) -> str:
    try:
        request = WebSearchRequest(query=search_query)
        response = asyncio.run(
            _ask_agent(TAVILY_SEARCH_AGENT_ADDRESS, request, int(timeout))
        )
        return _format_tavily_results(response)
    except Exception as e:
        return f"error: {e}"


def agent_search(
    search_text: str,
    limit: Any = DEFAULT_AGENT_SEARCH_LIMIT,
    timeout: int = 15,
) -> str:
    try:
        normalized_search_text = " ".join(search_text.split())
        search_limit = _coerce_search_limit(limit)
        timeout_seconds = _coerce_timeout_seconds(timeout, default=15.0)
        deadline = time.monotonic() + timeout_seconds
        query_variants = _search_query_variants(normalized_search_text)
        merged_candidates: dict[str, dict[str, Any]] = {}
        timed_out = False

        search_modes = ["relevancy", "interactions", "last-modified"]
        per_query_limit = min(
            max(search_limit * 2, DEFAULT_AGENT_SEARCH_PER_QUERY),
            MAX_AGENT_SEARCH_SCAN,
        )

        for query_variant in query_variants:
            for sort in search_modes:
                remaining_timeout = _remaining_timeout(deadline)
                if remaining_timeout <= 0:
                    timed_out = True
                    break

                try:
                    raw_agents = _search_agents(
                        query_variant,
                        limit=per_query_limit,
                        timeout=remaining_timeout,
                        sort=sort,
                    )
                except requests.Timeout:
                    timed_out = True
                    break
                _merge_search_results(
                    merged_candidates,
                    raw_agents,
                    query_variants,
                    sort,
                )
            if timed_out:
                break

        ranked_candidates = sorted(
            merged_candidates.values(),
            key=lambda candidate: _candidate_pre_score(candidate, query_variants),
            reverse=True,
        )
        ranked_candidates = [
            candidate
            for candidate in ranked_candidates
            if _candidate_pre_score(candidate, query_variants) > 0
        ]

        if not ranked_candidates:
            lines = [f"query: {normalized_search_text}", "matches: none"]
            if timed_out:
                lines.append(
                    f"note: search reached the {timeout_seconds:g}s timeout budget"
                )
            return "\n".join(lines)

        agents, verify_timed_out = _verify_search_results(
            ranked_candidates[:MAX_AGENT_SEARCH_SCAN],
            limit=search_limit * 2,
            query_variants=query_variants,
            deadline=deadline,
        )
        timed_out = timed_out or verify_timed_out
        agents = sorted(
            agents,
            key=lambda candidate: _final_candidate_score(candidate, query_variants),
            reverse=True,
        )[:search_limit]
        if not agents:
            timeout_note = (
                f"note: search reached the {timeout_seconds:g}s timeout budget\n"
                if timed_out
                else ""
            )
            return (
                f"query: {normalized_search_text}\n"
                "matches: none callable\n"
                f"raw-hits: {len(ranked_candidates)}\n"
                f"{timeout_note}"
                "hint: try a more specific query using a product, protocol, or agent name"
            )

        lines = [f"query: {normalized_search_text}"]
        if timed_out:
            lines.append(
                f"note: partial results after reaching the {timeout_seconds:g}s timeout budget"
            )
        lines.append("matches:")
        for index, agent in enumerate(agents, start=1):
            name = str(agent.get("name") or "<unnamed>")
            address = str(agent.get("address") or "unknown")
            summary_source = agent.get("readme") or ""
            if str(summary_source).strip().casefold() in {"", "none"}:
                summary_source = agent.get("description") or ""
            summary = _truncate_text(summary_source, 320) or "none"
            status = str(agent.get("status") or "unknown")
            agent_type = str(agent.get("type") or "unknown")
            category = str(agent.get("category") or "unknown")
            recent_interactions = agent.get("recent_interactions", "unknown")
            total_interactions = agent.get("total_interactions", "unknown")
            rating = _format_rating(agent.get("rating"))
            protocols = _format_protocols(agent.get("protocols"))
            domain = str(agent.get("domain") or "").strip()

            lines.append(f"{index}. name: {name}")
            lines.append(f"   address: {address}")
            if domain:
                lines.append(f"   domain: {domain}")
            lines.append(f"   summary: {summary}")
            lines.append(f"   protocols: {protocols}")
            lines.append(
                f"   status/type/category: {status} / {agent_type} / {category}"
            )
            lines.append(
                "   interactions: "
                f"recent90d={recent_interactions} total={total_interactions} rating={rating}"
            )
        return "\n".join(lines)
    except requests.HTTPError as e:
        status_code = getattr(e.response, "status_code", "unknown")
        return f"error: agentverse http {status_code}: {e}"
    except Exception as e:
        return f"error: {e}"


def agent_input_models(address: str, timeout: int = 15) -> str:
    try:
        request_models = _collect_request_models(address, int(timeout))
        return _format_request_models(address, request_models)
    except requests.HTTPError as e:
        status_code = getattr(e.response, "status_code", "unknown")
        return f"error: agentverse http {status_code}: {e}"
    except Exception as e:
        return f"error: {e}"


def call_agent(
    address: str,
    payload_json: str,
    selector: Optional[str] = None,
    timeout: int = 60,
) -> str:
    try:
        payload_obj = json.loads(payload_json)
    except json.JSONDecodeError as e:
        return f"error: invalid JSON payload: {e}"

    if not isinstance(payload_obj, dict):
        return "error: payload must be a JSON object"

    try:
        request_models = _collect_request_models(address, timeout=15)
        if not request_models:
            return f"error: no published request models found for {address}"

        request_model, selection_error = _find_request_model(
            request_models,
            payload_obj,
            selector,
        )
        if request_model is None:
            details = _format_request_models(address, request_models)
            return f"error: {selection_error}\n{details}"

        response = asyncio.run(
            _ask_agent_raw(
                destination=address,
                payload_obj=payload_obj,
                request_digest=request_model["request_digest"],
                timeout=int(timeout),
            )
        )
        return _format_json_text(str(response))
    except requests.HTTPError as e:
        status_code = getattr(e.response, "status_code", "unknown")
        return f"error: agentverse http {status_code}: {e}"
    except Exception as e:
        return f"error: {e}"
