import asyncio
import json
import os
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
