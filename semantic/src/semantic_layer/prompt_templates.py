"""Jinja2 prompt templates for LLM interactions."""

from __future__ import annotations

from jinja2 import Template


class PromptTemplates:
    """Manages Jinja2 prompt templates for the prescriptive LLM layer."""

    _SYSTEM_PROMPT = """You are an expert industrial condition monitoring assistant \
for the Semantic Acoustic Digital Twin system. Your role is to provide prescriptive \
maintenance recommendations based on acoustic anomaly detections, spatial analysis, \
and technical documentation.

Guidelines:
- Be concise and actionable
- Reference specific maintenance procedures from retrieved documentation
- Include severity assessment (INFO, WARNING, CRITICAL, EMERGENCY)
- Suggest concrete next steps with estimated timeframes
- Consider the spatial context (which BIM asset was identified)
- Explain the acoustic signature and what it typically indicates
- Always cite sources when referencing documentation
- If uncertainty is high, recommend further investigation"""

    _PRESCRIPTIVE_TEMPLATE = Template("""## Acoustic Event Analysis

**Device:** {{ device_id }}
**Detected Event:** {{ class_name }} (confidence: {{ "%.1f"|format(confidence * 100) }}%)
**DOA Vector:** [{{ doa_vector | join(', ') }}]

{% if spatial_hits %}
### Spatial Context
The sound appears to originate from or near:
{% for hit in spatial_hits %}
- **{{ hit.asset_name }}** ({{ hit.ifc_type }}) — distance: {{ "%.1f"|format(hit.distance) }}m
{% endfor %}
{% endif %}

### Retrieved Documentation
{% for chunk in retrieved_chunks %}
---
**Source:** {{ chunk.source }}{% if chunk.page %} (Page {{ chunk.page }}){% endif %}

{{ chunk.content }}
{% endfor %}

{% if additional_context %}
### Additional Context
{{ additional_context }}
{% endif %}

---

Based on the above acoustic detection and context, provide:
1. **Assessment:** What does this sound event likely indicate?
2. **Severity:** Rate the severity (INFO/WARNING/CRITICAL/EMERGENCY)
3. **Recommended Actions:** Numbered list of concrete steps
4. **Timeline:** Urgency and recommended response timeframe
5. **References:** Cite relevant documentation used""")

    _SEARCH_QUERY_TEMPLATE = Template(
        """{{ class_name }} acoustic anomaly maintenance procedure """
        """{% if spatial_hits %}near {{ spatial_hits[0].asset_name }} """
        """{{ spatial_hits[0].ifc_type }}{% endif %}"""
    )

    def system_prompt(self) -> str:
        """Return the system prompt for the LLM."""
        return self._SYSTEM_PROMPT

    def build_prescriptive_prompt(
        self,
        device_id: str,
        class_name: str,
        confidence: float,
        doa_vector: list[float],
        spatial_hits: list[dict],
        retrieved_chunks: list[dict],
        additional_context: str | None = None,
    ) -> str:
        """Render the prescriptive analysis prompt."""
        return self._PRESCRIPTIVE_TEMPLATE.render(
            device_id=device_id,
            class_name=class_name,
            confidence=confidence,
            doa_vector=doa_vector,
            spatial_hits=spatial_hits,
            retrieved_chunks=retrieved_chunks,
            additional_context=additional_context,
        )

    def build_search_query(
        self,
        class_name: str,
        confidence: float,
        spatial_hits: list[dict] | None = None,
    ) -> str:
        """Build a search query string for RAG retrieval."""
        return self._SEARCH_QUERY_TEMPLATE.render(
            class_name=class_name,
            confidence=confidence,
            spatial_hits=spatial_hits or [],
        )
