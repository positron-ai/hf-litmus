"""Prompt template loading for HF Litmus deep analysis.

Loads Markdown prompt templates with section markers and
renders them using string.Template ($-style substitution).
"""

from __future__ import annotations

import importlib.resources
import logging
import re
from string import Template

logger = logging.getLogger(__name__)

# Required sections that must be present in the template.
REQUIRED_SECTIONS = frozenset(
    {
        "main",
        "stage_trust_remote_code",
        "stage_unsupported_dynamic",
        "stage_export_default",
        "stage_ingest",
        "consensus_review",
    }
)

_SECTION_RE = re.compile(r"^<!--\s*SECTION:\s*(\w+)\s*-->$", re.MULTILINE)


def _parse_sections(text: str) -> dict[str, str]:
    """Split template text into named sections.

    Sections are delimited by HTML comments of the form:
      <!-- SECTION: name -->

    Content before the first section marker is discarded
    (it's the file-level header comment).
    """
    markers = list(_SECTION_RE.finditer(text))
    if not markers:
        raise ValueError(
            "No section markers found in prompt template."
            " Expected <!-- SECTION: name --> markers."
        )

    sections: dict[str, str] = {}
    for i, match in enumerate(markers):
        name = match.group(1)
        if name in sections:
            raise ValueError(f"Duplicate section '{name}' in prompt template.")
        start = match.end()
        end = markers[i + 1].start() if i + 1 < len(markers) else len(text)
        sections[name] = text[start:end].strip()

    return sections


def load_prompt_template(
    template_name: str = "deep_analysis_prompt.md",
) -> dict[str, str]:
    """Load and parse a prompt template from the prompts package.

    Returns a dict mapping section names to their content.
    Validates that all required sections are present.
    """
    ref = importlib.resources.files(__package__) / template_name
    text = ref.read_text(encoding="utf-8")
    sections = _parse_sections(text)

    missing = REQUIRED_SECTIONS - sections.keys()
    if missing:
        raise ValueError(
            f"Prompt template '{template_name}' is missing"
            f" required sections: {sorted(missing)}"
        )

    return sections


def render_template(
    template_text: str,
    variables: dict[str, str],
    strict: bool = True,
) -> str:
    """Render a template section with variable substitution.

    Uses string.Template for $-style substitution, which
    avoids conflicts with JSON {} braces in the template.

    Args:
      template_text: Template content with $variable placeholders.
      variables: Mapping of variable names to values.
      strict: If True, raise ValueError when unresolved
        $vars remain after substitution.

    Returns:
      Rendered text with variables substituted.

    Raises:
      ValueError: If strict=True and unresolved variables
        are found in the template.
    """
    tmpl = Template(template_text)
    result = tmpl.safe_substitute(variables)

    if strict:
        # Check for unresolved placeholders by inspecting the
        # template source (not the output) to avoid false positives
        # from $$ escapes and false negatives from ${braced} form.
        expected: set[str] = set()
        for m in Template.pattern.finditer(template_text):
            name = m.group("named") or m.group("braced")
            if name:
                expected.add(name)
        unresolved = sorted(expected - variables.keys())
        if unresolved:
            raise ValueError(f"Unresolved template variables: {unresolved}")

    return result
