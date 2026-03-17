from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from .models import ModelResult, ModelStatus

logger = logging.getLogger(__name__)

try:
  from mcp import ClientSession, types as mcp_types
  from mcp.client.streamablehttp import streamablehttp_client

  HAS_MCP = True
except ImportError:
  HAS_MCP = False

DATABASE_TITLE = "HF Litmus Results"

# Database schema: property name -> type
DATABASE_PROPERTIES = {
  "Model ID": "title",
  "Status": "rich_text",
  "Pipeline Tag": "rich_text",
  "Downloads": "number",
  "Likes": "number",
  "Failure Stage": "rich_text",
  "Failure Class": "rich_text",
  "Missing Ops": "rich_text",
  "Ingest Version": "rich_text",
  "Last Tested": "rich_text",
}


@dataclass
class NotionPage:
  """Reference to a created/updated Notion page."""
  page_id: str
  url: str = ""


def _run_async(coro: Any) -> Any:
  """Run an async coroutine from sync code."""
  try:
    asyncio.get_running_loop()
  except RuntimeError:
    return asyncio.run(coro)
  # Already in an event loop — run in a thread
  import concurrent.futures
  with concurrent.futures.ThreadPoolExecutor(
    max_workers=1
  ) as pool:
    return pool.submit(asyncio.run, coro).result(
      timeout=120
    )


class NotionPublisher:
  """Publishes litmus results to Notion via MCP.

  Manages a database under the parent page where each
  row is a model. On re-analysis, updates the existing
  entry rather than creating a duplicate.
  """

  def __init__(
    self,
    mcp_url: str,
    parent_page_id: str,
    database_id: str = "",
  ) -> None:
    if not HAS_MCP:
      raise ImportError(
        "mcp package required for Notion publishing."
        " Install with: pip install 'hf-litmus[notion]'"
      )
    self.mcp_url = mcp_url
    self.parent_page_id = parent_page_id
    self.database_id = database_id

  def publish_result(
    self,
    result: ModelResult,
    analysis_md: Optional[str] = None,
    gap_data: Optional[dict] = None,
  ) -> Optional[NotionPage]:
    """Publish a model result to Notion. Best-effort."""
    try:
      return _run_async(
        self._publish_result_async(
          result, analysis_md, gap_data,
        )
      )
    except Exception:
      logger.warning(
        "Notion publish failed for %s",
        result.model_id,
        exc_info=True,
      )
      return None

  async def _publish_result_async(
    self,
    result: ModelResult,
    analysis_md: Optional[str],
    gap_data: Optional[dict],
  ) -> Optional[NotionPage]:
    """Create or update a Notion database entry."""
    async with streamablehttp_client(
      self.mcp_url
    ) as (read_stream, write_stream, _):
      async with ClientSession(
        read_stream, write_stream
      ) as session:
        await session.initialize()

        tools = await session.list_tools()
        tool_names = {
          t.name for t in tools.tools
        }
        logger.debug(
          "Notion MCP tools: %s", tool_names
        )

        # Step 1: Find or create the database
        db_id = await self._find_or_create_database(
          session, tool_names,
        )
        if not db_id:
          # Fallback: create a standalone page
          return await self._create_standalone_page(
            session, tool_names, result,
            analysis_md, gap_data,
          )

        # Step 2: Find existing entry or create new
        existing_id = None
        if result.notion_page_id:
          # Validate cached page ID
          existing_id = await self._validate_page(
            session, tool_names,
            result.notion_page_id,
          )

        if not existing_id:
          existing_id = await self._find_entry(
            session, tool_names, db_id,
            result.model_id,
          )

        # Step 3: Upsert
        if existing_id:
          page = await self._update_entry(
            session, tool_names,
            existing_id, result,
            analysis_md, gap_data,
          )
        else:
          page = await self._create_entry(
            session, tool_names, db_id,
            result, analysis_md, gap_data,
          )

        return page

  # -- Database management --

  async def _find_or_create_database(
    self,
    session: ClientSession,
    tool_names: set[str],
  ) -> str:
    """Find or create the litmus results database."""
    # Try cached ID first
    if self.database_id:
      valid = await self._validate_database(
        session, tool_names, self.database_id,
      )
      if valid:
        return self.database_id
      logger.warning(
        "Cached database ID %s invalid,"
        " searching for database",
        self.database_id,
      )
      self.database_id = ""

    # Search for existing database
    db_id = await self._search_database(
      session, tool_names,
    )
    if db_id:
      self.database_id = db_id
      return db_id

    # Create new database
    db_id = await self._create_database(
      session, tool_names,
    )
    if db_id:
      self.database_id = db_id
    return db_id

  async def _validate_database(
    self,
    session: ClientSession,
    tool_names: set[str],
    db_id: str,
  ) -> bool:
    """Check if a database ID is still valid."""
    fetch_tool = _find_tool(tool_names, [
      "notion-fetch",
      "fetch",
    ])
    if not fetch_tool:
      # Can't validate, assume valid
      return True

    try:
      resp = await session.call_tool(
        fetch_tool, {"pageId": db_id},
      )
      data = _parse_response(resp)
      return bool(
        data and data.get("object") == "database"
      )
    except Exception:
      return False

  async def _search_database(
    self,
    session: ClientSession,
    tool_names: set[str],
  ) -> str:
    """Search for our database under the parent."""
    search_tool = _find_tool(tool_names, [
      "notion-search",
      "search",
    ])
    if not search_tool:
      return ""

    try:
      resp = await session.call_tool(
        search_tool,
        {"query": DATABASE_TITLE},
      )
      data = _parse_response(resp)
      if not data:
        return ""

      # Response may be a list or have results key
      results = data
      if isinstance(data, dict):
        results = data.get("results", [])
      if not isinstance(results, list):
        return ""

      for item in results:
        if not isinstance(item, dict):
          continue
        if item.get("object") != "database":
          continue
        # Check parent matches
        parent = item.get("parent", {})
        parent_id = (
          parent.get("page_id", "")
          .replace("-", "")
        )
        our_id = self.parent_page_id.replace("-", "")
        if parent_id == our_id:
          db_id = item.get("id", "")
          if db_id:
            logger.info(
              "Found existing database: %s", db_id
            )
            return db_id
    except Exception:
      logger.debug(
        "Database search failed", exc_info=True,
      )
    return ""

  async def _create_database(
    self,
    session: ClientSession,
    tool_names: set[str],
  ) -> str:
    """Create the litmus results database."""
    create_tool = _find_tool(tool_names, [
      "notion-create-database",
      "create-database",
      "createDatabase",
    ])
    if not create_tool:
      logger.warning(
        "No database creation tool found."
        " Available: %s", tool_names,
      )
      return ""

    # Build property schema
    properties: dict[str, Any] = {}
    for name, ptype in DATABASE_PROPERTIES.items():
      if ptype == "title":
        properties[name] = {"title": {}}
      elif ptype == "number":
        properties[name] = {
          "number": {"format": "number"},
        }
      else:
        properties[name] = {"rich_text": {}}

    try:
      resp = await session.call_tool(
        create_tool,
        {
          "parent_page_id": self.parent_page_id,
          "title": DATABASE_TITLE,
          "properties": properties,
        },
      )
      db_id = _extract_page_id(resp)
      if db_id:
        logger.info(
          "Created database: %s", db_id,
        )
      return db_id
    except Exception:
      logger.warning(
        "Database creation failed", exc_info=True,
      )
      return ""

  # -- Entry management --

  async def _validate_page(
    self,
    session: ClientSession,
    tool_names: set[str],
    page_id: str,
  ) -> str:
    """Validate a cached page ID. Returns ID or ''."""
    fetch_tool = _find_tool(tool_names, [
      "notion-fetch",
      "fetch",
    ])
    if not fetch_tool:
      return page_id  # Can't validate, assume valid

    try:
      resp = await session.call_tool(
        fetch_tool, {"pageId": page_id},
      )
      data = _parse_response(resp)
      if data and data.get("object") == "page":
        return page_id
    except Exception:
      pass
    return ""

  async def _find_entry(
    self,
    session: ClientSession,
    tool_names: set[str],
    db_id: str,
    model_id: str,
  ) -> str:
    """Find an existing entry by model ID."""
    # Try notion-query-database-view first
    query_tool = _find_tool(tool_names, [
      "notion-query-database-view",
      "notion-query-data-sources",
      "query-database",
      "queryDatabase",
    ])
    if not query_tool:
      return ""

    try:
      resp = await session.call_tool(
        query_tool,
        {
          "databaseId": db_id,
          "filter": {
            "property": "Model ID",
            "title": {"equals": model_id},
          },
        },
      )
      data = _parse_response(resp)
      if not data:
        return ""

      results = data
      if isinstance(data, dict):
        results = data.get("results", [])
      if not isinstance(results, list):
        return ""

      for item in results:
        if isinstance(item, dict):
          pid = item.get("id", "")
          if pid:
            logger.debug(
              "Found existing entry for %s: %s",
              model_id, pid,
            )
            return pid
    except Exception:
      logger.debug(
        "Database query failed for %s",
        model_id, exc_info=True,
      )
    return ""

  async def _create_entry(
    self,
    session: ClientSession,
    tool_names: set[str],
    db_id: str,
    result: ModelResult,
    analysis_md: Optional[str],
    gap_data: Optional[dict],
  ) -> Optional[NotionPage]:
    """Create a new database entry for a model."""
    create_tool = _find_tool(tool_names, [
      "notion-create-pages",
      "notion-create-a-page",
      "create-a-page",
      "createPage",
    ])
    if not create_tool:
      logger.error(
        "No page creation tool found."
        " Available: %s", tool_names,
      )
      return None

    props = self._build_properties(result)

    try:
      resp = await session.call_tool(
        create_tool,
        {
          "parent": {
            "database_id": db_id,
          },
          "properties": props,
        },
      )
    except Exception:
      # Retry with alternative param format
      logger.debug(
        "Retrying entry creation with"
        " parent_database_id param"
      )
      try:
        resp = await session.call_tool(
          create_tool,
          {
            "parent_database_id": db_id,
            "properties": props,
          },
        )
      except Exception:
        logger.warning(
          "Entry creation failed for %s",
          result.model_id, exc_info=True,
        )
        return None

    page_id = _extract_page_id(resp)
    if not page_id:
      logger.error(
        "Could not extract page ID"
        " from create response"
      )
      return None

    url = _extract_url(resp)

    # Append content blocks
    blocks = self._build_blocks(
      result, analysis_md, gap_data,
    )
    if blocks:
      await self._append_blocks(
        session, tool_names, page_id, blocks,
      )

    logger.info(
      "Created entry for %s: %s",
      result.model_id, url or page_id,
    )
    return NotionPage(page_id=page_id, url=url)

  async def _update_entry(
    self,
    session: ClientSession,
    tool_names: set[str],
    page_id: str,
    result: ModelResult,
    analysis_md: Optional[str],
    gap_data: Optional[dict],
  ) -> Optional[NotionPage]:
    """Update an existing database entry."""
    update_tool = _find_tool(tool_names, [
      "notion-update-page",
      "update-page",
      "updatePage",
    ])

    url = ""
    if update_tool:
      props = self._build_properties(result)
      try:
        resp = await session.call_tool(
          update_tool,
          {
            "pageId": page_id,
            "properties": props,
          },
        )
        url = _extract_url(resp)
      except Exception:
        logger.warning(
          "Property update failed for %s,"
          " appending content only",
          result.model_id, exc_info=True,
        )
    else:
      logger.debug(
        "No update tool, appending content only"
      )

    # Append new run section with divider
    now = datetime.now(timezone.utc)
    divider_blocks = [
      _divider(),
      _heading2(
        f"Run: {now:%Y-%m-%d %H:%M UTC}"
      ),
    ]
    content_blocks = self._build_blocks(
      result, analysis_md, gap_data,
    )
    all_blocks = divider_blocks + content_blocks

    if all_blocks:
      await self._append_blocks(
        session, tool_names, page_id, all_blocks,
      )

    logger.info(
      "Updated entry for %s: %s",
      result.model_id, url or page_id,
    )
    return NotionPage(page_id=page_id, url=url)

  # -- Standalone page fallback --

  async def _create_standalone_page(
    self,
    session: ClientSession,
    tool_names: set[str],
    result: ModelResult,
    analysis_md: Optional[str],
    gap_data: Optional[dict],
  ) -> Optional[NotionPage]:
    """Fallback: create a standalone page under parent."""
    create_tool = _find_tool(tool_names, [
      "notion-create-a-page",
      "create-a-page",
      "createPage",
    ])
    if not create_tool:
      logger.error(
        "No page creation tool found."
        " Available: %s", tool_names,
      )
      return None

    status = result.status.value
    title = f"HF Litmus: {result.model_id} — {status}"

    try:
      resp = await session.call_tool(
        create_tool,
        {
          "parent_page_id": self.parent_page_id,
          "title": title,
        },
      )
    except Exception:
      logger.warning(
        "Standalone page creation failed for %s",
        result.model_id, exc_info=True,
      )
      return None

    page_id = _extract_page_id(resp)
    if not page_id:
      return None

    url = _extract_url(resp)
    blocks = self._build_blocks(
      result, analysis_md, gap_data,
    )
    if blocks:
      await self._append_blocks(
        session, tool_names, page_id, blocks,
      )

    logger.info(
      "Created standalone page for %s: %s",
      result.model_id, url or page_id,
    )
    return NotionPage(page_id=page_id, url=url)

  # -- Property & block builders --

  def _build_properties(
    self, result: ModelResult,
  ) -> dict[str, Any]:
    """Build Notion database properties for a result."""
    status = result.status.value
    stage = (
      result.failure_stage.value
      if result.failure_stage else ""
    )
    fclass = (
      result.failure_class.value
      if result.failure_class else ""
    )
    ops = ", ".join(result.missing_ops[:20])
    tested = f"{result.tested_at:%Y-%m-%d %H:%M UTC}"

    return {
      "Model ID": {
        "title": [
          {
            "type": "text",
            "text": {"content": result.model_id},
          }
        ],
      },
      "Status": _rich_text_prop(status),
      "Pipeline Tag": _rich_text_prop(
        result.pipeline_tag
      ),
      "Downloads": {"number": result.downloads},
      "Likes": {"number": result.likes},
      "Failure Stage": _rich_text_prop(stage),
      "Failure Class": _rich_text_prop(fclass),
      "Missing Ops": _rich_text_prop(ops[:2000]),
      "Ingest Version": _rich_text_prop(
        result.ingest_version
      ),
      "Last Tested": _rich_text_prop(tested),
    }

  def _build_blocks(
    self,
    result: ModelResult,
    analysis_md: Optional[str],
    gap_data: Optional[dict],
  ) -> list[dict]:
    """Build Notion block content for the page."""
    blocks: list[dict] = []

    # Model info section
    blocks.append(_heading2("Model Information"))
    blocks.append(_paragraph(
      f"**Model:** {result.model_id}"
    ))
    blocks.append(_paragraph(
      f"**Pipeline Tag:** {result.pipeline_tag}"
    ))
    blocks.append(_paragraph(
      f"**Downloads:** {result.downloads:,}"
    ))
    blocks.append(_paragraph(
      f"**Likes:** {result.likes:,}"
    ))
    blocks.append(_paragraph(
      f"**Tested:** "
      f"{result.tested_at:%Y-%m-%d %H:%M UTC}"
    ))
    blocks.append(_paragraph(
      f"**Ingest Version:** {result.ingest_version}"
    ))

    # Result section
    blocks.append(_heading2("Result"))
    if result.status == ModelStatus.SUCCESS:
      blocks.append(_paragraph(
        "Tron ingest successfully generated"
        " a C++ plugin for this model."
      ))
    else:
      stage = (
        result.failure_stage.value
        if result.failure_stage else "unknown"
      )
      fclass = (
        result.failure_class.value
        if result.failure_class else "unknown"
      )
      blocks.append(_paragraph(
        f"**Status:** {result.status.value}"
      ))
      blocks.append(_paragraph(
        f"**Failure Stage:** {stage}"
      ))
      blocks.append(_paragraph(
        f"**Failure Class:** {fclass}"
      ))

    # Missing ops
    if result.missing_ops:
      blocks.append(_heading2("Missing Operations"))
      for op in result.missing_ops:
        blocks.append(_bulleted(f"`{op}`"))

    # Gap data sections
    if gap_data:
      furthest = gap_data.get(
        "furthest_stage", ""
      )
      if furthest:
        blocks.append(_heading2("Pipeline Progress"))
        blocks.append(_paragraph(
          f"**Furthest Stage:** {furthest}"
        ))

      fixes = gap_data.get("fixes_applied", [])
      if fixes:
        blocks.append(_heading2("Fixes Applied"))
        for fix in fixes:
          blocks.append(_bulleted(fix))

      kernels = gap_data.get(
        "missing_kernels", []
      )
      if kernels:
        blocks.append(_heading2("Missing Kernels"))
        for k in kernels:
          blocks.append(_bulleted(f"`{k}`"))

      patterns = gap_data.get(
        "missing_patterns", []
      )
      if patterns:
        blocks.append(
          _heading2("Missing Patterns")
        )
        for p in patterns:
          blocks.append(_bulleted(p))

      blockers = gap_data.get("blockers", [])
      if blockers:
        blocks.append(_heading2("Blockers"))
        for b in blockers:
          sev = b.get("severity", "?")
          desc = b.get("description", "")
          bstage = b.get("stage", "")
          effort = b.get("effort", "")
          blocks.append(_bulleted(
            f"[{sev}] {bstage}: {desc}"
            f" (effort: {effort})"
          ))

      # Include raw JSON
      blocks.append(
        _heading2("Gap Summary (JSON)")
      )
      blocks.append(_code_block(
        json.dumps(gap_data, indent=2),
        language="json",
      ))

    # Analysis markdown
    if analysis_md:
      blocks.append(
        _heading2("Detailed Analysis")
      )
      md_blocks = _markdown_to_blocks(
        analysis_md
      )
      blocks.extend(md_blocks)

    # Error output (truncated)
    if (
      result.error_output
      and result.status != ModelStatus.SUCCESS
    ):
      blocks.append(
        _heading2("Error Output")
      )
      error_text = result.error_output[:3000]
      blocks.append(
        _code_block(error_text, language="plain text")
      )

    return blocks

  async def _append_blocks(
    self,
    session: ClientSession,
    tool_names: set[str],
    page_id: str,
    blocks: list[dict],
  ) -> None:
    """Append blocks to a page in batches."""
    append_tool = _find_tool(tool_names, [
      "notion-append-block-children",
      "append-block-children",
      "appendBlockChildren",
    ])
    if not append_tool:
      logger.warning(
        "No block append tool found."
        " Available: %s", tool_names,
      )
      return

    # Notion limits to 100 blocks per request
    batch_size = 100
    for i in range(0, len(blocks), batch_size):
      batch = blocks[i:i + batch_size]
      try:
        await session.call_tool(
          append_tool,
          {
            "block_id": page_id,
            "children": batch,
          },
        )
      except Exception:
        logger.warning(
          "Failed to append blocks batch"
          " %d-%d for page %s",
          i, i + len(batch), page_id,
          exc_info=True,
        )
        # Try sending blocks as markdown text
        await self._append_as_markdown(
          session, tool_names,
          page_id, batch,
        )
        break

  async def _append_as_markdown(
    self,
    session: ClientSession,
    tool_names: set[str],
    page_id: str,
    blocks: list[dict],
  ) -> None:
    """Fallback: send content as simple paragraphs."""
    append_tool = _find_tool(tool_names, [
      "notion-append-block-children",
      "append-block-children",
    ])
    if not append_tool:
      return

    simple = []
    for b in blocks:
      btype = b.get("type", "paragraph")
      text = ""
      if btype in b:
        rt = b[btype].get("rich_text", [])
        if rt:
          text = rt[0].get("text", {}).get(
            "content", ""
          )
      if text:
        simple.append(_paragraph(text))

    if simple:
      try:
        await session.call_tool(
          append_tool,
          {
            "block_id": page_id,
            "children": simple[:100],
          },
        )
      except Exception:
        logger.warning(
          "Markdown fallback also failed"
          " for page %s", page_id,
          exc_info=True,
        )


# -- Helpers --

def _find_tool(
  available: set[str], candidates: list[str],
) -> str:
  """Find first matching tool name."""
  for name in candidates:
    if name in available:
      return name
  return ""


def _rich_text_prop(text: str) -> dict:
  """Build a rich_text property value."""
  text = text[:2000]
  return {
    "rich_text": [
      {
        "type": "text",
        "text": {"content": text},
      }
    ],
  }


# Block construction helpers

def _divider() -> dict:
  return {"type": "divider", "divider": {}}


def _heading2(text: str) -> dict:
  return {
    "type": "heading_2",
    "heading_2": {
      "rich_text": [
        {"type": "text", "text": {"content": text}}
      ],
    },
  }


def _heading3(text: str) -> dict:
  return {
    "type": "heading_3",
    "heading_3": {
      "rich_text": [
        {"type": "text", "text": {"content": text}}
      ],
    },
  }


def _paragraph(text: str) -> dict:
  text = text[:2000]
  return {
    "type": "paragraph",
    "paragraph": {
      "rich_text": [
        {"type": "text", "text": {"content": text}}
      ],
    },
  }


def _bulleted(text: str) -> dict:
  text = text[:2000]
  return {
    "type": "bulleted_list_item",
    "bulleted_list_item": {
      "rich_text": [
        {"type": "text", "text": {"content": text}}
      ],
    },
  }


def _code_block(
  text: str, language: str = "plain text"
) -> dict:
  text = text[:2000]
  return {
    "type": "code",
    "code": {
      "rich_text": [
        {"type": "text", "text": {"content": text}}
      ],
      "language": language,
    },
  }


def _markdown_to_blocks(md: str) -> list[dict]:
  """Convert markdown text to Notion blocks."""
  blocks: list[dict] = []
  lines = md.split("\n")
  in_code = False
  code_buf: list[str] = []
  code_lang = "plain text"

  for line in lines:
    if line.startswith("```"):
      if in_code:
        blocks.append(_code_block(
          "\n".join(code_buf), code_lang,
        ))
        code_buf = []
        in_code = False
      else:
        lang = line[3:].strip()
        code_lang = lang if lang else "plain text"
        in_code = True
      continue

    if in_code:
      code_buf.append(line)
      continue

    stripped = line.strip()
    if not stripped:
      continue

    if stripped.startswith("## "):
      blocks.append(
        _heading2(stripped[3:].strip())
      )
    elif stripped.startswith("### "):
      blocks.append(
        _heading3(stripped[4:].strip())
      )
    elif stripped.startswith("# "):
      blocks.append(
        _heading2(stripped[2:].strip())
      )
    elif (
      stripped.startswith("- ")
      or stripped.startswith("* ")
    ):
      blocks.append(
        _bulleted(stripped[2:].strip())
      )
    elif stripped.startswith("|"):
      continue
    else:
      blocks.append(_paragraph(stripped))

  if in_code and code_buf:
    blocks.append(_code_block(
      "\n".join(code_buf), code_lang,
    ))

  return blocks


def _parse_response(resp: Any) -> Any:
  """Parse JSON from MCP tool response."""
  if not resp or not resp.content:
    return None
  for item in resp.content:
    if hasattr(item, "text"):
      try:
        return json.loads(item.text)
      except (json.JSONDecodeError, TypeError):
        pass
  return None


def _extract_page_id(resp: Any) -> str:
  """Extract page ID from MCP tool response."""
  if not resp or not resp.content:
    return ""
  for item in resp.content:
    if hasattr(item, "text"):
      text = item.text
      try:
        data = json.loads(text)
        if isinstance(data, dict):
          pid = data.get(
            "id", data.get("page_id", "")
          )
          if pid:
            return str(pid)
      except (json.JSONDecodeError, TypeError):
        pass
      match = re.search(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}"
        r"-[0-9a-f]{4}-[0-9a-f]{12}",
        text,
      )
      if match:
        return match.group(0)
      match = re.search(r"[0-9a-f]{32}", text)
      if match:
        return match.group(0)
  return ""


def _extract_url(resp: Any) -> str:
  """Extract page URL from MCP tool response."""
  if not resp or not resp.content:
    return ""
  for item in resp.content:
    if hasattr(item, "text"):
      try:
        data = json.loads(item.text)
        if isinstance(data, dict):
          return str(data.get("url", ""))
      except (json.JSONDecodeError, TypeError):
        pass
  return ""
