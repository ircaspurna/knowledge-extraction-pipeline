# Knowledge Extraction Pipeline - Complete Workflow Guide

**Version:** 3.1 (100% MCP-Native)
**Last Updated:** 2026-01-12
**Status:** ✅ Production Ready

---

## ⚠️ CRITICAL: Preventing Data Loss

**DO NOT bypass the MCP workflow or manually edit JSON files!**

Skipping the `parse_extraction_responses` tool breaks metadata flow and causes **0 relationships** in your knowledge graph.

### The Problem

When metadata (especially `chunk_id`) is lost during extraction:
- ✅ Entities extracted: Yes (203 concepts)
- ❌ Relationships found: **ZERO** (graph is disconnected)

### The Solution

**ALWAYS use `parse_extraction_responses` after processing extraction prompts.**

This tool preserves the critical `chunk_id` metadata that links concepts to text chunks, enabling relationship extraction. See [Troubleshooting](#-troubleshooting) below for details.

---

## 🎯 Overview

This is a **complete, production-ready system** for extracting knowledge from PDFs and building professional knowledge graphs. Everything runs through MCP tools - **no manual Python scripts needed**.

**What you get:**
- Extract concepts from academic PDFs
- Build knowledge graphs automatically
- Visualize in browser or Neo4j
- Search across all documents
- 40-60% cost savings vs direct API calls

---

## 🚀 Quick Start (5 Minutes)

**Have a single PDF? Start here:**

```
1. Tell Claude Code:
   "Process /path/to/paper.pdf and build a knowledge graph"

2. Claude Code will automatically:
   - Extract text and create chunks
   - Generate and process extraction prompts
   - Deduplicate entities
   - Build the graph
   - Create visualization

3. Done! Open the HTML file or import to Neo4j
```

**That's it!** The pipeline is 100% conversational.

---

## 📚 Complete Workflows

### Workflow 1: Single PDF → Knowledge Graph

**Use case:** Process one research paper

**Steps:**
```
1. mcp__knowledge-extraction__process_pdf_document
   - Input: PDF path
   - Output: document.json with extracted text

2. mcp__knowledge-extraction__create_semantic_chunks
   - Input: document.json
   - Output: chunks.json (semantic chunks)

3. mcp__knowledge-extraction__generate_extraction_prompts
   - Input: chunks.json
   - Output: extraction_batch.json

4. [Claude Code processes extraction prompts automatically]

5. mcp__knowledge-extraction__parse_extraction_responses
   - Input: extraction_responses.json (from Claude Code)
   - Output: concepts.json

6. mcp__knowledge-extraction__resolve_entities_automatic
   - Input: concepts.json
   - Output: entities.json (deduplicated)

7. mcp__knowledge-extraction__build_knowledge_graph
   - Input: entities.json
   - Output: knowledge_graph.json + .graphml + HTML viz

8. mcp__knowledge-extraction__create_graph_visualization
   - Input: knowledge_graph.json
   - Output: Interactive HTML (open in browser)

   OR

   mcp__knowledge-extraction__import_graph_to_neo4j
   - Input: knowledge_graph.json
   - Output: Graph in Neo4j (http://localhost:7474)
```

**Time:** 1-2 hours (mostly LLM processing)
**Cost:** Free (Claude Code subscription)

---

### Workflow 2: Batch PDFs → Unified Graph

**Use case:** Process 10-100 research papers into one graph

**Steps:**
```
1. mcp__knowledge-extraction__batch_process_pdfs ⭐ NEW!
   - Input: Directory with PDFs
   - Output: One subdirectory per PDF with document.json, chunks.json, extraction_batch.json
   - Saves hours of manual work!

2. For each PDF directory (or combined):
   a. [Claude Code processes extraction_batch.json]
   b. mcp__knowledge-extraction__parse_extraction_responses
   c. Repeat for all papers

3. Combine all concepts.json files:
   - Manual step: Merge JSON arrays from all papers
   - Or: Use jq command to combine

4. mcp__knowledge-extraction__resolve_entities_automatic
   - Input: all_concepts.json (combined)
   - Output: entities.json (deduplicated across ALL papers)

5. mcp__knowledge-extraction__build_knowledge_graph
   - Input: entities.json
   - Output: Unified knowledge graph

6. mcp__knowledge-extraction__get_graph_statistics ⭐ NEW!
   - Check: Node count, categories, top concepts
   - Quality check before visualization

7. mcp__knowledge-extraction__import_graph_to_neo4j
   - Professional graph database
   - Advanced queries and analysis
```

**Time:** 4-8 hours for 50 papers (mostly automated)
**Cost:** Free (Claude Code)
**Result:** Unified knowledge graph spanning entire research domain

---

### Workflow 3: Search & Explore Existing Data

**Use case:** Query processed documents

**Steps:**
```
1. mcp__knowledge-extraction__search_semantic_documents ⭐ NEW!
   - Query: "cognitive load in deception detection"
   - Base directory: Where your processed PDFs are
   - Returns: Top matching passages across ALL documents

2. mcp__knowledge-extraction__get_graph_statistics ⭐ NEW!
   - Analyze graph structure
   - Find hub concepts
   - Check connectivity

3. Open Neo4j Browser (if imported):
   - http://localhost:7474
   - Run Cypher queries
   - Visual exploration
```

**Use when:**
- Finding information across documents
- Literature review
- Identifying research gaps
- Connecting concepts

---

### Workflow 4: Add New Papers to Existing Graph

**Use case:** Grow your knowledge graph over time

**Steps:**
```
1. Process new PDF (Workflow 1, steps 1-5)
   - Get concepts.json for new paper

2. Combine with existing concepts:
   - Load existing: all_concepts_previous.json
   - Append new concepts
   - Save as: all_concepts_updated.json

3. Re-resolve entities:
   - mcp__knowledge-extraction__resolve_entities_automatic
   - Input: all_concepts_updated.json
   - Output: entities_updated.json
   - New concepts merge with existing ones

4. Rebuild graph:
   - mcp__knowledge-extraction__build_knowledge_graph
   - Creates updated graph

5. Re-import to Neo4j:
   - mcp__knowledge-extraction__import_graph_to_neo4j
   - clear_existing=true
   - Graph now includes new paper!
```

**Time:** 1-2 hours per new paper
**Incremental:** Graph grows organically

---

## 🔧 Tool Reference

### Document Processing (Phase 1-3)

| Tool | Purpose | Input | Output |
|------|---------|-------|--------|
| `process_pdf_document` | Extract text from PDF | PDF path | document.json |
| `create_semantic_chunks` | Create semantic chunks | document.json | chunks.json |
| `batch_process_pdfs` ⭐ | Process multiple PDFs | PDF directory | Multiple doc/chunk files |

### Concept Extraction (Phase 4)

| Tool | Purpose | Input | Output |
|------|---------|-------|--------|
| `generate_extraction_prompts` | Create prompts for LLM | chunks.json | extraction_batch.json |
| `parse_extraction_responses` | Parse LLM responses | extraction_responses.json | concepts.json |

### Entity Resolution (Phase 5-6)

| Tool | Purpose | Input | Output |
|------|---------|-------|--------|
| `resolve_entities_automatic` | Deduplicate concepts | concepts.json | entities.json |

### Relationships (Phase 7-8)

| Tool | Purpose | Input | Output |
|------|---------|-------|--------|
| `create_relationship_batch` | Find co-occurring entities | entities.json + chunks.json | relationship_batch.json |
| `parse_relationship_responses` | Parse relationship types | relationship_responses.json | relationships.json |

### Graph Building (Phase 9-10)

| Tool | Purpose | Input | Output |
|------|---------|-------|--------|
| `build_knowledge_graph` | Build NetworkX graph | entities.json | knowledge_graph.json + .graphml |
| `create_graph_visualization` ⭐ | Interactive HTML viz | knowledge_graph.json | Interactive HTML |
| `import_graph_to_neo4j` | Import to Neo4j | knowledge_graph.json | Graph in Neo4j |

### Utilities

| Tool | Purpose | Input | Output |
|------|---------|-------|--------|
| `search_semantic_documents` ⭐ | Semantic search | Query + base directory | Ranked results |
| `get_graph_statistics` ⭐ | Analyze graph | knowledge_graph.json | Statistics report |
| `get_extraction_stats` | Extraction stats | concepts.json / entities.json | Stats report |

⭐ = New in version 3.0

---

## 💡 Pro Tips

### For Best Results

1. **Use batch processing for 10+ PDFs**
   - One command instead of processing individually
   - Automatic error recovery
   - Saves hours of work

2. **Check statistics before visualization**
   - Use `get_graph_statistics` first
   - Verify graph quality
   - Identify issues early

3. **Semantic search is powerful**
   - Use it for literature review
   - Find connections between papers
   - Locate supporting evidence

4. **Neo4j for large graphs**
   - HTML visualization: < 5,000 nodes
   - Neo4j: 5,000+ nodes
   - Better queries and analysis

### Cost Optimization

- **All LLM work through Claude Code:** Free with subscription
- **No direct API calls:** 40-60% savings if you were using API
- **Batch processing:** Process multiple documents efficiently

### Quality Control

1. **After extraction:** Check `get_extraction_stats`
   - Concepts per chunk should be 3-7
   - High confidence scores (>0.7)

2. **After resolution:** Check reduction percentage
   - Should reduce by 30-50%
   - Too low = concepts too similar
   - Too high = concepts too different

3. **After graph building:** Check `get_graph_statistics`
   - Connected components should be 1-3
   - Density should be 0.001-0.1
   - Top concepts make sense

---

## 🎓 Examples

### Example 1: Process 50 Deception Detection Papers

```
Step 1: Batch process
> mcp__knowledge-extraction__batch_process_pdfs
  input_dir: "/Users/IRI/Knowledge Base/PDF_INBOX/deception_papers"
  output_dir: "/Users/IRI/Knowledge Base/PIPELINE_OUTPUT/deception_batch"

Step 2: Extract concepts (automated)
> For each paper, Claude Code processes extraction_batch.json
> Save all responses

Step 3: Parse and combine
> For each paper: parse_extraction_responses
> Combine all concepts.json → all_concepts.json

Step 4: Build graph
> resolve_entities_automatic (all_concepts.json)
> build_knowledge_graph (entities.json)
> import_graph_to_neo4j (knowledge_graph.json)

Result: 1,537 entities, 1,158 relationships, ready for analysis!
```

### Example 2: Find Related Concepts

```
> mcp__knowledge-extraction__search_semantic_documents
  query: "linguistic markers of deception"
  base_dir: "/Users/IRI/Knowledge Base/PIPELINE_OUTPUT/deception_batch"
  top_k: 20
  min_similarity: 0.75

Returns: Top 20 passages about linguistic deception markers across all 50 papers
```

### Example 3: Analyze Graph Quality

```
> mcp__knowledge-extraction__get_graph_statistics
  graph_file: "/Users/IRI/Knowledge Base/PIPELINE_OUTPUT/deception_batch/knowledge_graph.json"

Output:
- Nodes: 1,537
- Edges: 1,158
- Density: 0.001
- Top concept: "Deception Detection" (891 occurrences)
- Most connected: "Linguistic cues" (46 connections)
- Categories: 8 (method, finding, concept, problem, etc.)
```

---

## 🔍 Troubleshooting

### Zero Relationships in Knowledge Graph ⚠️ CRITICAL

**Problem:** Graph shows nodes but 0 edges (e.g., "203 nodes, 0 edges")

**Cause:** Metadata flow broken - `chunk_id` fields are empty in entities

**Diagnosis:**
```python
import json
with open('entities.json') as f:
    entities = json.load(f).get('entities', [])

empty = sum(1 for e in entities for ev in e.get('evidence', [])
            if not ev.get('chunk_id'))

if empty > 0:
    print(f"⚠️ WORKFLOW ERROR: {empty} empty chunk_ids!")
```

**Solutions:**

1. **If you manually created concept files:**
   - ❌ Don't do this! It breaks metadata flow
   - ✅ Start over using MCP tools (see Quick Start)
   - Specifically: **MUST use `parse_extraction_responses`**

2. **If you skipped `parse_extraction_responses`:**
   - This tool is NOT optional - it injects `chunk_id` metadata
   - Re-run: `parse_extraction_responses(responses_file, output_file)`
   - Verify output has non-empty `chunk_id` fields

3. **Emergency repair (for already-broken data):**
   ```python
   # Use repair tool to match quotes back to chunks
   from Pipeline import chunk_id_repair
   chunk_id_repair.repair('output_directory/')
   ```

**Prevention:**
- ALWAYS use complete MCP workflow (never skip steps)
- NEVER manually edit JSON files
- Use `parse_extraction_responses` after processing prompts

See [Critical Warning](#️-critical-preventing-data-loss) above.

---

### MCP Server Not Found

**Problem:** Tools not available in Claude Code

**Solution:**
1. Check MCP server configured in `.claude.json`
2. Restart Claude Code
3. Verify server path points to `/Users/IRI/Knowledge Base/Pipeline/modules/mcp/mcp_server.py`

### Low Extraction Quality

**Problem:** Too few or too many concepts extracted

**Solution:**
1. Check `config/prompts.yaml` settings
2. Adjust chunk size (try 1200-1500 for dense papers)
3. Review sample extractions before processing all

### Graph Too Large to Visualize

**Problem:** HTML visualization slow or crashes

**Solution:**
1. Use Neo4j instead (`import_graph_to_neo4j`)
2. Filter to top N entities before building graph
3. Use `get_graph_statistics` to check size first

### Search Returns No Results

**Problem:** `search_semantic_documents` finds nothing

**Solution:**
1. Verify PDFs were processed with vector indexing
2. Check `chroma_db` directories exist
3. Try broader query terms
4. Lower `min_similarity` threshold

---

## 📞 Support

### Documentation

- **This file (WORKFLOW.md):** Complete workflows
- **MCP_TOOLS_REFERENCE.md:** Detailed tool documentation
- **START_HERE.md:** Quick orientation

### Common Questions

**Q: Do I need to know Python?**
A: No! Everything is through Claude Code conversational interface.

**Q: How much does it cost?**
A: Free with Claude Code subscription. No direct API costs.

**Q: Can I process non-English PDFs?**
A: Yes, but extraction quality depends on language. English works best.

**Q: How many PDFs can I process?**
A: Tested with 188 papers. Should handle 500+ papers.

**Q: What about relationships between concepts?**
A: Use `create_relationship_batch` after building graph. Adds typed relationships.

---

## ✅ Success Checklist

Before you start:
- [ ] Neo4j Desktop installed (optional, for visualization)
- [ ] Claude Code configured with MCP server
- [ ] PDFs in a directory
- [ ] Output directory prepared

After processing:
- [ ] Check extraction stats (3-7 concepts/chunk is good)
- [ ] Verify entity resolution (30-50% reduction is good)
- [ ] Review graph statistics (make sense?)
- [ ] Visualize and explore

---

## 🚀 Next Steps

1. **Start small:** Process 1-5 PDFs first
2. **Check quality:** Use statistics tools
3. **Scale up:** Batch process more PDFs
4. **Explore:** Use search and Neo4j
5. **Iterate:** Refine prompts if needed

---

**🎉 You now have a complete, professional knowledge extraction system!**

**Questions?** See MCP_TOOLS_REFERENCE.md for detailed tool documentation.

---

**Last Updated:** 2025-12-01
**Pipeline Version:** 3.0 (Complete)
**Tools:** 14 (100% MCP coverage)
