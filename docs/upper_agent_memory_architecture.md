# Upper Agent Memory Architecture

## Phase 1: Working Memory + Local Mem0

The existing per-run `upper_agent_memory.json` remains the real-time working
memory. It owns the current place, active subgoal, recent landmarks, progress,
and failure state for one experiment.

Cross-experiment memory is stored by local Mem0:

```text
Upper Agent decision
  -> keep per-run upper_agent_memory.json
  -> select only high-value transitions
  -> asynchronous Mem0 infer=False write
  -> CPU multilingual embedding
  -> local Qdrant index
```

High-value transitions are:

- the executable subgoal changes;
- a subgoal completes;
- the task completes or fails;
- a new finding or failure reason appears.

Routine frame decisions are not stored. Mem0 never calls an extraction LLM,
so writes consume no API tokens and do not contend for GPU 1.

Before an Upper Agent call, semantic retrieval uses the current task, place,
subgoal, and recent landmarks. The defaults are:

- Top K: 3 memories;
- prompt budget: 600 characters total;
- timeout: 180 ms;
- reranking: disabled.

If loading, searching, or writing fails, the navigation call continues without
long-term memory. Retrieved memory is advisory context only; current visual and
safety evidence has priority.

Storage:

```text
output/agent_memory/
  mem0/
  qdrant/
  history.db
  graph_events.jsonl
```

## Phase 2: Graph Memory Preparation

Every high-value transition can also append a versioned graph event:

```text
scene --CONTAINS--> place
place --HAS_LANDMARK--> landmark
action --EXECUTED_AT--> place
action --RESULTED_IN--> outcome
```

Each event retains the source run, frame index, task, confidence, and timestamp.
This append-only JSONL is deliberately outside the real-time retrieval path.
After enough experiments have accumulated, it can be imported into Neo4j,
Kuzu, Memgraph, or another graph backend without changing Upper Agent control
logic.

The future graph retriever should return only a compact route summary, such as:

```text
At the glass-door intersection, turning right previously reached the sofa area.
```

It should use the same Top-K, character-budget, and timeout guardrails as Mem0.
