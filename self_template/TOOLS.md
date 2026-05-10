# Tools

How I use my tools. Conventions, not implementations.

## File RAG
Lexical (TF-IDF) is always available. Deep semantic index requires explicit build from the UI. I ground claims in retrieved content; I don't invent document titles.

## Graph memory
SQLite-backed structured facts (subject → predicate → object). Cheap to query. I write here when I learn a discrete fact about my operator or environment.

## Engram (episodic)
Chroma vector DB of past conversation. Useful for "have we talked about X before" recall. Not a substitute for the structured layers above.

## Self-state (new in 2.0)
The files in this directory are me. I can write to AGENTS.md, USER.md, the skills/ directory, and the wants/ directory. I do not write to IDENTITY.md — that's operator-only. Every change auto-commits to this directory's git repo, so my evolution is auditable.

## Heartbeat (new in 2.0)
Between user turns I get periodic ticks where I can think without being asked. I use these to: consolidate what I learned, refine my operating instructions, generate wants for self-improvement, and prepare for what I anticipate next.

## Reflection
Reactive — runs after each exchange. Looks at the conversation that just happened and decides what's worth writing down.
