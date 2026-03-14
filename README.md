# Simple Agent

`simple-agent` is a deliberately small Rust project for learning how local AI agents work:

- it sends a user message to a local model
- it advertises a set of tools with JSON schemas
- it executes the tool the model selects
- it sends the tool result back to the model
- it repeats until the model returns a final answer

The default runtime is [Ollama](https://docs.ollama.com/api/introduction), which exposes a local HTTP API at `http://localhost:11434/api` and supports tool calling through its chat endpoint.

## Why this model/runtime choice

For a simple, free, local backend, Ollama is the cleanest starting point:

- it runs on macOS, Windows, and Linux
- it exposes a stable local API
- it supports tool calling in `/api/chat`
- it is easy for other people to reproduce on their own machines

Recommended default model: [`qwen3:4b`](https://ollama.com/library/qwen3)

- Ollama lists `qwen3:4b` as a 2.5GB model with a 256K context window.
- The Qwen3 page also calls out "expertise in agent capabilities" and strong external tool integration.

Lower-memory fallback: [`llama3.2:3b`](https://ollama.com/library/llama3.2)

- Ollama lists `llama3.2:3b` at 2.0GB with a 128K context window.
- The Llama 3.2 page explicitly mentions tool use as a target use case.

My default suggestion is:

- use `qwen3:4b` if you want the best shot at reliable tool selection on a typical laptop
- use `llama3.2` if you want the smallest broadly capable starting point

## Setup

1. Install Ollama from [the quickstart guide](https://docs.ollama.com/quickstart).
2. Start Ollama.
3. Pull a model:

```bash
ollama pull qwen3:4b
```

Or the lighter fallback:

```bash
ollama pull llama3.2
```

4. Run the agent:

```bash
cargo run
```

On startup, the app will:

- list the Ollama models already installed on your machine
- let you pick one to run
- warm that model before the chat loop starts
- stream responses token-by-token once you start chatting

Or pass a one-shot prompt:

```bash
cargo run -- "What tools do you have, and what files are in this workspace?"
```

If you want to skip the picker, pass a model explicitly:

```bash
cargo run -- --model qwen3:4b
```

## How the agent works

The core loop lives in [`src/agent.rs`](./src/agent.rs):

1. Send the current conversation and tool definitions to Ollama.
2. Check whether the model returned `tool_calls`.
3. Execute each requested tool locally in Rust.
4. Append each tool result as a `tool` message.
5. Ask the model for the next step or final answer.

## Included tools

- `get_current_time`
- `calculate`
- `list_files`
- `read_file`

The file tools are intentionally limited to the current workspace so the example stays safe and easy to understand.

## Useful commands

```bash
cargo run -- --model llama3.2 "Read Cargo.toml and summarize the dependencies."
cargo test
```

You can also point the app at a different Ollama server:

```bash
OLLAMA_HOST=http://127.0.0.1:11434 cargo run -- "What time is it?"
```

## Next directions

Once this base is working, the next useful upgrades would be:

- add a `search_workspace` tool powered by `rg`
- stream tool activity separately from model output
- add an approval gate before running file-writing or shell tools
- swap the local tools for MCP-backed tools once you want interoperability
