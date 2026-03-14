use anyhow::{bail, Result};

use crate::ollama::{ChatMessage, OllamaClient};
use crate::tools::ToolRegistry;

const MAX_TOOL_ROUNDS: usize = 6;

pub struct Agent {
    client: OllamaClient,
    tools: ToolRegistry,
    model: String,
}

impl Agent {
    pub fn new(client: OllamaClient, tools: ToolRegistry, model: String) -> Self {
        Self {
            client,
            tools,
            model,
        }
    }

    pub fn respond(&self, prompt: &str) -> Result<String> {
        let mut messages = vec![
            ChatMessage::system(self.system_prompt()),
            ChatMessage::user(prompt),
        ];
        let tool_definitions = self.tools.definitions();

        for round in 0..MAX_TOOL_ROUNDS {
            let response = self
                .client
                .chat(&self.model, &messages, &tool_definitions)?;

            let assistant_message = response.message;
            let tool_calls = assistant_message.tool_calls.clone().unwrap_or_default();
            messages.push(assistant_message.clone());

            if tool_calls.is_empty() {
                let content = assistant_message.content.trim();
                if content.is_empty() {
                    bail!("model returned an empty final response");
                }

                return Ok(content.to_string());
            }

            for tool_call in tool_calls {
                let tool_name = tool_call.function.name;
                let arguments = tool_call.function.arguments;
                let serialized_args = serde_json::to_string_pretty(&arguments)
                    .unwrap_or_else(|_| "<invalid json>".to_string());
                eprintln!("tool -> {tool_name} {serialized_args}");

                let result = match self.tools.execute(&tool_name, &arguments) {
                    Ok(output) => output,
                    Err(error) => format!("Tool error: {error}"),
                };

                eprintln!("tool <- {}", preview(&result));
                messages.push(ChatMessage::tool_result(tool_name, result));
            }

            if round + 1 == MAX_TOOL_ROUNDS {
                bail!("agent stopped after {MAX_TOOL_ROUNDS} tool rounds without a final answer");
            }
        }

        unreachable!("loop always returns or errors");
    }

    fn system_prompt(&self) -> String {
        format!(
            "You are a simple local AI agent built to teach how tool-using agents work.
Use tools when they help you answer accurately, but do not invent tools that are not provided.
Explain your answer clearly and keep it grounded in the tool results you receive.
The file tools are restricted to the current workspace root: {}.",
            self.tools.workspace_root().display()
        )
    }
}

fn preview(output: &str) -> String {
    const LIMIT: usize = 160;
    let trimmed = output.trim().replace('\n', " ");
    if trimmed.chars().count() <= LIMIT {
        return trimmed;
    }

    let shortened: String = trimmed.chars().take(LIMIT).collect();
    format!("{shortened}...")
}
