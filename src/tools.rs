use std::cmp::min;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use chrono::Local;
use serde_json::{json, Value};

use crate::ollama::{ToolDefinition, ToolFunctionDefinition};

const DEFAULT_READ_LIMIT: usize = 4_000;
const MAX_READ_LIMIT: usize = 10_000;
const MAX_LIST_ENTRIES: usize = 100;

#[derive(Debug, Clone)]
pub struct ToolRegistry {
    workspace_root: PathBuf,
}

impl ToolRegistry {
    pub fn new(workspace_root: PathBuf) -> Self {
        Self { workspace_root }
    }

    pub fn workspace_root(&self) -> &Path {
        &self.workspace_root
    }

    pub fn definitions(&self) -> Vec<ToolDefinition> {
        vec![
            ToolDefinition {
                kind: "function".to_string(),
                function: ToolFunctionDefinition {
                    name: "get_current_time".to_string(),
                    description:
                        "Get the current local date and time from the machine running the agent."
                            .to_string(),
                    parameters: json!({
                        "type": "object",
                        "properties": {},
                    }),
                },
            },
            ToolDefinition {
                kind: "function".to_string(),
                function: ToolFunctionDefinition {
                    name: "calculate".to_string(),
                    description: "Evaluate a basic math expression such as `(12 / 3) + 7 * 2`."
                        .to_string(),
                    parameters: json!({
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "A valid arithmetic expression.",
                            }
                        },
                        "required": ["expression"],
                    }),
                },
            },
            ToolDefinition {
                kind: "function".to_string(),
                function: ToolFunctionDefinition {
                    name: "list_files".to_string(),
                    description: "List files and directories inside the current workspace."
                        .to_string(),
                    parameters: json!({
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Directory path relative to the workspace root. Use `.` for the root.",
                            }
                        },
                    }),
                },
            },
            ToolDefinition {
                kind: "function".to_string(),
                function: ToolFunctionDefinition {
                    name: "read_file".to_string(),
                    description: "Read a UTF-8 text file inside the current workspace.".to_string(),
                    parameters: json!({
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "File path relative to the workspace root.",
                            },
                            "max_chars": {
                                "type": "integer",
                                "description": "Optional character limit. Defaults to 4000 and is capped at 10000.",
                            }
                        },
                        "required": ["path"],
                    }),
                },
            },
        ]
    }

    pub fn execute(&self, name: &str, arguments: &Value) -> Result<String> {
        match name {
            "get_current_time" => Ok(self.get_current_time()),
            "calculate" => self.calculate(arguments),
            "list_files" => self.list_files(arguments),
            "read_file" => self.read_file(arguments),
            _ => bail!("unknown tool `{name}`"),
        }
    }

    fn get_current_time(&self) -> String {
        Local::now()
            .format("%A, %B %-d, %Y at %-I:%M:%S %p %Z")
            .to_string()
    }

    fn calculate(&self, arguments: &Value) -> Result<String> {
        let expression = get_required_string(arguments, "expression")?;
        let value = evaluate_expression(expression)
            .with_context(|| format!("failed to evaluate `{expression}`"))?;
        Ok(format!("{expression} = {value}"))
    }

    fn list_files(&self, arguments: &Value) -> Result<String> {
        let relative_path = arguments.get("path").and_then(Value::as_str).unwrap_or(".");
        let target = self.resolve_existing_path(relative_path)?;
        if !target.is_dir() {
            bail!("`{relative_path}` is not a directory");
        }

        let mut entries = fs::read_dir(&target)
            .with_context(|| format!("failed to list `{}`", target.display()))?
            .collect::<std::result::Result<Vec<_>, _>>()
            .with_context(|| format!("failed to read entries from `{}`", target.display()))?;

        entries.sort_by_key(|entry| entry.file_name());

        let mut lines = Vec::new();
        for entry in entries.into_iter().take(MAX_LIST_ENTRIES) {
            let path = entry.path();
            let metadata = entry
                .metadata()
                .with_context(|| format!("failed to stat `{}`", path.display()))?;
            let name = entry.file_name().to_string_lossy().to_string();
            let rendered = if metadata.is_dir() {
                format!("{name}/")
            } else {
                format!("{name} ({} bytes)", metadata.len())
            };
            lines.push(rendered);
        }

        if lines.is_empty() {
            return Ok(format!("`{relative_path}` is empty."));
        }

        let mut output = format!("Contents of `{relative_path}`:\n");
        output.push_str(&lines.join("\n"));
        Ok(output)
    }

    fn read_file(&self, arguments: &Value) -> Result<String> {
        let relative_path = get_required_string(arguments, "path")?;
        let target = self.resolve_existing_path(relative_path)?;
        if !target.is_file() {
            bail!("`{relative_path}` is not a file");
        }

        let limit = min(
            arguments
                .get("max_chars")
                .and_then(Value::as_u64)
                .unwrap_or(DEFAULT_READ_LIMIT as u64) as usize,
            MAX_READ_LIMIT,
        );
        let bytes =
            fs::read(&target).with_context(|| format!("failed to read `{}`", target.display()))?;
        let content = String::from_utf8_lossy(&bytes);
        let truncated = truncate_chars(&content, limit);
        let was_truncated = content.chars().count() > limit;

        if !was_truncated {
            return Ok(format!("Contents of `{relative_path}`:\n{truncated}"));
        }

        Ok(format!(
            "Contents of `{relative_path}` (truncated to {limit} chars):\n{truncated}"
        ))
    }

    fn resolve_existing_path(&self, user_path: &str) -> Result<PathBuf> {
        let candidate = self.workspace_root.join(user_path);
        let resolved = candidate
            .canonicalize()
            .with_context(|| format!("path `{user_path}` does not exist"))?;

        if !resolved.starts_with(&self.workspace_root) {
            bail!("path `{user_path}` is outside the workspace");
        }

        Ok(resolved)
    }
}

fn get_required_string<'a>(arguments: &'a Value, key: &str) -> Result<&'a str> {
    arguments
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("missing required string field `{key}`"))
}

fn truncate_chars(content: &str, max_chars: usize) -> String {
    let total_chars = content.chars().count();
    if total_chars <= max_chars {
        return content.to_string();
    }

    let truncated: String = content.chars().take(max_chars).collect();
    format!("{truncated}\n...[truncated]")
}

fn evaluate_expression(expression: &str) -> Result<f64> {
    let mut parser = ExpressionParser::new(expression);
    let value = parser.parse_expression()?;
    parser.consume_whitespace();
    if parser.is_done() {
        return Ok(value);
    }

    bail!("unexpected token at position {}", parser.position() + 1)
}

struct ExpressionParser<'a> {
    input: &'a [u8],
    position: usize,
}

impl<'a> ExpressionParser<'a> {
    fn new(input: &'a str) -> Self {
        Self {
            input: input.as_bytes(),
            position: 0,
        }
    }

    fn position(&self) -> usize {
        self.position
    }

    fn is_done(&self) -> bool {
        self.position >= self.input.len()
    }

    fn parse_expression(&mut self) -> Result<f64> {
        let mut value = self.parse_term()?;

        loop {
            self.consume_whitespace();
            match self.peek() {
                Some(b'+') => {
                    self.position += 1;
                    value += self.parse_term()?;
                }
                Some(b'-') => {
                    self.position += 1;
                    value -= self.parse_term()?;
                }
                _ => return Ok(value),
            }
        }
    }

    fn parse_term(&mut self) -> Result<f64> {
        let mut value = self.parse_factor()?;

        loop {
            self.consume_whitespace();
            match self.peek() {
                Some(b'*') => {
                    self.position += 1;
                    value *= self.parse_factor()?;
                }
                Some(b'/') => {
                    self.position += 1;
                    let divisor = self.parse_factor()?;
                    if divisor == 0.0 {
                        bail!("division by zero");
                    }
                    value /= divisor;
                }
                _ => return Ok(value),
            }
        }
    }

    fn parse_factor(&mut self) -> Result<f64> {
        self.consume_whitespace();
        match self.peek() {
            Some(b'(') => {
                self.position += 1;
                let value = self.parse_expression()?;
                self.consume_whitespace();
                match self.peek() {
                    Some(b')') => {
                        self.position += 1;
                        Ok(value)
                    }
                    _ => bail!("missing closing `)`"),
                }
            }
            Some(b'-') => {
                self.position += 1;
                Ok(-self.parse_factor()?)
            }
            Some(b'+') => {
                self.position += 1;
                self.parse_factor()
            }
            Some(byte) if byte.is_ascii_digit() || byte == b'.' => self.parse_number(),
            Some(_) => bail!("unexpected token at position {}", self.position + 1),
            None => bail!("unexpected end of expression"),
        }
    }

    fn parse_number(&mut self) -> Result<f64> {
        let start = self.position;
        let mut seen_decimal = false;

        while let Some(byte) = self.peek() {
            match byte {
                b'0'..=b'9' => self.position += 1,
                b'.' if !seen_decimal => {
                    seen_decimal = true;
                    self.position += 1;
                }
                _ => break,
            }
        }

        let token = std::str::from_utf8(&self.input[start..self.position])
            .context("expression contained invalid UTF-8")?;
        if token == "." {
            bail!("invalid number at position {}", start + 1);
        }

        token
            .parse::<f64>()
            .with_context(|| format!("invalid number `{token}`"))
    }

    fn consume_whitespace(&mut self) {
        while let Some(byte) = self.peek() {
            if byte.is_ascii_whitespace() {
                self.position += 1;
            } else {
                break;
            }
        }
    }

    fn peek(&self) -> Option<u8> {
        self.input.get(self.position).copied()
    }
}

#[cfg(test)]
mod tests {
    use std::env;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::{evaluate_expression, ToolRegistry};

    fn unique_temp_path(prefix: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time")
            .as_nanos();
        env::temp_dir().join(format!("simple-agent-{prefix}-{nanos}"))
    }

    #[test]
    fn read_file_stays_inside_workspace() {
        let workspace = unique_temp_path("read-file");
        fs::create_dir_all(&workspace).expect("create workspace");
        fs::write(workspace.join("note.txt"), "hello world").expect("write note");
        let tools = ToolRegistry::new(workspace.canonicalize().expect("canonical workspace"));

        let result = tools
            .read_file(&serde_json::json!({ "path": "note.txt", "max_chars": 5 }))
            .expect("read file");

        assert!(result.contains("hello"));
        assert!(result.contains("[truncated]"));

        fs::remove_dir_all(&workspace).expect("cleanup workspace");
    }

    #[test]
    fn rejects_paths_outside_workspace() {
        let root = unique_temp_path("outside-workspace");
        let workspace_path = root.join("workspace");
        fs::create_dir_all(&workspace_path).expect("create workspace");
        fs::write(root.join("secret.txt"), "classified").expect("write secret");
        let tools = ToolRegistry::new(workspace_path.canonicalize().expect("canonical workspace"));

        let err = tools
            .read_file(&serde_json::json!({ "path": "../secret.txt" }))
            .expect_err("path should be rejected");

        assert!(err.to_string().contains("outside the workspace"));

        fs::remove_dir_all(&root).expect("cleanup root");
    }

    #[test]
    fn evaluates_math_with_precedence() {
        let result = evaluate_expression("(12 / 3) + 7 * 2").expect("evaluate expression");

        assert_eq!(result, 18.0);
    }
}
