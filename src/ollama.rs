use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::TcpStream;

use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_name: Option<String>,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".to_string(),
            content: content.into(),
            tool_calls: None,
            tool_name: None,
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            content: content.into(),
            tool_calls: None,
            tool_name: None,
        }
    }

    pub fn tool_result(tool_name: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: "tool".to_string(),
            content: content.into(),
            tool_calls: None,
            tool_name: Some(tool_name.into()),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ToolDefinition {
    #[serde(rename = "type")]
    pub kind: String,
    pub function: ToolFunctionDefinition,
}

#[derive(Debug, Clone, Serialize)]
pub struct ToolFunctionDefinition {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub function: ToolFunctionCall,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolFunctionCall {
    pub name: String,
    #[serde(default)]
    pub arguments: Value,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChatResponse {
    pub message: ChatMessage,
}

#[derive(Debug, Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: &'a [ChatMessage],
    tools: &'a [ToolDefinition],
    stream: bool,
}

#[derive(Debug, Clone)]
pub struct OllamaClient {
    address: String,
    host_header: String,
    chat_path: String,
}

impl OllamaClient {
    pub fn new(base_url: &str) -> Result<Self> {
        let (address, host_header, chat_path) = parse_http_endpoint(base_url)?;
        Ok(Self {
            address,
            host_header,
            chat_path,
        })
    }

    pub fn chat(
        &self,
        model: &str,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
    ) -> Result<ChatResponse> {
        let request = ChatRequest {
            model,
            messages,
            tools,
            stream: false,
        };
        let payload = serde_json::to_vec(&request).context("failed to encode Ollama request")?;

        let mut stream = TcpStream::connect(&self.address)
            .with_context(|| format!("failed to reach Ollama at http://{}", self.address))?;
        let request_head = format!(
            "POST {} HTTP/1.1\r\nHost: {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            self.chat_path,
            self.host_header,
            payload.len()
        );

        stream
            .write_all(request_head.as_bytes())
            .context("failed to write HTTP request head")?;
        stream
            .write_all(&payload)
            .context("failed to write Ollama request body")?;
        stream.flush().context("failed to flush Ollama request")?;

        let mut response_bytes = Vec::new();
        stream
            .read_to_end(&mut response_bytes)
            .context("failed to read Ollama response")?;

        let response = parse_http_response(&response_bytes)?;
        if response.status_code != 200 {
            bail!(
                "Ollama returned {}: {}",
                response.status_code,
                String::from_utf8_lossy(&response.body)
            );
        }

        serde_json::from_slice::<ChatResponse>(&response.body)
            .context("failed to decode Ollama response")
    }
}

#[derive(Debug)]
struct ParsedHttpResponse {
    status_code: u16,
    body: Vec<u8>,
}

fn parse_http_endpoint(base_url: &str) -> Result<(String, String, String)> {
    let normalized = base_url.trim().trim_end_matches('/');
    let without_scheme = normalized
        .strip_prefix("http://")
        .ok_or_else(|| anyhow!("only plain `http://` Ollama hosts are supported"))?;

    let (host_port, raw_path) = match without_scheme.split_once('/') {
        Some((host_port, path)) => (host_port, format!("/{}", path.trim_start_matches('/'))),
        None => (without_scheme, String::new()),
    };

    if host_port.is_empty() {
        bail!("Ollama host cannot be empty");
    }

    let chat_path = match raw_path.as_str() {
        "" => "/api/chat".to_string(),
        "/api" => "/api/chat".to_string(),
        "/api/chat" => "/api/chat".to_string(),
        path => bail!(
            "unsupported Ollama path `{path}`; use a host like `http://127.0.0.1:11434` or `http://127.0.0.1:11434/api`"
        ),
    };

    Ok((host_port.to_string(), host_port.to_string(), chat_path))
}

fn parse_http_response(response: &[u8]) -> Result<ParsedHttpResponse> {
    let header_end = response
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .ok_or_else(|| anyhow!("invalid HTTP response from Ollama"))?;

    let head = &response[..header_end];
    let body = &response[header_end + 4..];
    let head_text = String::from_utf8_lossy(head);
    let mut lines = head_text.split("\r\n");
    let status_line = lines.next().ok_or_else(|| anyhow!("missing status line"))?;
    let status_code = status_line
        .split_whitespace()
        .nth(1)
        .ok_or_else(|| anyhow!("missing HTTP status code"))?
        .parse::<u16>()
        .context("invalid HTTP status code")?;

    let mut headers = HashMap::new();
    for line in lines {
        if let Some((name, value)) = line.split_once(':') {
            headers.insert(name.trim().to_ascii_lowercase(), value.trim().to_string());
        }
    }

    let decoded_body = if headers
        .get("transfer-encoding")
        .map(|value| value.to_ascii_lowercase().contains("chunked"))
        .unwrap_or(false)
    {
        decode_chunked_body(body)?
    } else {
        body.to_vec()
    };

    Ok(ParsedHttpResponse {
        status_code,
        body: decoded_body,
    })
}

fn decode_chunked_body(mut bytes: &[u8]) -> Result<Vec<u8>> {
    let mut decoded = Vec::new();

    loop {
        let size_end = find_crlf(bytes).ok_or_else(|| anyhow!("invalid chunked response"))?;
        let size_text = std::str::from_utf8(&bytes[..size_end]).context("invalid chunk size")?;
        let size_hex = size_text.split(';').next().unwrap_or(size_text).trim();
        let chunk_size = usize::from_str_radix(size_hex, 16).context("invalid hex chunk length")?;
        bytes = &bytes[size_end + 2..];

        if chunk_size == 0 {
            break;
        }

        if bytes.len() < chunk_size + 2 {
            bail!("truncated chunked response body");
        }

        decoded.extend_from_slice(&bytes[..chunk_size]);
        bytes = &bytes[chunk_size + 2..];
    }

    Ok(decoded)
}

fn find_crlf(bytes: &[u8]) -> Option<usize> {
    bytes.windows(2).position(|window| window == b"\r\n")
}

#[cfg(test)]
mod tests {
    use super::{parse_http_endpoint, parse_http_response};

    #[test]
    fn parses_default_ollama_endpoint() {
        let (address, host_header, path) =
            parse_http_endpoint("http://127.0.0.1:11434/api").expect("parse endpoint");

        assert_eq!(address, "127.0.0.1:11434");
        assert_eq!(host_header, "127.0.0.1:11434");
        assert_eq!(path, "/api/chat");
    }

    #[test]
    fn decodes_chunked_http_response() {
        let response =
            b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n4\r\ntest\r\n0\r\n\r\n";
        let parsed = parse_http_response(response).expect("parse response");

        assert_eq!(parsed.status_code, 200);
        assert_eq!(parsed.body, b"test");
    }
}
