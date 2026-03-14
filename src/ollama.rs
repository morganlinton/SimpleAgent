use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
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

#[derive(Debug, Clone, Deserialize)]
struct StreamChatResponse {
    pub message: ChatMessage,
    pub done: bool,
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
    api_base_path: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct InstalledModel {
    pub name: String,
    pub size: u64,
}

#[derive(Debug, Deserialize)]
struct ModelListResponse {
    models: Vec<InstalledModel>,
}

impl OllamaClient {
    pub fn new(base_url: &str) -> Result<Self> {
        let (address, host_header, api_base_path) = parse_http_endpoint(base_url)?;
        Ok(Self {
            address,
            host_header,
            api_base_path,
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
        let response = self.send_request("POST", &self.chat_path(), Some(&payload))?;
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

    pub fn chat_stream<F>(
        &self,
        model: &str,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
        mut on_content: F,
    ) -> Result<ChatResponse>
    where
        F: FnMut(&str) -> Result<()>,
    {
        let request = ChatRequest {
            model,
            messages,
            tools,
            stream: true,
        };
        let payload = serde_json::to_vec(&request).context("failed to encode Ollama request")?;
        let mut response = self.send_stream_request("POST", &self.chat_path(), Some(&payload))?;

        if response.status_code != 200 {
            let mut body = String::new();
            response
                .body
                .read_to_string(&mut body)
                .context("failed to read Ollama error response")?;
            bail!("Ollama returned {}: {}", response.status_code, body);
        }

        let mut aggregated = ChatMessage {
            role: "assistant".to_string(),
            content: String::new(),
            tool_calls: None,
            tool_name: None,
        };
        let mut collected_tool_calls = Vec::new();

        while let Some(chunk) = read_stream_chunk(&mut response.body)? {
            on_content(&chunk.message.content)?;

            if aggregated.role == "assistant" && chunk.message.role != "assistant" {
                aggregated.role = chunk.message.role.clone();
            }

            if !chunk.message.content.is_empty() {
                aggregated.content.push_str(&chunk.message.content);
            }

            if let Some(tool_calls) = chunk.message.tool_calls {
                collected_tool_calls.extend(tool_calls);
            }

            if chunk.done {
                break;
            }
        }

        if !collected_tool_calls.is_empty() {
            aggregated.tool_calls = Some(collected_tool_calls);
        }

        Ok(ChatResponse {
            message: aggregated,
        })
    }

    pub fn list_models(&self) -> Result<Vec<InstalledModel>> {
        let response = self.send_request("GET", &self.tags_path(), None)?;
        if response.status_code != 200 {
            bail!(
                "Ollama returned {} while listing models: {}",
                response.status_code,
                String::from_utf8_lossy(&response.body)
            );
        }

        let model_list = serde_json::from_slice::<ModelListResponse>(&response.body)
            .context("failed to decode Ollama model list")?;
        Ok(model_list.models)
    }

    pub fn warm_model(&self, model: &str) -> Result<()> {
        let messages = [
            ChatMessage::system("Reply with exactly OK."),
            ChatMessage::user("OK"),
        ];
        self.chat(model, &messages, &[]).map(|_| ())
    }

    fn chat_path(&self) -> String {
        format!("{}/chat", self.api_base_path)
    }

    fn tags_path(&self) -> String {
        format!("{}/tags", self.api_base_path)
    }

    fn send_request(
        &self,
        method: &str,
        path: &str,
        payload: Option<&[u8]>,
    ) -> Result<ParsedHttpResponse> {
        let body = payload.unwrap_or(&[]);
        let mut stream = TcpStream::connect(&self.address)
            .with_context(|| format!("failed to reach Ollama at http://{}", self.address))?;

        let mut request_head = format!(
            "{method} {path} HTTP/1.1\r\nHost: {}\r\nConnection: close\r\n",
            self.host_header
        );
        if payload.is_some() {
            request_head.push_str("Content-Type: application/json\r\n");
            request_head.push_str(&format!("Content-Length: {}\r\n", body.len()));
        }
        request_head.push_str("\r\n");

        stream
            .write_all(request_head.as_bytes())
            .context("failed to write HTTP request head")?;
        if payload.is_some() {
            stream
                .write_all(body)
                .context("failed to write Ollama request body")?;
        }
        stream.flush().context("failed to flush Ollama request")?;

        let mut response_bytes = Vec::new();
        stream
            .read_to_end(&mut response_bytes)
            .context("failed to read Ollama response")?;

        parse_http_response(&response_bytes)
    }

    fn send_stream_request(
        &self,
        method: &str,
        path: &str,
        payload: Option<&[u8]>,
    ) -> Result<StreamingHttpResponse> {
        let body = payload.unwrap_or(&[]);
        let mut stream = TcpStream::connect(&self.address)
            .with_context(|| format!("failed to reach Ollama at http://{}", self.address))?;

        let mut request_head = format!(
            "{method} {path} HTTP/1.1\r\nHost: {}\r\nConnection: close\r\n",
            self.host_header
        );
        if payload.is_some() {
            request_head.push_str("Content-Type: application/json\r\n");
            request_head.push_str(&format!("Content-Length: {}\r\n", body.len()));
        }
        request_head.push_str("\r\n");

        stream
            .write_all(request_head.as_bytes())
            .context("failed to write HTTP request head")?;
        if payload.is_some() {
            stream
                .write_all(body)
                .context("failed to write Ollama request body")?;
        }
        stream.flush().context("failed to flush Ollama request")?;

        let mut reader = BufReader::new(stream);
        let (status_code, headers) = parse_http_response_head(&mut reader)?;
        let body_reader = build_body_reader(reader, &headers)?;

        Ok(StreamingHttpResponse {
            status_code,
            body: BufReader::new(body_reader),
        })
    }
}

#[derive(Debug)]
struct ParsedHttpResponse {
    status_code: u16,
    body: Vec<u8>,
}

struct StreamingHttpResponse {
    status_code: u16,
    body: BufReader<Box<dyn Read>>,
}

struct ChunkedReader<R: BufRead> {
    inner: R,
    chunk_remaining: usize,
    done: bool,
}

impl<R: BufRead> ChunkedReader<R> {
    fn new(inner: R) -> Self {
        Self {
            inner,
            chunk_remaining: 0,
            done: false,
        }
    }

    fn read_next_chunk_size(&mut self) -> Result<()> {
        let mut line = String::new();
        let bytes_read = self
            .inner
            .read_line(&mut line)
            .context("failed to read chunk size")?;
        if bytes_read == 0 {
            bail!("unexpected end of chunked response");
        }

        let trimmed = line.trim();
        let size_hex = trimmed.split(';').next().unwrap_or(trimmed);
        self.chunk_remaining =
            usize::from_str_radix(size_hex, 16).context("invalid hex chunk length")?;

        if self.chunk_remaining == 0 {
            self.done = true;
            self.consume_trailers()?;
        }

        Ok(())
    }

    fn consume_chunk_terminator(&mut self) -> Result<()> {
        let mut terminator = [0_u8; 2];
        self.inner
            .read_exact(&mut terminator)
            .context("failed to read chunk terminator")?;
        if terminator != *b"\r\n" {
            bail!("invalid chunk terminator");
        }
        Ok(())
    }

    fn consume_trailers(&mut self) -> Result<()> {
        loop {
            let mut line = String::new();
            let bytes_read = self
                .inner
                .read_line(&mut line)
                .context("failed to read chunk trailers")?;
            if bytes_read == 0 || line == "\r\n" {
                return Ok(());
            }
        }
    }
}

impl<R: BufRead> Read for ChunkedReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }

        if self.done {
            return Ok(0);
        }

        if self.chunk_remaining == 0 {
            self.read_next_chunk_size()
                .map_err(|error| std::io::Error::new(std::io::ErrorKind::Other, error))?;
            if self.done {
                return Ok(0);
            }
        }

        let to_read = self.chunk_remaining.min(buf.len());
        let bytes_read = self.inner.read(&mut buf[..to_read])?;
        if bytes_read == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "unexpected end of chunked response body",
            ));
        }

        self.chunk_remaining -= bytes_read;
        if self.chunk_remaining == 0 {
            self.consume_chunk_terminator()
                .map_err(|error| std::io::Error::new(std::io::ErrorKind::Other, error))?;
        }

        Ok(bytes_read)
    }
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

    let api_base_path = match raw_path.as_str() {
        "" => "/api".to_string(),
        "/api" => "/api".to_string(),
        "/api/chat" => "/api".to_string(),
        path => bail!(
            "unsupported Ollama path `{path}`; use a host like `http://127.0.0.1:11434` or `http://127.0.0.1:11434/api`"
        ),
    };

    Ok((host_port.to_string(), host_port.to_string(), api_base_path))
}

fn parse_http_response_head<R: BufRead>(reader: &mut R) -> Result<(u16, HashMap<String, String>)> {
    let mut status_line = String::new();
    let bytes_read = reader
        .read_line(&mut status_line)
        .context("failed to read HTTP status line")?;
    if bytes_read == 0 {
        bail!("missing HTTP status line");
    }

    let status_code = status_line
        .split_whitespace()
        .nth(1)
        .ok_or_else(|| anyhow!("missing HTTP status code"))?
        .parse::<u16>()
        .context("invalid HTTP status code")?;

    let mut headers = HashMap::new();
    loop {
        let mut line = String::new();
        let bytes_read = reader
            .read_line(&mut line)
            .context("failed to read HTTP headers")?;
        if bytes_read == 0 || line == "\r\n" {
            break;
        }

        if let Some((name, value)) = line.split_once(':') {
            headers.insert(name.trim().to_ascii_lowercase(), value.trim().to_string());
        }
    }

    Ok((status_code, headers))
}

fn build_body_reader(
    reader: BufReader<TcpStream>,
    headers: &HashMap<String, String>,
) -> Result<Box<dyn Read>> {
    if headers
        .get("transfer-encoding")
        .map(|value| value.to_ascii_lowercase().contains("chunked"))
        .unwrap_or(false)
    {
        return Ok(Box::new(ChunkedReader::new(reader)));
    }

    if let Some(length) = headers.get("content-length") {
        let content_length = length
            .parse::<u64>()
            .context("invalid content-length header")?;
        return Ok(Box::new(reader.take(content_length)));
    }

    Ok(Box::new(reader))
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

fn read_stream_chunk<R: BufRead>(reader: &mut R) -> Result<Option<StreamChatResponse>> {
    loop {
        let mut line = String::new();
        let bytes_read = reader
            .read_line(&mut line)
            .context("failed to read streaming chat chunk")?;
        if bytes_read == 0 {
            return Ok(None);
        }

        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let chunk =
            serde_json::from_str::<StreamChatResponse>(trimmed).context("invalid stream chunk")?;
        return Ok(Some(chunk));
    }
}

#[cfg(test)]
mod tests {
    use std::io::{BufReader, Cursor, Read};

    use super::{
        parse_http_endpoint, parse_http_response, parse_http_response_head, read_stream_chunk,
        ChunkedReader, ModelListResponse,
    };

    #[test]
    fn parses_default_ollama_endpoint() {
        let (address, host_header, api_base_path) =
            parse_http_endpoint("http://127.0.0.1:11434/api").expect("parse endpoint");

        assert_eq!(address, "127.0.0.1:11434");
        assert_eq!(host_header, "127.0.0.1:11434");
        assert_eq!(api_base_path, "/api");
    }

    #[test]
    fn decodes_chunked_http_response() {
        let response =
            b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n4\r\ntest\r\n0\r\n\r\n";
        let parsed = parse_http_response(response).expect("parse response");

        assert_eq!(parsed.status_code, 200);
        assert_eq!(parsed.body, b"test");
    }

    #[test]
    fn parses_model_list_response() {
        let response = br#"{
            "models": [
                { "name": "qwen3:4b", "size": 2684354560 },
                { "name": "llama3.2:latest", "size": 2147483648 }
            ]
        }"#;

        let parsed =
            serde_json::from_slice::<ModelListResponse>(response).expect("parse model list");

        assert_eq!(parsed.models.len(), 2);
        assert_eq!(parsed.models[0].name, "qwen3:4b");
        assert_eq!(parsed.models[1].size, 2_147_483_648);
    }

    #[test]
    fn parses_http_response_head() {
        let raw = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nX-Test: yes\r\n\r\n";
        let mut reader = BufReader::new(Cursor::new(raw));
        let (status_code, headers) = parse_http_response_head(&mut reader).expect("parse head");

        assert_eq!(status_code, 200);
        assert_eq!(
            headers.get("transfer-encoding"),
            Some(&"chunked".to_string())
        );
        assert_eq!(headers.get("x-test"), Some(&"yes".to_string()));
    }

    #[test]
    fn chunked_reader_decodes_stream_body() {
        let raw = b"4\r\ntest\r\n5\r\n-data\r\n0\r\n\r\n";
        let cursor = BufReader::new(Cursor::new(raw));
        let mut reader = ChunkedReader::new(cursor);
        let mut body = String::new();

        reader.read_to_string(&mut body).expect("read chunked body");

        assert_eq!(body, "test-data");
    }

    #[test]
    fn reads_stream_chunk_lines() {
        let raw = b"\n{\"message\":{\"role\":\"assistant\",\"content\":\"Hi\"},\"done\":false}\n{\"message\":{\"role\":\"assistant\",\"content\":\" there\"},\"done\":true}\n";
        let mut reader = BufReader::new(Cursor::new(raw));

        let first = read_stream_chunk(&mut reader)
            .expect("read first chunk")
            .expect("first chunk");
        let second = read_stream_chunk(&mut reader)
            .expect("read second chunk")
            .expect("second chunk");
        let end = read_stream_chunk(&mut reader).expect("read end");

        assert_eq!(first.message.content, "Hi");
        assert!(!first.done);
        assert_eq!(second.message.content, " there");
        assert!(second.done);
        assert!(end.is_none());
    }
}
