mod agent;
mod ollama;
mod tools;

use std::env;
use std::io::{self, Write};
use std::process;

use agent::Agent;
use anyhow::{Context, Result};
use ollama::OllamaClient;
use tools::ToolRegistry;

const DEFAULT_MODEL: &str = "qwen3:4b";
const DEFAULT_HOST: &str = "http://127.0.0.1:11434";

#[derive(Debug)]
struct Config {
    model: String,
    host: String,
    prompt: Option<String>,
}

impl Config {
    fn from_args() -> Result<Self> {
        let mut model =
            env::var("SIMPLE_AGENT_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string());
        let mut host = env::var("OLLAMA_HOST").unwrap_or_else(|_| DEFAULT_HOST.to_string());
        let mut prompt_parts = Vec::new();

        let mut args = env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--model" => {
                    model = args
                        .next()
                        .context("`--model` expects a value, for example `--model qwen3:4b`")?;
                }
                "--host" => {
                    host = args.next().context(
                        "`--host` expects a value, for example `--host http://127.0.0.1:11434`",
                    )?;
                }
                "-h" | "--help" => {
                    print_help();
                    process::exit(0);
                }
                _ => prompt_parts.push(arg),
            }
        }

        let prompt = (!prompt_parts.is_empty()).then(|| prompt_parts.join(" "));

        Ok(Self {
            model,
            host,
            prompt,
        })
    }
}

fn print_help() {
    println!("simple-agent");
    println!();
    println!("Usage:");
    println!("  cargo run -- [--model MODEL] [--host URL] [prompt]");
    println!();
    println!("Examples:");
    println!("  cargo run -- \"What tools do you have?\"");
    println!("  cargo run -- --model llama3.2 \"List the files in src\"");
    println!();
    println!("Environment variables:");
    println!("  SIMPLE_AGENT_MODEL   Default model name (default: {DEFAULT_MODEL})");
    println!("  OLLAMA_HOST          Ollama host URL (default: {DEFAULT_HOST})");
}

fn main() -> Result<()> {
    let config = Config::from_args()?;
    let workspace_root = env::current_dir()
        .context("failed to detect current workspace")?
        .canonicalize()
        .context("failed to resolve workspace path")?;
    let client = OllamaClient::new(&config.host)?;
    let agent = Agent::new(
        client,
        ToolRegistry::new(workspace_root),
        config.model.clone(),
    );

    if let Some(prompt) = config.prompt {
        let answer = agent.respond(&prompt)?;
        println!("{answer}");
        return Ok(());
    }

    run_repl(&agent, &config.model)
}

fn run_repl(agent: &Agent, model: &str) -> Result<()> {
    println!("Simple Agent");
    println!("Model: {model}");
    println!("Type `exit` or `quit` to stop.");
    println!();

    let stdin = io::stdin();
    loop {
        print!("you> ");
        io::stdout().flush().context("failed to flush stdout")?;

        let mut line = String::new();
        let bytes_read = stdin
            .read_line(&mut line)
            .context("failed to read from stdin")?;
        if bytes_read == 0 {
            println!();
            break;
        }

        let input = line.trim();
        if input.is_empty() {
            continue;
        }

        if matches!(input, "exit" | "quit") {
            break;
        }

        match agent.respond(input) {
            Ok(answer) => {
                println!("agent> {answer}");
                println!();
            }
            Err(error) => {
                eprintln!("error: {error}");
                eprintln!();
            }
        }
    }

    Ok(())
}
