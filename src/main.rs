mod agent;
mod ollama;
mod tools;

use std::env;
use std::io::{self, Write};
use std::process;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::thread;
use std::time::{Duration, Instant};

use agent::Agent;
use anyhow::{bail, Context, Result};
use ollama::{InstalledModel, OllamaClient};
use tools::ToolRegistry;

const DEFAULT_MODEL: &str = "qwen3:4b";
const DEFAULT_HOST: &str = "http://127.0.0.1:11434";

#[derive(Debug)]
struct Config {
    model: Option<String>,
    host: String,
    prompt: Option<String>,
}

impl Config {
    fn from_args() -> Result<Self> {
        let mut model = env::var("SIMPLE_AGENT_MODEL").ok();
        let mut host = env::var("OLLAMA_HOST").unwrap_or_else(|_| DEFAULT_HOST.to_string());
        let mut prompt_parts = Vec::new();

        let mut args = env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--model" => {
                    model = Some(
                        args.next()
                            .context("`--model` expects a value, for example `--model qwen3:4b`")?,
                    );
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
    println!("Interactive mode:");
    println!("  If you do not pass `--model`, the app will list your installed Ollama models,");
    println!("  let you pick one, and warm it up before the REPL starts.");
    println!();
    println!("Examples:");
    println!("  cargo run");
    println!("  cargo run -- \"What tools do you have?\"");
    println!("  cargo run -- --model llama3.2 \"List the files in src\"");
    println!();
    println!("Environment variables:");
    println!("  SIMPLE_AGENT_MODEL   Skip the picker and use this model");
    println!("  OLLAMA_HOST          Ollama host URL (default: {DEFAULT_HOST})");
}

fn main() -> Result<()> {
    let config = Config::from_args()?;
    let workspace_root = env::current_dir()
        .context("failed to detect current workspace")?
        .canonicalize()
        .context("failed to resolve workspace path")?;
    let client = OllamaClient::new(&config.host)?;
    let model = resolve_model(&client, &config)?;

    if config.prompt.is_none() {
        warm_selected_model(&client, &model)?;
    }

    let agent = Agent::new(client, ToolRegistry::new(workspace_root), model.clone());

    if let Some(prompt) = config.prompt {
        stream_response(&agent, &prompt, None)?;
        return Ok(());
    }

    run_repl(&agent, &model)
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

        match stream_response(agent, input, Some("agent> ")) {
            Ok(()) => {
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

fn stream_response(agent: &Agent, prompt: &str, prefix: Option<&str>) -> Result<()> {
    let spinner_message = match prefix {
        Some(prefix) => format!("{prefix}Thinking"),
        None => "Thinking".to_string(),
    };
    let mut spinner = Some(Spinner::start(&spinner_message));

    let result = agent.respond_streaming(prompt, |chunk| {
        if let Some(active_spinner) = spinner.take() {
            active_spinner.stop(prefix)?;
        }

        print!("{chunk}");
        io::stdout().flush().context("failed to flush stdout")
    });

    if let Some(active_spinner) = spinner.take() {
        active_spinner.stop(None)?;
    }

    result?;

    println!();
    Ok(())
}

fn resolve_model(client: &OllamaClient, config: &Config) -> Result<String> {
    if let Some(model) = &config.model {
        return Ok(model.clone());
    }

    if config.prompt.is_some() {
        return Ok(DEFAULT_MODEL.to_string());
    }

    select_model_interactively(client)
}

fn select_model_interactively(client: &OllamaClient) -> Result<String> {
    let models = client.list_models()?;
    if models.is_empty() {
        bail!("No local Ollama models are installed. Run `ollama pull {DEFAULT_MODEL}` first.");
    }

    let default_index = models
        .iter()
        .position(|model| model.name == DEFAULT_MODEL)
        .unwrap_or(0);

    println!("Select a model to run:");
    for (index, model) in models.iter().enumerate() {
        let marker = if index == default_index {
            " [recommended]"
        } else {
            ""
        };
        println!(
            "  {}. {} ({}){}",
            index + 1,
            model.name,
            format_model_size(model.size),
            marker
        );
    }
    println!();
    println!("Press Enter to use {}.", models[default_index].name);

    let stdin = io::stdin();
    loop {
        print!("model> ");
        io::stdout().flush().context("failed to flush stdout")?;

        let mut line = String::new();
        let bytes_read = stdin
            .read_line(&mut line)
            .context("failed to read model selection")?;
        if bytes_read == 0 {
            println!();
            return Ok(models[default_index].name.clone());
        }

        let input = line.trim();
        if input.is_empty() {
            return Ok(models[default_index].name.clone());
        }

        if let Ok(index) = input.parse::<usize>() {
            if (1..=models.len()).contains(&index) {
                return Ok(models[index - 1].name.clone());
            }
        }

        if let Some(model) = match_model_name(&models, input) {
            return Ok(model.name.clone());
        }

        println!(
            "Enter a number from 1-{} or the exact name of an installed model.",
            models.len()
        );
    }
}

fn match_model_name<'a>(models: &'a [InstalledModel], input: &str) -> Option<&'a InstalledModel> {
    models.iter().find(|model| model.name == input)
}

fn warm_selected_model(client: &OllamaClient, model: &str) -> Result<()> {
    let message = format!("Loading model `{model}`");
    run_with_spinner(&message, || client.warm_model(model))?;
    println!("Model `{model}` is ready.");
    println!();
    Ok(())
}

fn format_model_size(size_bytes: u64) -> String {
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    const MIB: f64 = 1024.0 * 1024.0;

    let size = size_bytes as f64;
    if size >= GIB {
        return format!("{:.1} GB", size / GIB);
    }

    format!("{:.0} MB", size / MIB)
}

fn run_with_spinner<T, F>(message: &str, task: F) -> Result<T>
where
    F: FnOnce() -> Result<T>,
{
    let spinner = Spinner::start(message);
    let result = task();
    spinner.stop(None)?;
    result
}

struct Spinner {
    active: Arc<AtomicBool>,
    handle: Option<thread::JoinHandle<()>>,
    width: usize,
}

impl Spinner {
    fn start(message: &str) -> Self {
        let active = Arc::new(AtomicBool::new(true));
        let active_for_thread = Arc::clone(&active);
        let spinner_message = message.to_string();
        let width = spinner_message.len() + 16;

        let handle = thread::spawn(move || {
            let frames = ['|', '/', '-', '\\'];
            let start = Instant::now();
            let mut frame_index = 0;

            while active_for_thread.load(Ordering::Relaxed) {
                let elapsed = start.elapsed().as_secs_f32();
                print!(
                    "\r{} {} {:.1}s",
                    frames[frame_index % frames.len()],
                    spinner_message,
                    elapsed
                );
                let _ = io::stdout().flush();
                thread::sleep(Duration::from_millis(120));
                frame_index += 1;
            }
        });

        Self {
            active,
            handle: Some(handle),
            width,
        }
    }

    fn stop(mut self, restore_text: Option<&str>) -> Result<()> {
        self.active.store(false, Ordering::Relaxed);

        if let Some(handle) = self.handle.take() {
            handle.join().expect("spinner thread should not panic");
        }

        print!("\r{}\r", " ".repeat(self.width));
        if let Some(restore_text) = restore_text {
            print!("{restore_text}");
        }
        io::stdout().flush().context("failed to flush stdout")?;

        Ok(())
    }
}
