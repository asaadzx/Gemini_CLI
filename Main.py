import asyncio
from google import genai 
from PIL import Image
import io
import click
import os
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text
from rich.theme import Theme
from typing import Optional

# --- API Key and Environment Setup ---
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "YOUR_API_KEY")
if GOOGLE_API_KEY == "YOUR_API_KEY": 
    console = Console()
    console.print("[warning]⚠️ Please set the GOOGLE_API_KEY environment variable for security.[/warning]")
else:
    console = Console()
    console.print("[info]🔑 Using API key from environment or hardcoded fallback.[/info]")

# --- Rich UI Setup ---
custom_theme = Theme({
    "info": "cyan",
    "warning": "bold yellow",
    "danger": "bold red",
    "success": "green",
    "panel.border": "bright_blue",
    "gemini": "yellow",
    "user": "bright_magenta",
    "markdown": "white",
    "command": "bold green"
})
console = Console(theme=custom_theme)

# --- Model and Client Initialization ---
DEFAULT_MODEL_ID = "gemini-2.0-flash-exp"  # Your choice
DEFAULT_API_VERSION = "v1alpha"  # Your preference

def initialize_client(api_version=DEFAULT_API_VERSION):
    """Initializes the Gemini client."""
    try:
        client = genai.Client(api_key=GOOGLE_API_KEY, http_options={'api_version': api_version})
        console.print("[success]🚀 Client initialized successfully![/success]")
        return client
    except AttributeError as e:
        console.print(f"[danger]❌ Error: 'Client' not found in genai module. Check import and SDK.[/danger]")
        console.print(f"[danger]Details: {e}[/danger]")
        exit(1)
    except Exception as e:
        console.print(f"[danger]❌ Error initializing client: {e}[/danger]")
        exit(1)

# --- Helper Functions ---
async def get_image_data(image_path: str) -> Optional[bytes]:
    """Loads and validates image data."""
    try:
        with open(image_path, "rb") as image_file:
            return image_file.read()
    except FileNotFoundError:
        console.print(f"[danger]❌ Image file not found: {image_path}[/danger]")
        return None
    except Exception as e:
        console.print(f"[danger]❌ Error reading image: {e}[/danger]")
        return None

async def send_message(session, message: str, image_path: Optional[str] = None) -> bool:
    """Sends a message with optional image to the session."""
    inputs = []
    if image_path:
        image_data = await get_image_data(image_path)
        if image_data:
            try:
                image = Image.open(io.BytesIO(image_data))
                inputs.append(image)
                console.print(f"[success]🖼️ Image '{image_path}' included.[/success]")
            except Exception as e:
                console.print(f"[danger]❌ Error processing image: {e}[/danger]")
                return False
    inputs.append(message)
    try:
        await session.send(input=inputs, end_of_turn=True)
        return True
    except Exception as e:
        console.print(f"[danger]❌ Error sending message: {e}[/danger]")
        return False

async def receive_responses(session):
    """Receives and displays streaming responses."""
    try:
        console.print("[gemini]🤖 Gemini:[/gemini] ", end=" ")
        async for response in session.receive():
            if hasattr(response, 'text') and response.text:
                print(response.text, end="", flush=True)
        print()
    except Exception as e:
        console.print(f"[danger]❌ Error receiving responses: {e}[/danger]")

def show_help():
    """Displays a help table with all commands."""
    table = Table(title="📜 Chatbot Commands", style="panel.border", header_style="command")
    table.add_column("Command", style="bold cyan")
    table.add_column("Description", style="markdown")
    table.add_column("Example", style="info")
    
    table.add_row(
        "/image <path>",
        "Send an image with an optional message",
        "/image photo.jpg What’s this?"
    )
    table.add_row(
        "help",
        "Show this command list",
        "help"
    )
    table.add_row(
        "exit, quit, bye",
        "Exit the chatbot",
        "exit"
    )
    table.add_row(
        "<message>",
        "Send a text message to Gemini",
        "Hello, how are you?"
    )
    
    console.print(Panel(table, title="✨ Gemini Chatbot Help ✨", border_style="panel.border", padding=(1, 2)))

async def process_input(user_input: str) -> tuple[str, Optional[str]]:
    """Processes user input, handling commands and image paths."""
    user_input = user_input.strip()
    if not user_input:
        return "", None
    
    parts = user_input.split()
    command = parts[0].lower()
    
    if command == "help":
        show_help()
        return "", None
    elif command == "/image":
        if len(parts) < 2:
            console.print("[warning]⚠️ Missing image path. Usage: /image <path> [message][/warning]")
            return "", None
        image_path = parts[1]
        message_text = " ".join(parts[2:]) if len(parts) > 2 else "Describe this image"
        if not os.path.exists(image_path):
            console.print(f"[warning]⚠️ Image file not found: {image_path}. Sending text only.[/warning]")
            return message_text, None
        return message_text, image_path
    else:
        return user_input, None

# --- Main Function with CLI Options ---
@click.command(help="A Gemini chatbot with Rich CLI UI, live streaming, and multimodal support.")
@click.option('--model', '-m', default=DEFAULT_MODEL_ID, help='Gemini model ID.')
@click.option('--api-version', '-a', default=DEFAULT_API_VERSION, help='API version.')
@click.option('--voice', '-v', default=None, help="Specify a voice (e.g., Aoede, Charon).")
@click.option('--temperature', '-t', default=None, type=float, help='Set the generation temperature.')
@click.option('--top-p', '-p', default=None, type=float, help='Set the Top P value.')
@click.option('--top-k', '-k', default=None, type=int, help='Set the Top K value.')
@click.option('--max-output-tokens', type=int, default=None, help='Maximum number of output tokens.')
@click.option('--candidate-count', type=int, default=None, help='Number of candidate responses.')
@click.option('--presence-penalty', type=float, default=None, help='Presence penalty.')
@click.option('--frequency-penalty', type=float, default=None, help='Frequency penalty.')
def main(model, api_version, voice, temperature, top_p, top_k, max_output_tokens, 
         candidate_count, presence_penalty, frequency_penalty):
    """Main function for the Gemini chatbot."""
    client = initialize_client(api_version)

    # Enhanced Welcome Message
    console.print(Panel(
        Text(f"Welcome to Gemini Chatbot ({model})! 🌌\nType 'help' for commands.", justify="center", style="bold white"),
        title="✨ Powered by Google AI ✨",
        border_style="panel.border",
        padding=(1, 2)
    ))

    config = {"response_modalities": ["TEXT"]}

    # Optional settings for Gemini still in development
    # if voice:
    #     valid_voices = ["Aoede", "Charon", "Fenrir", "Kore", "Puck"]
    #     if voice not in valid_voices:
    #         console.print("[warning]⚠️ Invalid voice name. Ignoring voice setting.[/warning]")
    #     else:
    #         config["response_modalities"].append("SPEECH")

    async def interactive_chat():
        try:
            async with client.aio.live.connect(model=model, config=config) as session:
                while True:
                    raw_input = await asyncio.to_thread(Prompt.ask, "[user]👤 You[/user]")
                    if raw_input.lower() in ["exit", "quit", "bye"]:
                        console.print(Panel(
                            Text("Goodbye! 👋", justify="center", style="bold white"),
                            border_style="success",
                            padding=(1, 2)
                        ))
                        break
                    message_text, image_path = await process_input(raw_input)
                    if message_text == "" and image_path is None:  # Help command or empty
                        continue
                    if not await send_message(session, message_text, image_path):
                        continue
                    await receive_responses(session)
        except Exception as e:
            console.print(f"[danger]❌ Error in chat loop: {str(e)}[/danger]")

    try:
        asyncio.run(interactive_chat())
    except KeyboardInterrupt:
        console.print("\n[info]👋 Chatbot exited by user.[/info]")

if __name__ == "__main__":
    main()