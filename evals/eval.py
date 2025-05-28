#!/usr/bin/env python3
"""
llm_query_generation.py - LLM-based Cypher Query Generation Evaluation
----------------------------------------------------------------------
This script uses an LLM to generate Cypher queries based on descriptions
from the descriptions.txt file using the provided system prompt.
"""

import os
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Rich imports for beautiful console output
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TimeElapsedColumn
from rich.syntax import Syntax
from rich.padding import Padding

# Import validation utilities
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.cypher_validator import validate_query, ValidationResult

# Load environment variables
load_dotenv()

# Initialize Rich Console
console = Console()

# LLM API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

# Alternative LLM providers (uncomment and configure as needed)
# ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
# ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-sonnet-20240229")

# Rate limiting
REQUEST_DELAY_SECONDS = float(os.getenv("REQUEST_DELAY_SECONDS", "1.0"))

# File paths
DESCRIPTIONS_FILE = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'descriptions.txt')
SYSTEM_PROMPT_FILE = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'system_prompt.txt')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

class LLMClient:
    """Generic LLM client that can work with different providers."""
    
    def __init__(self, provider: str = "openai"):
        self.provider = provider.lower()
        self.client = None
        self._setup_client()
    
    def _setup_client(self):
        """Setup the appropriate LLM client based on provider."""
        if self.provider == "openai":
            try:
                import openai
                if not OPENAI_API_KEY:
                    raise ValueError("OPENAI_API_KEY environment variable is required")
                
                self.client = openai.OpenAI(
                    api_key=OPENAI_API_KEY,
                    base_url=OPENAI_BASE_URL
                )
                self.model = OPENAI_MODEL
                console.print(f"[green]✓[/green] OpenAI client initialized with model: {self.model}")
                
            except ImportError:
                console.print("[red]Error:[/red] OpenAI library not installed. Run: pip install openai")
                raise
            except Exception as e:
                console.print(f"[red]Error setting up OpenAI client:[/red] {e}")
                raise
        
        # Add other providers here as needed
        # elif self.provider == "anthropic":
        #     # Anthropic setup code
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")
    
    def generate_query(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a Cypher query using the LLM."""
        if self.provider == "openai":
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1,  # Low temperature for more consistent results
                    max_tokens=1000
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                console.print(f"[red]Error generating query:[/red] {e}")
                return f"ERROR: {str(e)}"
        
        return "ERROR: Unsupported provider"

def load_descriptions(file_path: str) -> List[str]:
    """Load descriptions from the descriptions.txt file."""
    try:
        with open(file_path, 'r') as f:
            descriptions = [line.strip() for line in f.readlines() if line.strip()]
        
        console.print(f"[green]✓[/green] Loaded {len(descriptions)} descriptions from {file_path}")
        return descriptions
        
    except FileNotFoundError:
        console.print(f"[red]Error:[/red] Descriptions file not found at {file_path}")
        return []
    except Exception as e:
        console.print(f"[red]Error loading descriptions:[/red] {e}")
        return []

def load_system_prompt(file_path: str) -> str:
    """Load the system prompt from file."""
    try:
        with open(file_path, 'r') as f:
            system_prompt = f.read().strip()
        
        console.print(f"[green]✓[/green] Loaded system prompt from {file_path}")
        return system_prompt
        
    except FileNotFoundError:
        console.print(f"[red]Error:[/red] System prompt file not found at {file_path}")
        return ""
    except Exception as e:
        console.print(f"[red]Error loading system prompt:[/red] {e}")
        return ""

def extract_query_from_response(response: str) -> str:
    """Extract the Cypher query from the LLM response."""
    # Look for query tags
    if "<query>" in response and "</query>" in response:
        start = response.find("<query>") + 7
        end = response.find("</query>")
        return response[start:end].strip()
    
    # If no tags found, return the whole response (might be just the query)
    return response.strip()

def generate_queries_from_descriptions(
    descriptions: List[str],
    system_prompt: str,
    llm_client: LLMClient,
    validate_syntax: bool = True,
    save_results: bool = True
) -> Dict[str, Any]:
    """Generate Cypher queries for all descriptions and evaluate them."""
    
    total_descriptions = len(descriptions)
    generated_queries = []
    syntax_valid_count = 0
    syntax_invalid_count = 0
    generation_errors = 0
    
    console.print(Panel(
        f"[bold cyan]LLM Query Generation Evaluation[/bold cyan]\n"
        f"Descriptions: [yellow]{total_descriptions}[/yellow]\n"
        f"LLM Provider: [green]{llm_client.provider.title()}[/green]\n"
        f"Model: [green]{llm_client.model}[/green]\n"
        f"Syntax Validation: [{'green]Enabled[/green]' if validate_syntax else 'yellow]Disabled[/yellow]'}",
        title="Generation Configuration",
        border_style="blue"
    ))
    
    overall_start_time = time.monotonic()
    
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False
    ) as progress:
        
        task = progress.add_task("Generating Queries", total=total_descriptions)
        
        for i, description in enumerate(descriptions):
            progress.update(task, description=f"Query {i+1}/{total_descriptions}")
            
            # Display current description
            console.print(Panel(
                Text(description, style="bold white"),
                title=f"Description {i+1}/{total_descriptions}",
                border_style="blue"
            ))
            
            # Generate query
            generation_start = time.monotonic()
            try:
                raw_response = llm_client.generate_query(system_prompt, description)
                generated_query = extract_query_from_response(raw_response)
                generation_time = time.monotonic() - generation_start
                
                if generated_query.startswith("ERROR:"):
                    generation_errors += 1
                    console.print(Panel(
                        Text(f"Generation failed: {generated_query}", style="bold red"),
                        title="Generation Error",
                        border_style="red"
                    ))
                    
                    query_result = {
                        "description_number": i + 1,
                        "description": description,
                        "raw_response": raw_response,
                        "generated_query": None,
                        "generation_time": generation_time,
                        "generation_error": generated_query,
                        "syntax_valid": False,
                        "syntax_errors": ["Generation failed"]
                    }
                else:
                    # Display generated query
                    query_syntax = Syntax(generated_query, "cypher", theme="monokai", line_numbers=True)
                    console.print(Panel(
                        query_syntax,
                        title="Generated Query",
                        border_style="green"
                    ))
                    
                    # Validate syntax if enabled
                    syntax_valid = True
                    syntax_errors = []
                    validation_time = 0
                    
                    if validate_syntax:
                        console.print("[dim]Validating syntax...[/dim]")
                        validation_result = validate_query(generated_query, show_progress=False)
                        validation_time = validation_result.execution_time
                        syntax_valid = validation_result.offline_ok
                        syntax_errors = validation_result.errors
                        
                        if syntax_valid:
                            syntax_valid_count += 1
                            console.print(Panel(
                                Text("✓ Syntax validation passed", style="bold green"),
                                title="Validation Result",
                                border_style="green",
                                subtitle=f"Took {validation_time:.3f}s"
                            ))
                        else:
                            syntax_invalid_count += 1
                            console.print(Panel(
                                Text(f"✗ Syntax validation failed\nErrors: {'; '.join(syntax_errors)}", style="bold red"),
                                title="Validation Result",
                                border_style="red",
                                subtitle=f"Took {validation_time:.3f}s"
                            ))
                    
                    query_result = {
                        "description_number": i + 1,
                        "description": description,
                        "raw_response": raw_response,
                        "generated_query": generated_query,
                        "generation_time": generation_time,
                        "generation_error": None,
                        "syntax_valid": syntax_valid,
                        "syntax_errors": syntax_errors,
                        "validation_time": validation_time
                    }
                
                generated_queries.append(query_result)
                
            except Exception as e:
                generation_time = time.monotonic() - generation_start
                generation_errors += 1
                
                console.print(Panel(
                    Text(f"Unexpected error during generation: {str(e)}", style="bold red"),
                    title="Generation Error",
                    border_style="red"
                ))
                
                query_result = {
                    "description_number": i + 1,
                    "description": description,
                    "raw_response": None,
                    "generated_query": None,
                    "generation_time": generation_time,
                    "generation_error": str(e),
                    "syntax_valid": False,
                    "syntax_errors": ["Generation failed with exception"]
                }
                generated_queries.append(query_result)
            
            progress.advance(task)
            console.print()  # Add spacing
            
            # Rate limiting
            time.sleep(REQUEST_DELAY_SECONDS)
    
    overall_end_time = time.monotonic()
    total_runtime = overall_end_time - overall_start_time
    
    # Calculate statistics
    successful_generations = total_descriptions - generation_errors
    
    results = {
        "evaluation_timestamp": datetime.now().isoformat(),
        "llm_provider": llm_client.provider,
        "llm_model": llm_client.model,
        "total_descriptions": total_descriptions,
        "successful_generations": successful_generations,
        "generation_errors": generation_errors,
        "syntax_validation_enabled": validate_syntax,
        "syntax_valid_queries": syntax_valid_count,
        "syntax_invalid_queries": syntax_invalid_count,
        "total_runtime": total_runtime,
        "generated_queries": generated_queries
    }
    
    # Display summary statistics
    display_summary_statistics(results)
    
    # Save results if requested
    if save_results:
        save_evaluation_results(results)
    
    return results

def display_summary_statistics(results: Dict[str, Any]):
    """Display summary statistics in a nice table."""
    stats_table = Table(
        title="LLM Query Generation Results",
        show_header=True,
        header_style="bold cyan"
    )
    stats_table.add_column("Metric", style="dim", width=40)
    stats_table.add_column("Value", justify="right")
    
    stats_table.add_row("Total Descriptions", str(results["total_descriptions"]))
    stats_table.add_row(
        Text("Successful Generations", style="green"),
        Text(str(results["successful_generations"]), style="green")
    )
    stats_table.add_row(
        Text("Generation Errors", style="red"),
        Text(str(results["generation_errors"]), style="red")
    )
    
    if results["syntax_validation_enabled"]:
        stats_table.add_row(
            Text("Syntax Valid Queries", style="green"),
            Text(str(results["syntax_valid_queries"]), style="green")
        )
        stats_table.add_row(
            Text("Syntax Invalid Queries", style="red"),
            Text(str(results["syntax_invalid_queries"]), style="red")
        )
        
        if results["successful_generations"] > 0:
            syntax_success_rate = (results["syntax_valid_queries"] / results["successful_generations"]) * 100
            stats_table.add_row("Syntax Success Rate", f"{syntax_success_rate:.1f}%")
    
    stats_table.add_row("Total Runtime", f"{results['total_runtime']:.2f} seconds")
    
    if results["successful_generations"] > 0:
        avg_time = results["total_runtime"] / results["total_descriptions"]
        stats_table.add_row("Average Time per Query", f"{avg_time:.2f} seconds")
    
    console.print(Padding(stats_table, (1, 0)))

def save_evaluation_results(results: Dict[str, Any]):
    """Save evaluation results to JSON files."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Save complete results
    results_file = os.path.join(OUTPUT_DIR, f'llm_generation_results_{timestamp}.json')
    
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        console.print(Panel(
            Text(f"Complete results saved to:\n{results_file}", style="bold cyan"),
            title="Results Saved",
            border_style="cyan"
        ))
        
        # Also save just the valid queries in the format expected by other scripts
        valid_queries = []
        for query_result in results["generated_queries"]:
            if query_result["generated_query"] and query_result["syntax_valid"]:
                valid_queries.append({
                    "description": query_result["description"],
                    "query": query_result["generated_query"],
                    "source": f"llm_generated_{results['llm_provider']}_{results['llm_model']}"
                })
        
        if valid_queries:
            queries_file = os.path.join(OUTPUT_DIR, f'llm_generated_queries_{timestamp}.json')
            with open(queries_file, 'w') as f:
                json.dump(valid_queries, f, indent=2)
            
            console.print(Panel(
                Text(f"Valid queries saved to:\n{queries_file}\n\nTotal valid queries: {len(valid_queries)}", style="bold green"),
                title="Valid Queries Saved",
                border_style="green"
            ))
        
    except Exception as e:
        console.print(Panel(
            Text(f"Failed to save results: {e}", style="bold red"),
            title="Save Error",
            border_style="red"
        ))

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate Cypher queries using LLM based on descriptions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate queries using OpenAI GPT-4
  python llm_query_generation.py
  
  # Generate without syntax validation
  python llm_query_generation.py --no-validate
  
  # Use custom files
  python llm_query_generation.py --descriptions /path/to/descriptions.txt --system-prompt /path/to/prompt.txt
  
  # Don't save results
  python llm_query_generation.py --no-save
        """
    )
    
    parser.add_argument(
        "--descriptions",
        default=DESCRIPTIONS_FILE,
        help=f"Path to descriptions file (default: {DESCRIPTIONS_FILE})"
    )
    
    parser.add_argument(
        "--system-prompt",
        default=SYSTEM_PROMPT_FILE,
        help=f"Path to system prompt file (default: {SYSTEM_PROMPT_FILE})"
    )
    
    parser.add_argument(
        "--provider",
        default="openai",
        choices=["openai"],  # Add more as implemented
        help="LLM provider to use (default: openai)"
    )
    
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip syntax validation of generated queries"
    )
    
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to files"
    )
    
    args = parser.parse_args()
    
    # Load required files
    descriptions = load_descriptions(args.descriptions)
    system_prompt = load_system_prompt(args.system_prompt)
    
    if not descriptions:
        console.print("[red]No descriptions loaded. Exiting.[/red]")
        return
    
    if not system_prompt:
        console.print("[red]No system prompt loaded. Exiting.[/red]")
        return
    
    # Initialize LLM client
    try:
        llm_client = LLMClient(provider=args.provider)
    except Exception as e:
        console.print(f"[red]Failed to initialize LLM client:[/red] {e}")
        return
    
    # Generate queries
    results = generate_queries_from_descriptions(
        descriptions=descriptions,
        system_prompt=system_prompt,
        llm_client=llm_client,
        validate_syntax=not args.no_validate,
        save_results=not args.no_save
    )
    
    # Final summary
    if results["generation_errors"] == 0 and results["syntax_valid_queries"] == results["successful_generations"]:
        console.print(Panel(
            Text("All queries generated successfully with valid syntax! 🎉", style="bold green"),
            title="Perfect Score",
            border_style="green"
        ))

if __name__ == "__main__":
    main() 