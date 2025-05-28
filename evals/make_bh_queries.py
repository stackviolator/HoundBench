import os
import json
import time
from datetime import datetime
from dotenv import load_dotenv
# Rich imports
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TimeElapsedColumn
from rich.syntax import Syntax
from rich.live import Live
from rich.padding import Padding

# Assuming apiclient.py is in the utils directory, and utils is in the parent directory of evals
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.apiclient import Client, Credentials
from utils.cypher_validator import validate_query, ValidationResult

# Load environment variables from .env file
load_dotenv()

# Initialize Rich Console
console = Console()

# API Configuration - Ensure these are set in your .env file
BHE_DOMAIN = os.getenv("BHE_DOMAIN")
BHE_PORT = int(os.getenv("BHE_PORT", "443"))
BHE_SCHEME = os.getenv("BHE_SCHEME", "https")
BHE_TOKEN_ID = os.getenv("BHE_TOKEN_ID")
BHE_TOKEN_KEY = os.getenv("BHE_TOKEN_KEY")

# Rate limiting configuration
QUERY_DELAY_SECONDS = float(os.getenv("QUERY_DELAY_SECONDS", "0.5"))  # Default 500ms delay between queries

# Validation configuration
ENABLE_PRE_VALIDATION = os.getenv("ENABLE_PRE_VALIDATION", "true").lower() == "true"

# Path to the queries dataset
QUERIES_FILE_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'queries.json')

def load_queries_from_file(file_path: str) -> list[dict]:
    """Loads Cypher queries and their descriptions from a JSON file using Rich for output."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        # Extract query and description, ensure both exist
        queries_with_descriptions = []
        for item in data:
            if "query" in item and "description" in item:
                queries_with_descriptions.append({"description": item["description"], "query": item["query"]})
            elif "query" in item: # Handle case where description might be missing
                queries_with_descriptions.append({"description": "No description provided.", "query": item["query"]})
        
        if not queries_with_descriptions:
            console.print(f"[bold yellow]Warning:[/bold yellow] No queries (or queries with descriptions) found in {file_path}.")
        return queries_with_descriptions
    except FileNotFoundError:
        console.print(f"[bold red]Error:[/bold red] Queries file not found at {file_path}")
        return []
    except json.JSONDecodeError:
        console.print(f"[bold red]Error:[/bold red] Could not decode JSON from {file_path}")
        return []
    except Exception as e:
        console.print(f"[bold red]An unexpected error occurred while loading queries:[/bold red] {e}")
        return []

def run_and_evaluate_queries():
    """
    Initializes the API client, runs a set of Cypher queries, and prints evaluation statistics using Rich.
    Now includes optional pre-validation using ANTLR syntax checking.
    """
    if not all([BHE_DOMAIN, BHE_TOKEN_ID, BHE_TOKEN_KEY]):
        console.print("[bold red]Error:[/bold red] Missing one or more environment variables (BHE_DOMAIN, BHE_TOKEN_ID, BHE_TOKEN_KEY).")
        console.print("Please ensure they are set in your .env file or environment.")
        return

    try:
        credentials = Credentials(token_id=BHE_TOKEN_ID, token_key=BHE_TOKEN_KEY)
        client = Client(scheme=BHE_SCHEME, host=BHE_DOMAIN, port=BHE_PORT, credentials=credentials)
    except ValueError as e:
        console.print(Panel(f"[bold red]Error initializing client:[/bold red] {e}", title="Client Initialization Failed", border_style="red"))
        return
    except Exception as e:
        console.print(Panel(f"[bold red]An unexpected error occurred during client initialization:[/bold red] {e}", title="Client Initialization Error", border_style="red"))
        return

    console.print(Panel(f"Attempting to connect to BloodHound at [cyan]{BHE_SCHEME}://{BHE_DOMAIN}:{BHE_PORT}[/cyan]\nUsing Token ID: [bold yellow]{BHE_TOKEN_ID[:8]}...[/bold yellow]\nPre-validation: [{'[green]Enabled[/green]' if ENABLE_PRE_VALIDATION else '[yellow]Disabled[/yellow]'}", title="Connection Details", expand=False))

    loaded_queries = load_queries_from_file(QUERIES_FILE_PATH)

    if not loaded_queries:
        console.print("[bold red]No Cypher queries loaded. Exiting.[/bold red]")
        return

    total_queries = len(loaded_queries)
    successful_queries = 0
    failed_queries = 0
    syntax_failed_queries = 0
    failed_queries_data = []  # Track failed queries and their errors
    syntax_failed_queries_data = []  # Track syntax validation failures
    query_execution_times = []
    validation_times = []
    
    overall_start_time = time.monotonic()

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False # Keep progress bar after completion
    ) as progress:
        task = progress.add_task("Evaluating Queries", total=total_queries)

        for i, query_item in enumerate(loaded_queries):
            query_description = query_item["description"]
            query_text = query_item["query"]
            
            progress.update(task, description=f"Query {i+1}/{total_queries}")
            
            query_panel_title = f"Query {i+1}/{total_queries} - {query_description}"
            query_syntax = Syntax(query_text, "cypher", theme="monokai", line_numbers=True)
            console.print(Panel(query_syntax, title=query_panel_title, border_style="blue", expand=False))

            # Pre-validation step (if enabled)
            validation_result = None
            if ENABLE_PRE_VALIDATION:
                console.print("[dim]Running syntax pre-validation...[/dim]")
                validation_start = time.monotonic()
                validation_result = validate_query(query_text, show_progress=False)
                validation_end = time.monotonic()
                validation_time = validation_end - validation_start
                validation_times.append(validation_time)
                
                if not validation_result.offline_ok:
                    syntax_failed_queries += 1
                    
                    # Track this syntax failure
                    syntax_failed_data = {
                        "query_number": i + 1,
                        "query_description": query_description,
                        "query_text": query_text,
                        "syntax_errors": validation_result.errors,
                        "validation_time": validation_time
                    }
                    syntax_failed_queries_data.append(syntax_failed_data)
                    
                    console.print(Panel(
                        Text(f"Syntax validation failed. Skipping execution.\nErrors: {'; '.join(validation_result.errors)}", style="bold red"),
                        title=f"Query {i+1} Syntax Error",
                        border_style="red",
                        subtitle=f"Validation took {validation_time:.3f}s"
                    ))
                    
                    progress.advance(task)
                    console.print() # Add a little space
                    continue  # Skip execution for syntax-invalid queries
                else:
                    console.print(Panel(
                        Text(f"Syntax validation passed ✓", style="bold green"),
                        title="Pre-validation",
                        border_style="green",
                        subtitle=f"Took {validation_time:.3f}s"
                    ))

            query_start_time = time.monotonic()
            
            try:
                result = client.run_cypher(query_text, include_properties=True)
                query_end_time = time.monotonic()
                execution_time = query_end_time - query_start_time
                query_execution_times.append(execution_time)

                if "error" in result:
                    error_content = result["error"]
                    # Check if the string representation of the error content contains "404"
                    if "404" in str(error_content):
                        successful_queries += 1
                        console.print(Panel(
                            Text(f"Query completed, no data returned (interpreted as 404). Details: {str(error_content)[:200]}...", style="bold yellow"), # Truncate long error messages
                            title=f"Query {i+1} No Data / 404",
                            border_style="yellow",
                            subtitle=f"Took {execution_time:.2f}s"
                        ))
                    else:  # Actual failure
                        failed_queries += 1
                        error_message = f"Error: {error_content}"
                        if "details" in result: # 'details' might co-exist with 'error'
                            error_message += f"\nDetails: {result['details']}"
                        
                        # Track this failed query
                        failed_queries_data.append({
                            "query_number": i + 1,
                            "query_description": query_description,
                            "query_text": query_text,
                            "error_message": error_message,
                            "execution_time": execution_time,
                            "syntax_validation": "passed" if validation_result and validation_result.offline_ok else "not_checked"
                        })
                        
                        console.print(Panel(
                            Text(error_message, style="bold red"),
                            title=f"Query {i+1} Failed",
                            border_style="red",
                            subtitle=f"Took {execution_time:.2f}s"
                        ))
                else:
                    successful_queries += 1
                    # Check if result is empty or contains no data
                    if not result or (isinstance(result, dict) and not result) or (isinstance(result, list) and not result):
                        console.print(Panel(
                            Text("Query completed successfully but returned no data.", style="bold yellow"),
                            title=f"Query {i+1} Succeeded (No Data)",
                            border_style="yellow",
                            subtitle=f"Took {execution_time:.2f}s"
                        ))
                    else:
                        response_preview = json.dumps(result)
                        if len(response_preview) > 200:
                            response_preview = response_preview[:200] + "..."
                        console.print(Panel(
                            Text(f"Successfully executed.\nResponse preview: {response_preview}", style="bold green"),
                            title=f"Query {i+1} Succeeded",
                            border_style="green",
                            subtitle=f"Took {execution_time:.2f}s"
                        ))
            
            except json.JSONDecodeError as e: # Catching the specific error you encountered
                query_end_time = time.monotonic()
                execution_time = query_end_time - query_start_time
                query_execution_times.append(execution_time)
                failed_queries += 1
                
                # Track this failed query
                failed_queries_data.append({
                    "query_number": i + 1,
                    "query_description": query_description,
                    "query_text": query_text,
                    "error_message": f"JSONDecodeError: {e}",
                    "execution_time": execution_time,
                    "syntax_validation": "passed" if validation_result and validation_result.offline_ok else "not_checked"
                })
                
                console.print(Panel(Text(f"JSONDecodeError: {e}\nThis usually means the API response was not valid JSON. Check debug output from apiclient.py.", style="bold red"), title=f"Query {i+1} Failed to Parse", border_style="red", subtitle=f"Attempted for {execution_time:.2f}s"))

            except Exception as e:
                query_end_time = time.monotonic()
                execution_time = query_end_time - query_start_time
                query_execution_times.append(execution_time)
                # Check if the exception message contains 404
                if "404" in str(e):
                    successful_queries += 1
                    console.print(Panel(
                        Text(f"Query completed, no data returned (interpreted as 404). Details: {str(e)[:200]}...", style="bold yellow"),
                        title=f"Query {i+1} No Data / 404",
                        border_style="yellow",
                        subtitle=f"Took {execution_time:.2f}s"
                    ))
                else:
                    failed_queries += 1
                    
                    # Track this failed query
                    failed_queries_data.append({
                        "query_number": i + 1,
                        "query_description": query_description,
                        "query_text": query_text,
                        "error_message": f"Unexpected error: {str(e)}",
                        "execution_time": execution_time,
                        "syntax_validation": "passed" if validation_result and validation_result.offline_ok else "not_checked"
                    })
                    
                    console.print(Panel(Text(f"An unexpected error occurred: {e}", style="bold red"), title=f"Query {i+1} Unexpected Error", border_style="red", subtitle=f"Failed after {execution_time:.2f}s"))
            
            progress.advance(task)
            console.print() # Add a little space

            # Wait between queries
            time.sleep(QUERY_DELAY_SECONDS)

    overall_end_time = time.monotonic()
    total_runtime = overall_end_time - overall_start_time

    # --- Enhanced Statistics Table ---
    stats_table = Table(title=Text("Cypher Query Evaluation Statistics", style="bold magenta"), show_header=True, header_style="bold blue")
    stats_table.add_column("Metric", style="dim", width=40)
    stats_table.add_column("Value", justify="right")

    stats_table.add_row("Total Queries Attempted", str(total_queries))
    
    if ENABLE_PRE_VALIDATION:
        stats_table.add_row(Text("Syntax Validation Failures", style="red"), Text(str(syntax_failed_queries), style="red"))
        executed_queries = total_queries - syntax_failed_queries
        stats_table.add_row("Queries Executed (passed syntax)", str(executed_queries))
    else:
        executed_queries = total_queries
    
    stats_table.add_row(Text("Successful Executions", style="green"), Text(str(successful_queries), style="green"))
    stats_table.add_row(Text("Failed Executions", style="red"), Text(str(failed_queries), style="red"))
    stats_table.add_row("Total Execution Time", f"{total_runtime:.2f} seconds")

    if query_execution_times:
        average_time = sum(query_execution_times) / len(query_execution_times) if query_execution_times else 0
        min_time = min(query_execution_times) if query_execution_times else 0
        max_time = max(query_execution_times) if query_execution_times else 0
        stats_table.add_row("Average Execution Time per Query", f"{average_time:.2f} seconds")
        stats_table.add_row("Fastest Query Execution Time", f"{min_time:.2f} seconds")
        stats_table.add_row("Slowest Query Execution Time", f"{max_time:.2f} seconds")
    
    if ENABLE_PRE_VALIDATION and validation_times:
        avg_validation_time = sum(validation_times) / len(validation_times)
        stats_table.add_row("Average Validation Time", f"{avg_validation_time:.3f} seconds")
    
    console.print(Padding(stats_table, (1,0)))

    # Save failed queries and their error messages to a file
    if failed_queries_data:
        failed_queries_file = os.path.join(os.path.dirname(__file__), '..', 'data', f'failed_queries_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.json')
        try:
            # Ensure the data directory exists
            os.makedirs(os.path.dirname(failed_queries_file), exist_ok=True)
            
            with open(failed_queries_file, 'w') as f:
                json.dump(failed_queries_data, f, indent=2)
            
            console.print(Panel(
                Text(f"Failed execution queries saved to:\n{failed_queries_file}\n\nTotal failed executions: {len(failed_queries_data)}", style="bold cyan"),
                title="Failed Execution Queries Report",
                border_style="cyan"
            ))
        except Exception as e:
            console.print(Panel(
                Text(f"Failed to save failed queries to file: {e}", style="bold red"),
                title="File Save Error",
                border_style="red"
            ))

    # Save syntax validation failures if any
    if ENABLE_PRE_VALIDATION and syntax_failed_queries_data:
        syntax_failed_file = os.path.join(os.path.dirname(__file__), '..', 'data', f'syntax_failed_queries_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.json')
        try:
            with open(syntax_failed_file, 'w') as f:
                json.dump({
                    "validation_timestamp": datetime.now().isoformat(),
                    "total_queries": total_queries,
                    "syntax_failed_count": len(syntax_failed_queries_data),
                    "validation_type": "pre_execution_syntax_check",
                    "syntax_failed_queries": syntax_failed_queries_data
                }, f, indent=2)
            
            console.print(Panel(
                Text(f"Syntax validation failures saved to:\n{syntax_failed_file}\n\nTotal syntax failures: {len(syntax_failed_queries_data)}", style="bold yellow"),
                title="Syntax Validation Failures Report",
                border_style="yellow"
            ))
        except Exception as e:
            console.print(Panel(
                Text(f"Failed to save syntax validation failures: {e}", style="bold red"),
                title="File Save Error",
                border_style="red"
            ))

    if not failed_queries_data and not syntax_failed_queries_data:
        console.print(Panel(
            Text("All queries passed validation and execution! 🎉", style="bold green"),
            title="Perfect Score",
            border_style="green"
        ))

if __name__ == "__main__":
    run_and_evaluate_queries() 