import os
import json
import time
from datetime import datetime
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
# Rich imports
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TimeElapsedColumn
from rich.syntax import Syntax
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

# Dataset configuration
MAX_RESULTS_PER_QUERY = int(os.getenv("MAX_RESULTS_PER_QUERY", "100"))  # Limit results per query
INCLUDE_EMPTY_RESULTS = os.getenv("INCLUDE_EMPTY_RESULTS", "false").lower() == "true"

# Path to the queries dataset
QUERIES_FILE_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'queries.json')

def load_queries_from_file(file_path: str) -> List[Dict[str, str]]:
    """Loads Cypher queries and their descriptions from a JSON file using Rich for output."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        # Extract query and description, ensure both exist
        queries_with_descriptions = []
        for item in data:
            if "query" in item and "description" in item:
                queries_with_descriptions.append({
                    "description": item["description"], 
                    "query": item["query"],
                    "source": item.get("source", "Unknown")
                })
            elif "query" in item: # Handle case where description might be missing
                queries_with_descriptions.append({
                    "description": "No description provided.", 
                    "query": item["query"],
                    "source": item.get("source", "Unknown")
                })
        
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

def is_404_or_empty_result(result: Any) -> bool:
    """Check if the result is a 404 error or empty result that should be filtered out."""
    if isinstance(result, dict):
        # Check for explicit error with 404
        if "error" in result and "404" in str(result["error"]):
            return True
        # Check if result is empty dict
        if not result:
            return True
    elif isinstance(result, list):
        # Check if result is empty list
        if not result:
            return True
    elif result is None:
        return True
    
    return False

def process_query_result(result: Any, max_results: int = MAX_RESULTS_PER_QUERY) -> Optional[Any]:
    """Process and potentially truncate query results."""
    if is_404_or_empty_result(result):
        return None
    
    # If result is a list and exceeds max_results, truncate it
    if isinstance(result, list) and len(result) > max_results:
        console.print(f"[yellow]Truncating result from {len(result)} to {max_results} items[/yellow]")
        return result[:max_results]
    
    return result

def sanitize_result_for_json(obj: Any) -> Any:
    """Recursively sanitize an object to ensure it's JSON serializable."""
    if isinstance(obj, dict):
        return {k: sanitize_result_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_result_for_json(item) for item in obj]
    elif isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    else:
        # Convert other types to string representation
        return str(obj)

def generate_golden_dataset():
    """
    Executes BloodHound queries and generates a golden dataset from successful results.
    Filters out 404 responses and empty results.
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

    console.print(Panel(
        f"Generating Golden Dataset from BloodHound\n"
        f"[cyan]{BHE_SCHEME}://{BHE_DOMAIN}:{BHE_PORT}[/cyan]\n"
        f"Token ID: [bold yellow]{BHE_TOKEN_ID[:8]}...[/bold yellow]\n"
        f"Pre-validation: [{'[green]Enabled[/green]' if ENABLE_PRE_VALIDATION else '[yellow]Disabled[/yellow]'}\n"
        f"Max results per query: [cyan]{MAX_RESULTS_PER_QUERY}[/cyan]\n"
        f"Include empty results: [{'[green]Yes[/green]' if INCLUDE_EMPTY_RESULTS else '[red]No[/red]'}",
        title="Golden Dataset Generation",
        expand=False
    ))

    loaded_queries = load_queries_from_file(QUERIES_FILE_PATH)

    if not loaded_queries:
        console.print("[bold red]No Cypher queries loaded. Exiting.[/bold red]")
        return

    total_queries = len(loaded_queries)
    successful_queries = 0
    failed_queries = 0
    syntax_failed_queries = 0
    filtered_queries = 0  # Queries that returned 404 or empty results
    
    golden_dataset = []
    failed_queries_data = []
    syntax_failed_queries_data = []
    
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
        task = progress.add_task("Processing Queries", total=total_queries)

        for i, query_item in enumerate(loaded_queries):
            query_description = query_item["description"]
            query_text = query_item["query"]
            query_source = query_item.get("source", "Unknown")
            
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
                
                if not validation_result.offline_ok:
                    syntax_failed_queries += 1
                    
                    # Track this syntax failure
                    syntax_failed_data = {
                        "query_number": i + 1,
                        "query_description": query_description,
                        "query_text": query_text,
                        "query_source": query_source,
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
                    console.print()
                    continue
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

                # Process the result
                processed_result = process_query_result(result)
                
                if processed_result is None:
                    # Result was filtered out (404 or empty)
                    filtered_queries += 1
                    console.print(Panel(
                        Text(f"Query result filtered out (404 or empty)", style="bold yellow"),
                        title=f"Query {i+1} Filtered",
                        border_style="yellow",
                        subtitle=f"Took {execution_time:.2f}s"
                    ))
                else:
                    # Successful query with data
                    successful_queries += 1
                    
                    # Sanitize result for JSON serialization
                    sanitized_result = sanitize_result_for_json(processed_result)
                    
                    # Add to golden dataset
                    dataset_entry = {
                        "description": query_description,
                        "query": query_text,
                        "source": query_source,
                        "result": sanitized_result,
                        "execution_time": execution_time,
                        "timestamp": datetime.now().isoformat(),
                        "result_count": len(sanitized_result) if isinstance(sanitized_result, list) else 1
                    }
                    golden_dataset.append(dataset_entry)
                    
                    result_preview = json.dumps(sanitized_result)
                    if len(result_preview) > 200:
                        result_preview = result_preview[:200] + "..."
                    
                    console.print(Panel(
                        Text(f"Successfully executed and added to dataset.\nResult preview: {result_preview}", style="bold green"),
                        title=f"Query {i+1} Success",
                        border_style="green",
                        subtitle=f"Took {execution_time:.2f}s"
                    ))
            
            except json.JSONDecodeError as e:
                query_end_time = time.monotonic()
                execution_time = query_end_time - query_start_time
                failed_queries += 1
                
                failed_queries_data.append({
                    "query_number": i + 1,
                    "query_description": query_description,
                    "query_text": query_text,
                    "query_source": query_source,
                    "error_message": f"JSONDecodeError: {e}",
                    "execution_time": execution_time,
                    "syntax_validation": "passed" if validation_result and validation_result.offline_ok else "not_checked"
                })
                
                console.print(Panel(
                    Text(f"JSONDecodeError: {e}\nThis usually means the API response was not valid JSON.", style="bold red"),
                    title=f"Query {i+1} Failed to Parse",
                    border_style="red",
                    subtitle=f"Attempted for {execution_time:.2f}s"
                ))

            except Exception as e:
                query_end_time = time.monotonic()
                execution_time = query_end_time - query_start_time
                
                # Check if the exception message contains 404
                if "404" in str(e):
                    filtered_queries += 1
                    console.print(Panel(
                        Text(f"Query returned 404 (no data). Details: {str(e)[:200]}...", style="bold yellow"),
                        title=f"Query {i+1} No Data / 404",
                        border_style="yellow",
                        subtitle=f"Took {execution_time:.2f}s"
                    ))
                else:
                    failed_queries += 1
                    
                    failed_queries_data.append({
                        "query_number": i + 1,
                        "query_description": query_description,
                        "query_text": query_text,
                        "query_source": query_source,
                        "error_message": f"Unexpected error: {str(e)}",
                        "execution_time": execution_time,
                        "syntax_validation": "passed" if validation_result and validation_result.offline_ok else "not_checked"
                    })
                    
                    console.print(Panel(
                        Text(f"An unexpected error occurred: {e}", style="bold red"),
                        title=f"Query {i+1} Unexpected Error",
                        border_style="red",
                        subtitle=f"Failed after {execution_time:.2f}s"
                    ))
            
            progress.advance(task)
            console.print()

            # Wait between queries
            time.sleep(QUERY_DELAY_SECONDS)

    overall_end_time = time.monotonic()
    total_runtime = overall_end_time - overall_start_time

    # Generate statistics
    stats_table = Table(title=Text("Golden Dataset Generation Statistics", style="bold magenta"), show_header=True, header_style="bold blue")
    stats_table.add_column("Metric", style="dim", width=40)
    stats_table.add_column("Value", justify="right")

    stats_table.add_row("Total Queries Processed", str(total_queries))
    
    if ENABLE_PRE_VALIDATION:
        stats_table.add_row(Text("Syntax Validation Failures", style="red"), Text(str(syntax_failed_queries), style="red"))
        executed_queries = total_queries - syntax_failed_queries
        stats_table.add_row("Queries Executed (passed syntax)", str(executed_queries))
    else:
        executed_queries = total_queries
    
    stats_table.add_row(Text("Successful Queries (added to dataset)", style="green"), Text(str(successful_queries), style="green"))
    stats_table.add_row(Text("Filtered Queries (404/empty)", style="yellow"), Text(str(filtered_queries), style="yellow"))
    stats_table.add_row(Text("Failed Queries", style="red"), Text(str(failed_queries), style="red"))
    stats_table.add_row("Total Runtime", f"{total_runtime:.2f} seconds")
    stats_table.add_row("Dataset Entries Created", str(len(golden_dataset)))

    console.print(Padding(stats_table, (1,0)))

    # Save the golden dataset
    if golden_dataset:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # Create organized directory structure
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
        golden_datasets_dir = os.path.join(data_dir, 'golden_datasets')
        failed_queries_dir = os.path.join(data_dir, 'failed_queries')
        syntax_failures_dir = os.path.join(data_dir, 'syntax_failures')
        
        # Ensure directories exist
        os.makedirs(golden_datasets_dir, exist_ok=True)
        os.makedirs(failed_queries_dir, exist_ok=True)
        os.makedirs(syntax_failures_dir, exist_ok=True)
        
        golden_dataset_file = os.path.join(golden_datasets_dir, f'golden_dataset_{timestamp}.json')
        
        try:
            # Create the final dataset structure
            final_dataset = {
                "metadata": {
                    "generation_timestamp": datetime.now().isoformat(),
                    "total_queries_processed": total_queries,
                    "successful_queries": successful_queries,
                    "filtered_queries": filtered_queries,
                    "failed_queries": failed_queries,
                    "syntax_failed_queries": syntax_failed_queries,
                    "total_runtime_seconds": total_runtime,
                    "bloodhound_endpoint": f"{BHE_SCHEME}://{BHE_DOMAIN}:{BHE_PORT}",
                    "max_results_per_query": MAX_RESULTS_PER_QUERY,
                    "include_empty_results": INCLUDE_EMPTY_RESULTS,
                    "pre_validation_enabled": ENABLE_PRE_VALIDATION
                },
                "dataset": golden_dataset
            }
            
            with open(golden_dataset_file, 'w') as f:
                json.dump(final_dataset, f, indent=2, default=str)
            
            console.print(Panel(
                Text(f"Golden dataset saved to:\n{golden_dataset_file}\n\nDataset entries: {len(golden_dataset)}\nTotal size: {os.path.getsize(golden_dataset_file) / 1024 / 1024:.2f} MB", style="bold green"),
                title="Golden Dataset Saved",
                border_style="green"
            ))
        except Exception as e:
            console.print(Panel(
                Text(f"Failed to save golden dataset: {e}", style="bold red"),
                title="File Save Error",
                border_style="red"
            ))

    # Save failed queries if any
    if failed_queries_data:
        failed_queries_file = os.path.join(failed_queries_dir, f'failed_queries_golden_{timestamp}.json')
        try:
            with open(failed_queries_file, 'w') as f:
                json.dump(failed_queries_data, f, indent=2)
            
            console.print(Panel(
                Text(f"Failed queries saved to:\n{failed_queries_file}\n\nTotal failed: {len(failed_queries_data)}", style="bold cyan"),
                title="Failed Queries Report",
                border_style="cyan"
            ))
        except Exception as e:
            console.print(Panel(
                Text(f"Failed to save failed queries: {e}", style="bold red"),
                title="File Save Error",
                border_style="red"
            ))

    # Save syntax validation failures if any
    if ENABLE_PRE_VALIDATION and syntax_failed_queries_data:
        syntax_failed_file = os.path.join(syntax_failures_dir, f'syntax_failed_golden_{timestamp}.json')
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

    if successful_queries == 0:
        console.print(Panel(
            Text("No successful queries found. Check your BloodHound connection and query syntax.", style="bold red"),
            title="No Data Generated",
            border_style="red"
        ))
    else:
        console.print(Panel(
            Text(f"Golden dataset generation complete! 🎉\n{successful_queries} queries successfully processed and added to dataset.", style="bold green"),
            title="Generation Complete",
            border_style="green"
        ))

if __name__ == "__main__":
    generate_golden_dataset() 