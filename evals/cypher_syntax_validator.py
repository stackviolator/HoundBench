#!/usr/bin/env python3
"""
cypher_syntax_validator.py – Standalone Cypher syntax validation tool
--------------------------------------------------------------------
Validates Cypher queries from the queries.json dataset using ANTLR
for fast offline syntax checking with beautiful Rich console output.
"""

import os
import json
import argparse
import time
from datetime import datetime
from typing import List, Dict, Any

# Rich imports
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TimeElapsedColumn
from rich.syntax import Syntax
from rich.padding import Padding

# Import our validation utilities
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.cypher_validator import validate_query, get_validation_summary, ValidationResult

# Initialize Rich Console
console = Console()

# Default paths
DEFAULT_QUERIES_FILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'queries.json')
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

def load_queries_from_file(file_path: str) -> List[Dict[str, Any]]:
    """Load Cypher queries from JSON file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Ensure each query has required fields
        queries_with_descriptions = []
        for i, item in enumerate(data):
            if "query" in item:
                query_dict = {
                    "query": item["query"],
                    "description": item.get("description", f"Query {i+1}"),
                    "source": item.get("source", "Unknown")
                }
                queries_with_descriptions.append(query_dict)
        
        if not queries_with_descriptions:
            console.print(f"[bold yellow]Warning:[/bold yellow] No queries found in {file_path}.")
        
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

def validate_queries_dataset(queries_file: str = DEFAULT_QUERIES_FILE, 
                           show_individual_results: bool = False,
                           save_results: bool = True) -> None:
    """
    Validate all queries in the dataset and display results.
    
    Args:
        queries_file: Path to the queries JSON file
        show_individual_results: Whether to show detailed results for each query
        save_results: Whether to save failed queries to a file
    """
    console.print(Panel(
        f"[bold cyan]Cypher Syntax Validator[/bold cyan]\n"
        f"Validating queries from: [yellow]{queries_file}[/yellow]\n"
        f"Validation mode: [green]Offline (ANTLR)[/green]",
        title="Syntax Validation",
        border_style="blue"
    ))
    
    # Load queries
    loaded_queries = load_queries_from_file(queries_file)
    
    if not loaded_queries:
        console.print("[bold red]No queries to validate. Exiting.[/bold red]")
        return
    
    total_queries = len(loaded_queries)
    validation_results: List[ValidationResult] = []
    failed_queries_data: List[Dict[str, Any]] = []
    
    overall_start_time = time.monotonic()
    
    # Progress bar setup
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False
    ) as progress:
        
        task = progress.add_task("Validating Syntax", total=total_queries)
        
        for i, query_item in enumerate(loaded_queries):
            query_description = query_item["description"]
            query_text = query_item["query"]
            query_source = query_item.get("source", "Unknown")
            
            progress.update(task, description=f"Query {i+1}/{total_queries}")
            
            # Show individual query details if requested
            if show_individual_results:
                query_panel_title = f"Query {i+1}/{total_queries} - {query_description}"
                query_syntax = Syntax(query_text, "cypher", theme="monokai", line_numbers=True)
                console.print(Panel(query_syntax, title=query_panel_title, border_style="blue", expand=False))
            
            # Validate query (offline only)
            result = validate_query(query_text, show_progress=show_individual_results)
            validation_results.append(result)
            
            # Track failed queries
            if not result.offline_ok:
                failed_query_data = {
                    "query_number": i + 1,
                    "description": query_description,
                    "source": query_source,
                    "query": query_text,
                    "syntax_errors": result.errors,
                    "validation_time": result.execution_time
                }
                failed_queries_data.append(failed_query_data)
                
                if show_individual_results:
                    console.print(Panel(
                        Text(f"Syntax validation failed:\n" + "\n".join(result.errors), style="bold red"),
                        title=f"Query {i+1} Failed",
                        border_style="red",
                        subtitle=f"Took {result.execution_time:.3f}s"
                    ))
            elif show_individual_results:
                console.print(Panel(
                    Text("Syntax validation passed!", style="bold green"),
                    title=f"Query {i+1} Succeeded",
                    border_style="green",
                    subtitle=f"Took {result.execution_time:.3f}s"
                ))
            
            progress.advance(task)
            
            if show_individual_results:
                console.print()  # Add spacing
    
    overall_end_time = time.monotonic()
    total_runtime = overall_end_time - overall_start_time
    
    # Display summary statistics
    summary_table = get_validation_summary(validation_results)
    console.print(Padding(summary_table, (1, 0)))
    
    # Additional timing statistics
    timing_table = Table(title="Performance Statistics", show_header=True, header_style="bold cyan")
    timing_table.add_column("Metric", style="dim", width=30)
    timing_table.add_column("Value", justify="right")
    
    timing_table.add_row("Total Runtime", f"{total_runtime:.2f} seconds")
    timing_table.add_row("Queries per Second", f"{total_queries/total_runtime:.1f}")
    
    if validation_results:
        execution_times = [r.execution_time for r in validation_results]
        timing_table.add_row("Fastest Query", f"{min(execution_times):.3f} seconds")
        timing_table.add_row("Slowest Query", f"{max(execution_times):.3f} seconds")
    
    console.print(Padding(timing_table, (1, 0)))
    
    # Save failed queries if requested and there are failures
    if save_results and failed_queries_data:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        failed_queries_file = os.path.join(DEFAULT_OUTPUT_DIR, f'syntax_validation_failures_{timestamp}.json')
        
        try:
            os.makedirs(os.path.dirname(failed_queries_file), exist_ok=True)
            
            with open(failed_queries_file, 'w') as f:
                json.dump({
                    "validation_timestamp": timestamp,
                    "total_queries": total_queries,
                    "failed_queries_count": len(failed_queries_data),
                    "validation_type": "syntax_only",
                    "failed_queries": failed_queries_data
                }, f, indent=2)
            
            console.print(Panel(
                Text(f"Syntax validation failures saved to:\n{failed_queries_file}\n\n"
                     f"Failed queries: {len(failed_queries_data)}/{total_queries}", style="bold cyan"),
                title="Validation Report Saved",
                border_style="cyan"
            ))
            
        except Exception as e:
            console.print(Panel(
                Text(f"Failed to save validation report: {e}", style="bold red"),
                title="File Save Error",
                border_style="red"
            ))
    
    elif not failed_queries_data:
        console.print(Panel(
            Text("All queries passed syntax validation! 🎉", style="bold green"),
            title="Perfect Syntax Score",
            border_style="green"
        ))

def validate_single_query(query: str, description: str = "Single Query") -> None:
    """Validate a single query and display results."""
    console.print(Panel(
        f"[bold cyan]Single Query Validation[/bold cyan]\n"
        f"Description: [yellow]{description}[/yellow]",
        title="Syntax Validation",
        border_style="blue"
    ))
    
    # Display the query
    query_syntax = Syntax(query, "cypher", theme="monokai", line_numbers=True)
    console.print(Panel(query_syntax, title="Query to Validate", border_style="blue"))
    
    # Validate
    result = validate_query(query, show_progress=True)
    
    # Show final result
    if result.offline_ok:
        console.print(Panel(
            Text("✓ Query syntax is valid!", style="bold green"),
            title="Validation Complete",
            border_style="green"
        ))
    else:
        console.print(Panel(
            Text("✗ Query syntax is invalid!", style="bold red"),
            title="Validation Complete",
            border_style="red"
        ))

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Validate Cypher query syntax using ANTLR parser",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate all queries in the dataset
  python cypher_syntax_validator.py
  
  # Validate with detailed output for each query
  python cypher_syntax_validator.py --verbose
  
  # Validate a custom queries file
  python cypher_syntax_validator.py --file /path/to/queries.json
  
  # Validate a single query
  python cypher_syntax_validator.py --query "MATCH (n) RETURN n"
  
  # Validate without saving results
  python cypher_syntax_validator.py --no-save
        """
    )
    
    parser.add_argument(
        "--file", "-f",
        default=DEFAULT_QUERIES_FILE,
        help=f"Path to queries JSON file (default: {DEFAULT_QUERIES_FILE})"
    )
    
    parser.add_argument(
        "--query", "-q",
        help="Validate a single query string instead of a file"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed validation results for each query"
    )
    
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save failed queries to a file"
    )
    
    args = parser.parse_args()
    
    if args.query:
        # Single query validation
        validate_single_query(args.query)
    else:
        # Dataset validation
        validate_queries_dataset(
            queries_file=args.file,
            show_individual_results=args.verbose,
            save_results=not args.no_save
        )

if __name__ == "__main__":
    main() 