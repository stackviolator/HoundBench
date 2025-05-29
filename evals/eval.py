#!/usr/bin/env python3

import os
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv

# Rich imports for beautiful console output
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TimeElapsedColumn
from rich.syntax import Syntax
from rich.padding import Padding

from phoenix.otel import register

# Import validation utilities
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.cypher_validator import validate_query, ValidationResult
from utils.dataset_utils import (
    load_queries_dataset, 
    split_dataset, 
    calculate_query_similarity,
    sample_dataset,
    filter_dataset_by_source
)
from datasets import Dataset

# Load environment variables
load_dotenv()

# Initialize Rich Console
console = Console()

# LLM API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

# LLM Hyperparameters
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "5000"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.6"))

# Rate limiting
REQUEST_DELAY_SECONDS = float(os.getenv("REQUEST_DELAY_SECONDS", "1.0"))

# Query similarity configuration
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.8"))
EXACT_MATCH_WEIGHT = float(os.getenv("EXACT_MATCH_WEIGHT", "0.4"))
TOKEN_SIMILARITY_WEIGHT = float(os.getenv("TOKEN_SIMILARITY_WEIGHT", "0.3"))
STRUCTURAL_SIMILARITY_WEIGHT = float(os.getenv("STRUCTURAL_SIMILARITY_WEIGHT", "0.3"))

# File paths
SYSTEM_PROMPT_FILE = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'system_prompt.txt')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
QUERIES_DATASET_FILE = os.path.join(OUTPUT_DIR, 'queries.json')

class LLMClient:
    """Generic LLM client that can work with different providers."""
    
    def __init__(self, provider: str = "openai", temperature: float = None, max_tokens: int = None):
        self.provider = provider.lower()
        self.client = None
        self.temperature = temperature if temperature is not None else LLM_TEMPERATURE
        self.max_tokens = max_tokens if max_tokens is not None else LLM_MAX_TOKENS
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
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                console.print(f"[red]Error generating query:[/red] {e}")
                return f"ERROR: {str(e)}"
        
        return "ERROR: Unsupported provider"

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
        extracted_query = response[start:end].strip()
        
        # Additional cleaning: remove any remaining XML-like tags or thinking content
        lines = extracted_query.split('\n')
        clean_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip empty lines and lines that look like XML tags or thinking content
            if (line and 
                not line.startswith('<') and 
                not line.endswith('>') and
                not line.startswith('tags.') and
                not line.startswith('</think>')):
                clean_lines.append(line)
        
        if clean_lines:
            return '\n'.join(clean_lines)
        else:
            # If no clean lines found, return the original extracted content
            return extracted_query
    
    # If no tags found, return the whole response (might be just the query)
    # But clean it up first
    lines = response.strip().split('\n')
    clean_lines = []
    
    for line in lines:
        line = line.strip()
        # Skip lines that look like XML tags or thinking content
        if (line and 
            not line.startswith('<') and 
            not line.endswith('>') and
            not line.startswith('tags.') and
            not line.startswith('</think>')):
            clean_lines.append(line)
    
    if clean_lines:
        return '\n'.join(clean_lines)
    else:
        return response.strip()

def generate_queries_from_test_set(
    test_set: Dataset,
    system_prompt: str,
    llm_client: LLMClient,
    validate_syntax: bool = True,
    save_results: bool = True,
    similarity_threshold: float = SIMILARITY_THRESHOLD
) -> Dict[str, Any]:
    """Generate Cypher queries for test set descriptions and evaluate against ground truth."""
    
    total_test_cases = len(test_set)
    generated_queries = []
    syntax_valid_count = 0
    syntax_invalid_count = 0
    generation_errors = 0
    
    # Similarity metrics
    exact_matches = 0
    high_similarity_count = 0
    medium_similarity_count = 0
    low_similarity_count = 0
    similarity_scores = []
    
    console.print(Panel(
        f"[bold cyan]LLM Query Generation Evaluation[/bold cyan]\n"
        f"Test Cases: [yellow]{total_test_cases}[/yellow]\n"
        f"LLM Provider: [green]{llm_client.provider.title()}[/green]\n"
        f"Model: [green]{llm_client.model}[/green]\n"
        f"Temperature: [cyan]{llm_client.temperature}[/cyan]\n"
        f"Max Tokens: [cyan]{llm_client.max_tokens}[/cyan]\n"
        f"Syntax Validation: [{'green]Enabled[/green]' if validate_syntax else 'yellow]Disabled[/yellow]'}\n"
        f"Similarity Threshold: [cyan]{similarity_threshold}[/cyan]",
        title="Evaluation Configuration",
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
        
        task = progress.add_task("Evaluating Test Cases", total=total_test_cases)
        
        for i in range(total_test_cases):
            progress.update(task, description=f"Test Case {i+1}/{total_test_cases}")
            
            test_case = test_set[i]
            description = test_case["description"]
            ground_truth_query = test_case["query"]
            
            # Display current test case
            console.print(Panel(
                Text(description, style="bold white"),
                title=f"Test Case {i+1}/{total_test_cases}",
                border_style="blue"
            ))
            
            # Display ground truth query
            ground_truth_syntax = Syntax(ground_truth_query, "cypher", theme="monokai", line_numbers=True)
            console.print(Panel(
                ground_truth_syntax,
                title="Ground Truth Query",
                border_style="yellow"
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
                        "test_case_number": i + 1,
                        "description": description,
                        "ground_truth_query": ground_truth_query,
                        "raw_response": raw_response,
                        "generated_query": None,
                        "generation_time": generation_time,
                        "generation_error": generated_query,
                        "syntax_valid": False,
                        "syntax_errors": ["Generation failed"],
                        "similarity_evaluation": None
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
                    
                    # Calculate query similarity
                    console.print("[dim]Calculating query similarity...[/dim]")
                    similarity_start = time.monotonic()
                    similarity_result = calculate_query_similarity(
                        generated_query, 
                        ground_truth_query,
                        exact_match_weight=EXACT_MATCH_WEIGHT,
                        token_similarity_weight=TOKEN_SIMILARITY_WEIGHT,
                        structural_similarity_weight=STRUCTURAL_SIMILARITY_WEIGHT
                    )
                    similarity_time = time.monotonic() - similarity_start
                    
                    # Categorize similarity
                    overall_score = similarity_result["overall_score"]
                    similarity_scores.append(overall_score)
                    
                    if similarity_result["exact_match"]:
                        exact_matches += 1
                        console.print(Panel(
                            Text("✓ Exact match with ground truth!", style="bold green"),
                            title="Similarity Evaluation",
                            border_style="green",
                            subtitle=f"Score: {overall_score:.3f} | Took {similarity_time:.3f}s"
                        ))
                    elif overall_score >= similarity_threshold:
                        high_similarity_count += 1
                        console.print(Panel(
                            Text(f"✓ High similarity (score: {overall_score:.3f})", style="bold yellow"),
                            title="Similarity Evaluation",
                            border_style="yellow",
                            subtitle=f"Token: {similarity_result['token_similarity']:.3f} | Structural: {similarity_result['structural_similarity']:.3f} | Took {similarity_time:.3f}s"
                        ))
                    elif overall_score >= 0.5:
                        medium_similarity_count += 1
                        console.print(Panel(
                            Text(f"~ Medium similarity (score: {overall_score:.3f})", style="bold blue"),
                            title="Similarity Evaluation",
                            border_style="blue",
                            subtitle=f"Token: {similarity_result['token_similarity']:.3f} | Structural: {similarity_result['structural_similarity']:.3f} | Took {similarity_time:.3f}s"
                        ))
                    else:
                        low_similarity_count += 1
                        console.print(Panel(
                            Text(f"✗ Low similarity (score: {overall_score:.3f})", style="bold red"),
                            title="Similarity Evaluation",
                            border_style="red",
                            subtitle=f"Token: {similarity_result['token_similarity']:.3f} | Structural: {similarity_result['structural_similarity']:.3f} | Took {similarity_time:.3f}s"
                        ))
                    
                    query_result = {
                        "test_case_number": i + 1,
                        "description": description,
                        "ground_truth_query": ground_truth_query,
                        "raw_response": raw_response,
                        "generated_query": generated_query,
                        "generation_time": generation_time,
                        "generation_error": None,
                        "syntax_valid": syntax_valid,
                        "syntax_errors": syntax_errors,
                        "validation_time": validation_time,
                        "similarity_evaluation": similarity_result,
                        "similarity_time": similarity_time
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
                    "test_case_number": i + 1,
                    "description": description,
                    "ground_truth_query": ground_truth_query,
                    "raw_response": None,
                    "generated_query": None,
                    "generation_time": generation_time,
                    "generation_error": str(e),
                    "syntax_valid": False,
                    "syntax_errors": ["Generation failed with exception"],
                    "similarity_evaluation": None
                }
                generated_queries.append(query_result)
            
            progress.advance(task)
            console.print()  # Add spacing
            
            # Rate limiting
            time.sleep(REQUEST_DELAY_SECONDS)
    
    overall_end_time = time.monotonic()
    total_runtime = overall_end_time - overall_start_time
    
    # Calculate statistics
    successful_generations = total_test_cases - generation_errors
    avg_similarity_score = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0.0
    
    results = {
        "evaluation_timestamp": datetime.now().isoformat(),
        "llm_provider": llm_client.provider,
        "llm_model": llm_client.model,
        "total_test_cases": total_test_cases,
        "successful_generations": successful_generations,
        "generation_errors": generation_errors,
        "syntax_validation_enabled": validate_syntax,
        "syntax_valid_queries": syntax_valid_count,
        "syntax_invalid_queries": syntax_invalid_count,
        "similarity_threshold": similarity_threshold,
        "exact_matches": exact_matches,
        "high_similarity_count": high_similarity_count,
        "medium_similarity_count": medium_similarity_count,
        "low_similarity_count": low_similarity_count,
        "average_similarity_score": avg_similarity_score,
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
    
    stats_table.add_row("Total Test Cases", str(results["total_test_cases"]))
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
    
    # Add similarity statistics
    stats_table.add_row("", "")  # Separator
    stats_table.add_row(
        Text("Exact Matches", style="green"),
        Text(str(results["exact_matches"]), style="green")
    )
    stats_table.add_row(
        Text(f"High Similarity (≥{results['similarity_threshold']:.1f})", style="yellow"),
        Text(str(results["high_similarity_count"]), style="yellow")
    )
    stats_table.add_row(
        Text("Medium Similarity (0.5-0.8)", style="blue"),
        Text(str(results["medium_similarity_count"]), style="blue")
    )
    stats_table.add_row(
        Text("Low Similarity (<0.5)", style="red"),
        Text(str(results["low_similarity_count"]), style="red")
    )
    
    if results["total_test_cases"] > 0:
        accuracy_rate = ((results["exact_matches"] + results["high_similarity_count"]) / 
                        results["total_test_cases"]) * 100
        stats_table.add_row("Accuracy Rate (Exact + High)", f"{accuracy_rate:.1f}%")
        stats_table.add_row("Average Similarity Score", f"{results['average_similarity_score']:.3f}")
    
    stats_table.add_row("Total Runtime", f"{results['total_runtime']:.2f} seconds")
    
    if results["successful_generations"] > 0:
        avg_time = results["total_runtime"] / results["total_test_cases"]
        stats_table.add_row("Average Time per Test Case", f"{avg_time:.2f} seconds")
    
    console.print(Padding(stats_table, (1, 0)))

def save_evaluation_results(results: Dict[str, Any]):
    """Save evaluation results to JSON files."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Save complete results
    results_file = os.path.join(OUTPUT_DIR, f'llm_evaluation_results_{timestamp}.json')
    
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        console.print(Panel(
            Text(f"Complete results saved to:\n{results_file}", style="bold cyan"),
            title="Results Saved",
            border_style="cyan"
        ))
        
        # Also save just the valid queries in a simplified format
        valid_queries = []
        for query_result in results["generated_queries"]:
            if query_result["generated_query"] and query_result["syntax_valid"]:
                valid_queries.append({
                    "description": query_result["description"],
                    "generated_query": query_result["generated_query"],
                    "ground_truth_query": query_result["ground_truth_query"],
                    "similarity_score": query_result.get("similarity_evaluation", {}).get("overall_score", 0.0),
                    "source": f"eval_{results['llm_provider']}_{results['llm_model']}"
                })
        
        if valid_queries:
            queries_file = os.path.join(OUTPUT_DIR, f'eval_queries_{timestamp}.json')
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
    tracer_provider = register(
        project_name="HoundBench",
        auto_instrument=True,
        set_global_tracer_provider=False,
        # batch=True
    )

    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate LLM-generated Cypher queries against ground truth using train/test split",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation with default local dataset (uses entire dataset for testing)
  python eval.py
  
  # Use a different local JSON file
  python eval.py --dataset /path/to/custom_queries.json
  
  # Load from Hugging Face Hub (uses entire dataset for testing)
  python eval.py --dataset "username/cypher-queries-dataset"
  
  # Load specific split from HF dataset
  python eval.py --dataset "username/dataset" --dataset-split "test"
  
  # Load with dataset configuration
  python eval.py --dataset "username/dataset" --dataset-config "bloodhound"
  
  # Filter by sources
  python eval.py --filter-sources "hausec.com" "redfoxsec.com"
  
  # Sample 50 random examples for quick testing
  python eval.py --sample-size 50
  
  # Split dataset into train/test (85/15 by default)
  python eval.py --split-dataset
  
  # Custom train/test split ratio (requires --split-dataset)
  python eval.py --split-dataset --test-ratio 0.2
  
  # Generate without syntax validation
  python eval.py --no-validate
  
  # Use custom system prompt
  python eval.py --system-prompt /path/to/prompt.txt
  
  # Adjust LLM hyperparameters
  python eval.py --temperature 0.3 --max-tokens 2000
  
  # Custom similarity threshold
  python eval.py --similarity-threshold 0.9
  
  # Don't save results
  python eval.py --no-save
        """
    )
    
    parser.add_argument(
        "--dataset",
        default=QUERIES_DATASET_FILE,
        help=f"Path to queries dataset file, HF dataset name, or 'local' for local JSON (default: {QUERIES_DATASET_FILE})"
    )
    
    parser.add_argument(
        "--dataset-split",
        default=None,
        help="Dataset split to use (e.g., 'train', 'test', 'validation'). Only for HF datasets."
    )
    
    parser.add_argument(
        "--dataset-config",
        default=None,
        help="Dataset configuration name for HF datasets"
    )
    
    parser.add_argument(
        "--filter-sources",
        nargs='+',
        default=None,
        help="Filter dataset to only include entries from specific sources"
    )
    
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Sample N random examples from the dataset for faster testing"
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
        "--test-ratio",
        type=float,
        default=0.15,
        help="Ratio of dataset to use for testing (default: 0.15)"
    )
    
    parser.add_argument(
        "--split-dataset",
        action="store_true",
        help="Split the dataset into train/test sets. If not specified, uses entire dataset for testing."
    )
    
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for dataset splitting (default: 42)"
    )
    
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=SIMILARITY_THRESHOLD,
        help=f"Threshold for high similarity classification (default: {SIMILARITY_THRESHOLD})"
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
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=LLM_TEMPERATURE,
        help=f"LLM temperature for generation (default: {LLM_TEMPERATURE})"
    )
    
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=LLM_MAX_TOKENS,
        help=f"Maximum tokens for LLM generation (default: {LLM_MAX_TOKENS})"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not 0 < args.test_ratio < 1:
        console.print("[red]Error: test-ratio must be between 0 and 1[/red]")
        return
    
    if not 0 <= args.similarity_threshold <= 1:
        console.print("[red]Error: similarity-threshold must be between 0 and 1[/red]")
        return
    
    if args.test_ratio != 0.15 and not args.split_dataset:
        console.print("[yellow]Warning: test-ratio specified but --split-dataset not used. test-ratio will be ignored.[/yellow]")
    
    # Load dataset
    console.print(Panel(
        Text("Loading and preparing dataset...", style="bold cyan"),
        title="Dataset Preparation",
        border_style="blue"
    ))
    
    try:
        # Prepare dataset loading arguments
        if args.dataset.endswith('.json') or args.dataset == 'local':
            # Load from local JSON file
            dataset_path = QUERIES_DATASET_FILE if args.dataset == 'local' else args.dataset
            dataset = load_queries_dataset(dataset_path)
        else:
            # Load from Hugging Face Hub
            load_kwargs = {'dataset_path': args.dataset}
            if args.dataset_split:
                load_kwargs['split'] = args.dataset_split
            if args.dataset_config:
                load_kwargs['dataset_path'] = {
                    'path': args.dataset,
                    'name': args.dataset_config
                }
            dataset = load_queries_dataset(**load_kwargs)
        
        # Apply filters if specified
        if args.filter_sources:
            dataset = filter_dataset_by_source(dataset, args.filter_sources)
        
        # Sample dataset if specified
        if args.sample_size:
            dataset = sample_dataset(dataset, args.sample_size, random_seed=args.random_seed)
        
        if len(dataset) == 0:
            console.print("[red]No data in dataset after filtering/sampling. Exiting.[/red]")
            return
            
    except Exception as e:
        console.print(f"[red]Failed to load dataset:[/red] {e}")
        return
    
    # Split dataset
    if args.split_dataset:
        train_ratio = 1.0 - args.test_ratio
        train_set, test_set = split_dataset(dataset, train_ratio=train_ratio, random_seed=args.random_seed)
    else:
        train_set = dataset
        test_set = dataset
    
    if len(test_set) == 0:
        console.print("[red]No test cases available. Exiting.[/red]")
        return
    
    # Load system prompt
    system_prompt = load_system_prompt(args.system_prompt)
    if not system_prompt:
        console.print("[red]No system prompt loaded. Exiting.[/red]")
        return
    
    # Initialize LLM client
    try:
        llm_client = LLMClient(provider=args.provider, temperature=args.temperature, max_tokens=args.max_tokens)
    except Exception as e:
        console.print(f"[red]Failed to initialize LLM client:[/red] {e}")
        return
    
    # Display dataset information
    console.print(Panel(
        f"[bold cyan]Dataset Information[/bold cyan]\n"
        f"Total entries: [yellow]{len(dataset)}[/yellow]\n" +
        (f"Train set: [green]{len(train_set)}[/green] entries ({len(train_set)/len(dataset)*100:.1f}%)\n"
         f"Test set: [blue]{len(test_set)}[/blue] entries ({len(test_set)/len(dataset)*100:.1f}%)\n"
         f"Random seed: [dim]{args.random_seed}[/dim]" if args.split_dataset else
         f"Using entire dataset for testing: [blue]{len(test_set)}[/blue] entries\n"
         f"[dim]No train/test split applied[/dim]"),
        title="Dataset Configuration",
        border_style="green"
    ))
    
    # Run evaluation on test set
    results = generate_queries_from_test_set(
        test_set=test_set,
        system_prompt=system_prompt,
        llm_client=llm_client,
        validate_syntax=not args.no_validate,
        save_results=not args.no_save,
        similarity_threshold=args.similarity_threshold
    )
    
    # Final summary
    if results["generation_errors"] == 0 and results["syntax_valid_queries"] == results["successful_generations"]:
        if results["exact_matches"] == results["total_test_cases"]:
            console.print(Panel(
                Text("Perfect evaluation! All queries generated successfully with exact matches! 🎉", style="bold green"),
                title="Perfect Score",
                border_style="green"
            ))
        elif (results["exact_matches"] + results["high_similarity_count"]) == results["total_test_cases"]:
            console.print(Panel(
                Text("Excellent evaluation! All queries generated with high similarity! 🎯", style="bold yellow"),
                title="Excellent Score",
                border_style="yellow"
            ))
        else:
            accuracy_rate = ((results["exact_matches"] + results["high_similarity_count"]) / 
                           results["total_test_cases"]) * 100
            console.print(Panel(
                Text(f"All queries generated with valid syntax! Accuracy: {accuracy_rate:.1f}% 📊", style="bold cyan"),
                title="Good Syntax, Mixed Accuracy",
                border_style="cyan"
            ))
    else:
        accuracy_rate = ((results["exact_matches"] + results["high_similarity_count"]) / 
                       max(results["total_test_cases"], 1)) * 100
        console.print(Panel(
            Text(f"Evaluation complete! Accuracy: {accuracy_rate:.1f}% | Avg Similarity: {results['average_similarity_score']:.3f} 📊", style="bold cyan"),
            title="Evaluation Summary",
            border_style="cyan"
        ))

if __name__ == "__main__":
    main() 