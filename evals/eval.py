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
from utils.schema_loader import load_schema, test_schema_connection
from utils.query_executor import QueryExecutor, test_connection as test_query_executor_connection
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

# Result comparison configuration
ENABLE_RESULT_COMPARISON = os.getenv("ENABLE_RESULT_COMPARISON", "true").lower() == "true"
RESULT_FUZZY_THRESHOLD = float(os.getenv("RESULT_FUZZY_THRESHOLD", "0.8"))
RESULT_COMPARISON_TIMEOUT = float(os.getenv("RESULT_COMPARISON_TIMEOUT", "30.0"))  # seconds

# Neo4j Configuration for schema loading
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "changethispassword")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
INCLUDE_SCHEMA = os.getenv("INCLUDE_SCHEMA", "true").lower() == "true"

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

def load_system_prompt(file_path: str, include_schema: bool = INCLUDE_SCHEMA) -> str:
    """Load the system prompt from file and optionally include Neo4j schema."""
    try:
        with open(file_path, 'r') as f:
            system_prompt = f.read().strip()
        
        console.print(f"[green]✓[/green] Loaded system prompt from {file_path}")
        
        # Add schema information if requested
        if include_schema:
            console.print("[dim]Attempting to load Neo4j schema...[/dim]")

            # Test connection first
            connection_success, connection_message = test_schema_connection()

            print(f"connection_success: {connection_success}")
            
            if connection_success:
                schema_text = load_schema(
                    show_progress=True
                )
                
                if schema_text and "could not be retrieved" not in schema_text and "Schema loading failed" not in schema_text:
                    # Add schema to system prompt
                    system_prompt += f"\n\n# Neo4j Database Schema\n\nThe Neo4j database has the following schema:\n\n{schema_text}\n\nUse this schema information when generating Cypher queries to ensure you reference the correct node labels, relationship types, and properties."
                    console.print("[green]✓[/green] Schema successfully included in system prompt")
                else:
                    console.print("[yellow]⚠[/yellow] Schema could not be loaded, continuing without schema")
            else:
                console.print(f"[yellow]⚠[/yellow] Neo4j connection failed: {connection_message}")
                console.print("[dim]Continuing without schema information[/dim]")
        else:
            console.print("[dim]Schema inclusion disabled[/dim]")
        
        return system_prompt
        
    except FileNotFoundError:
        console.print(f"[red]Error:[/red] System prompt file not found at {file_path}")
        return ""
    except Exception as e:
        console.print(f"[red]Error loading system prompt:[/red] {e}")
        return ""

def extract_query_from_response(response: str) -> str:
    """Extract the Cypher query from the LLM response."""
    # First, remove all content within <think> </think> tags
    cleaned_response = response
    while "<think>" in cleaned_response and "</think>" in cleaned_response:
        think_start = cleaned_response.find("<think>")
        think_end = cleaned_response.find("</think>") + 8  # +8 to include the closing tag
        cleaned_response = cleaned_response[:think_start] + cleaned_response[think_end:]
    
    # Now look for query tags in the cleaned response
    if "<query>" in cleaned_response and "</query>" in cleaned_response:
        start = cleaned_response.find("<query>") + 7
        end = cleaned_response.find("</query>")
        extracted_query = cleaned_response[start:end].strip()
        
        # Basic cleaning: remove empty lines but preserve the query structure
        lines = extracted_query.split('\n')
        clean_lines = []
        
        for line in lines:
            line = line.strip()
            # Keep non-empty lines
            if line:
                clean_lines.append(line)
        
        if clean_lines:
            return '\n'.join(clean_lines)
        else:
            return extracted_query
    
    # If no query tags found after removing think tags, try to extract from the remaining content
    lines = cleaned_response.strip().split('\n')
    clean_lines = []
    
    for line in lines:
        line = line.strip()
        # Skip empty lines and any remaining XML-like tags
        if (line and 
            not (line.startswith('<') and line.endswith('>')) and
            not line.startswith('tags.')):
            clean_lines.append(line)
    
    if clean_lines:
        return '\n'.join(clean_lines)
    else:
        return cleaned_response.strip()

def generate_queries_from_test_set(
    test_set: Dataset,
    system_prompt: str,
    llm_client: LLMClient,
    validate_syntax: bool = True,
    save_results: bool = True,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
    enable_result_comparison: bool = ENABLE_RESULT_COMPARISON,
    result_fuzzy_threshold: float = RESULT_FUZZY_THRESHOLD,
    neo4j_uri: str = NEO4J_URI,
    neo4j_user: str = NEO4J_USER,
    neo4j_password: str = NEO4J_PASSWORD,
    neo4j_database: str = NEO4J_DATABASE
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
    
    # Result comparison metrics
    result_comparison_enabled = enable_result_comparison
    result_strict_matches = 0
    result_fuzzy_matches = 0
    result_comparison_failures = 0
    result_comparison_scores = []
    query_executor = None
    
    console.print(Panel(
        f"[bold cyan]HoundBench Evaluation[/bold cyan]\n"
        f"Test Cases: [yellow]{total_test_cases}[/yellow]\n"
        f"LLM Provider: [green]{llm_client.provider.title()}[/green]\n"
        f"Model: [green]{llm_client.model}[/green]\n"
        f"Temperature: [cyan]{llm_client.temperature}[/cyan]\n"
        f"Max Tokens: [cyan]{llm_client.max_tokens}[/cyan]\n"
        f"Syntax Validation: [{'green]Enabled[/green]' if validate_syntax else 'yellow]Disabled[/yellow]'}\n"
        f"Result Comparison: [{'green]Enabled[/green]' if result_comparison_enabled else 'yellow]Disabled[/yellow]'}\n"
        f"Similarity Threshold: [cyan]{similarity_threshold}[/cyan]" +
        (f"\nResult Fuzzy Threshold: [cyan]{result_fuzzy_threshold}[/cyan]" if result_comparison_enabled else ""),
        title="Evaluation Configuration",
        border_style="blue"
    ))
    
    # Initialize query executor if result comparison is enabled
    if result_comparison_enabled:
        console.print("[dim]Initializing Neo4j query executor for result comparison...[/dim]")
        try:
            query_executor = QueryExecutor(neo4j_uri, neo4j_user, neo4j_password, neo4j_database)
            if not query_executor.connect():
                console.print("[yellow]⚠[/yellow] Failed to connect to Neo4j. Result comparison will be disabled.")
                result_comparison_enabled = False
                query_executor = None
            else:
                console.print("[green]✓[/green] Neo4j connection established for result comparison")
        except Exception as e:
            console.print(f"[yellow]⚠[/yellow] Failed to initialize query executor: {e}. Result comparison will be disabled.")
            result_comparison_enabled = False
            query_executor = None
    
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
                        "similarity_evaluation": None,
                        "result_comparison": None,
                        "result_comparison_time": 0
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
                    
                    # Result comparison evaluation (if enabled and syntax is valid)
                    result_comparison = None
                    result_comparison_time = 0
                    
                    if result_comparison_enabled and query_executor and syntax_valid:
                        console.print("[dim]Comparing query results against database...[/dim]")
                        result_comparison_start = time.monotonic()
                        
                        try:
                            # Execute ground truth query
                            gt_result = query_executor.execute_query(ground_truth_query, show_progress=False)
                            
                            # Only proceed if ground truth query returns data
                            if gt_result.success and gt_result.record_count > 0:
                                # Execute generated query
                                gen_result = query_executor.execute_query(generated_query, show_progress=False)
                                
                                # Only compare if generated query also returns data (not 404)
                                if gen_result.success and gen_result.record_count > 0:
                                    # Compare results
                                    result_comparison = query_executor.compare_results(
                                        gt_result, gen_result, fuzzy_threshold=result_fuzzy_threshold
                                    )
                                    
                                    result_comparison_time = time.monotonic() - result_comparison_start
                                    result_comparison_scores.append(result_comparison.similarity_score)
                                    
                                    # Update counters
                                    if result_comparison.strict_match:
                                        result_strict_matches += 1
                                        console.print(Panel(
                                            Text("✓ Strict result match! Generated query returns identical data.", style="bold green"),
                                            title="Result Comparison",
                                            border_style="green",
                                            subtitle=f"GT: {result_comparison.ground_truth_count} records | Gen: {result_comparison.generated_count} records | Score: {result_comparison.similarity_score:.3f} | Took {result_comparison_time:.3f}s"
                                        ))
                                    elif result_comparison.fuzzy_match:
                                        result_fuzzy_matches += 1
                                        console.print(Panel(
                                            Text(f"✓ Fuzzy result match (score: {result_comparison.similarity_score:.3f})", style="bold yellow"),
                                            title="Result Comparison",
                                            border_style="yellow",
                                            subtitle=f"GT: {result_comparison.ground_truth_count} records | Gen: {result_comparison.generated_count} records | Common: {result_comparison.common_records} | Missing: {result_comparison.missing_records} | Extra: {result_comparison.extra_records} | Took {result_comparison_time:.3f}s"
                                        ))
                                    else:
                                        console.print(Panel(
                                            Text(f"✗ Result mismatch (score: {result_comparison.similarity_score:.3f})", style="bold red"),
                                            title="Result Comparison",
                                            border_style="red",
                                            subtitle=f"GT: {result_comparison.ground_truth_count} records | Gen: {result_comparison.generated_count} records | Common: {result_comparison.common_records} | Missing: {result_comparison.missing_records} | Extra: {result_comparison.extra_records} | Took {result_comparison_time:.3f}s"
                                        ))
                                
                                elif gen_result.success and gen_result.record_count == 0:
                                    console.print(Panel(
                                        Text("Generated query returned no data (404). Skipping result comparison.", style="dim"),
                                        title="Result Comparison",
                                        border_style="dim"
                                    ))
                                    result_comparison_time = time.monotonic() - result_comparison_start
                                
                                else:
                                    result_comparison_failures += 1
                                    console.print(Panel(
                                        Text(f"Generated query failed: {gen_result.error}", style="bold red"),
                                        title="Result Comparison Error",
                                        border_style="red"
                                    ))
                                    result_comparison_time = time.monotonic() - result_comparison_start
                            
                            elif gt_result.success and gt_result.record_count == 0:
                                console.print(Panel(
                                    Text("Ground truth query returned no data. Skipping result comparison.", style="dim"),
                                    title="Result Comparison",
                                    border_style="dim"
                                ))
                                result_comparison_time = time.monotonic() - result_comparison_start
                            
                            else:
                                result_comparison_failures += 1
                                console.print(Panel(
                                    Text(f"Ground truth query failed: {gt_result.error}", style="bold red"),
                                    title="Result Comparison Error",
                                    border_style="red"
                                ))
                                result_comparison_time = time.monotonic() - result_comparison_start
                        
                        except Exception as e:
                            result_comparison_failures += 1
                            result_comparison_time = time.monotonic() - result_comparison_start
                            console.print(Panel(
                                Text(f"Result comparison failed: {str(e)}", style="bold red"),
                                title="Result Comparison Error",
                                border_style="red"
                            ))
                    
                    elif result_comparison_enabled and not syntax_valid:
                        console.print(Panel(
                            Text("Skipping result comparison due to syntax validation failure.", style="dim"),
                            title="Result Comparison",
                            border_style="dim"
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
                        "similarity_time": similarity_time,
                        "result_comparison": result_comparison,
                        "result_comparison_time": result_comparison_time
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
                    "similarity_evaluation": None,
                    "result_comparison": None,
                    "result_comparison_time": 0
                }
                generated_queries.append(query_result)
            
            progress.advance(task)
            console.print()  # Add spacing
            
            # Rate limiting
            time.sleep(REQUEST_DELAY_SECONDS)
    
    overall_end_time = time.monotonic()
    total_runtime = overall_end_time - overall_start_time
    
    # Close query executor if it was used
    if query_executor:
        query_executor.close()
    
    # Calculate statistics
    successful_generations = total_test_cases - generation_errors
    avg_similarity_score = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0.0
    avg_result_comparison_score = sum(result_comparison_scores) / len(result_comparison_scores) if result_comparison_scores else 0.0
    
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
        "result_comparison_enabled": result_comparison_enabled,
        "result_fuzzy_threshold": result_fuzzy_threshold,
        "result_strict_matches": result_strict_matches,
        "result_fuzzy_matches": result_fuzzy_matches,
        "result_comparison_failures": result_comparison_failures,
        "average_result_comparison_score": avg_result_comparison_score,
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
        title=f"HoundBench Results - {results['llm_provider'].title()} ({results['llm_model']})",
        show_header=True,
        header_style="bold cyan"
    )
    stats_table.add_column("Metric", style="dim", width=40)
    stats_table.add_column("Value", justify="right")
    
    # Add model information at the top
    stats_table.add_row("LLM Provider", results["llm_provider"].title())
    stats_table.add_row("Model", results["llm_model"])
    stats_table.add_row("", "")  # Separator
    
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
    
    # Add result comparison statistics if enabled
    if results.get("result_comparison_enabled", False):
        stats_table.add_row("", "")  # Separator
        stats_table.add_row(
            Text("Result Strict Matches", style="green"),
            Text(str(results.get("result_strict_matches", 0)), style="green")
        )
        stats_table.add_row(
            Text("Result Fuzzy Matches", style="yellow"),
            Text(str(results.get("result_fuzzy_matches", 0)), style="yellow")
        )
        stats_table.add_row(
            Text("Result Comparison Failures", style="red"),
            Text(str(results.get("result_comparison_failures", 0)), style="red")
        )
        
        if results.get("result_strict_matches", 0) + results.get("result_fuzzy_matches", 0) > 0:
            result_accuracy_rate = ((results.get("result_strict_matches", 0) + results.get("result_fuzzy_matches", 0)) / 
                                  results["total_test_cases"]) * 100
            stats_table.add_row("Result Accuracy Rate (Strict + Fuzzy)", f"{result_accuracy_rate:.1f}%")
        
        avg_result_score = results.get("average_result_comparison_score", 0.0)
        if avg_result_score > 0:
            stats_table.add_row("Average Result Comparison Score", f"{avg_result_score:.3f}")
    
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
        
        # Save stats summary for analysis
        save_stats_summary(results)
        
    except Exception as e:
        console.print(Panel(
            Text(f"Failed to save results: {e}", style="bold red"),
            title="Save Error",
            border_style="red"
        ))

def save_stats_summary(results: Dict[str, Any]):
    """Save a summary of evaluation statistics to a separate file for analysis."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Calculate derived metrics
    syntax_success_rate = 0.0
    if results["successful_generations"] > 0:
        syntax_success_rate = (results["syntax_valid_queries"] / results["successful_generations"]) * 100
    
    accuracy_rate = 0.0
    if results["total_test_cases"] > 0:
        accuracy_rate = ((results["exact_matches"] + results["high_similarity_count"]) / 
                        results["total_test_cases"]) * 100
    
    avg_time_per_case = 0.0
    if results["total_test_cases"] > 0:
        avg_time_per_case = results["total_runtime"] / results["total_test_cases"]
    
    # Create summary stats
    stats_summary = {
        "evaluation_timestamp": results["evaluation_timestamp"],
        "llm_provider": results["llm_provider"],
        "llm_model": results["llm_model"],
        "total_test_cases": results["total_test_cases"],
        "successful_generations": results["successful_generations"],
        "generation_errors": results["generation_errors"],
        "syntax_validation_enabled": results["syntax_validation_enabled"],
        "syntax_valid_queries": results["syntax_valid_queries"],
        "syntax_invalid_queries": results["syntax_invalid_queries"],
        "syntax_success_rate_percent": round(syntax_success_rate, 2),
        "similarity_threshold": results["similarity_threshold"],
        "exact_matches": results["exact_matches"],
        "high_similarity_count": results["high_similarity_count"],
        "medium_similarity_count": results["medium_similarity_count"],
        "low_similarity_count": results["low_similarity_count"],
        "accuracy_rate_percent": round(accuracy_rate, 2),
        "average_similarity_score": round(results["average_similarity_score"], 4),
        "total_runtime_seconds": round(results["total_runtime"], 2),
        "average_time_per_case_seconds": round(avg_time_per_case, 2)
    }
    
    # Save stats summary
    stats_file = os.path.join(OUTPUT_DIR, f'eval_stats_{results["llm_provider"]}_{results["llm_model"].replace("/", "_")}_{timestamp}.json')
    
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        with open(stats_file, 'w') as f:
            json.dump(stats_summary, f, indent=2)
        
        console.print(Panel(
            Text(f"Evaluation stats saved to:\n{stats_file}", style="bold magenta"),
            title="Stats Summary Saved",
            border_style="magenta"
        ))
        
        return stats_file
        
    except Exception as e:
        console.print(Panel(
            Text(f"Failed to save stats summary: {e}", style="bold red"),
            title="Stats Save Error",
            border_style="red"
        ))
        return None

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
  
  # Neo4j Schema Configuration:
  # Disable schema loading (use static prompt only)
  python eval.py --no-schema
  
  # Use custom Neo4j connection
  python eval.py --neo4j-uri bolt://myserver:7687 --neo4j-user admin --neo4j-password mypass
  
  # Use different Neo4j database
  python eval.py --neo4j-database bloodhound
  
  # Result Comparison Configuration:
  # Disable result comparison evaluation
  python eval.py --no-result-comparison
  
  # Use custom fuzzy threshold for result matching
  python eval.py --result-fuzzy-threshold 0.9
  
  # Full evaluation with result comparison
  python eval.py --neo4j-uri bolt://localhost:7687 --neo4j-user neo4j --neo4j-password mypass
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
    
    # Neo4j Schema Configuration
    parser.add_argument(
        "--neo4j-uri",
        default=NEO4J_URI,
        help=f"Neo4j connection URI (default: {NEO4J_URI})"
    )
    
    parser.add_argument(
        "--neo4j-user",
        default=NEO4J_USER,
        help=f"Neo4j username (default: {NEO4J_USER})"
    )
    
    parser.add_argument(
        "--neo4j-password",
        default=NEO4J_PASSWORD,
        help="Neo4j password (default: from NEO4J_SECRET env var)"
    )
    
    parser.add_argument(
        "--neo4j-database",
        default=NEO4J_DATABASE,
        help=f"Neo4j database name (default: {NEO4J_DATABASE})"
    )
    
    parser.add_argument(
        "--no-schema",
        action="store_true",
        help="Disable automatic schema loading and inclusion in system prompt"
    )
    
    # Result Comparison Configuration
    parser.add_argument(
        "--no-result-comparison",
        action="store_true",
        help="Disable result comparison evaluation against the database"
    )
    
    parser.add_argument(
        "--result-fuzzy-threshold",
        type=float,
        default=RESULT_FUZZY_THRESHOLD,
        help=f"Threshold for fuzzy result matching (default: {RESULT_FUZZY_THRESHOLD})"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not 0 < args.test_ratio < 1:
        console.print("[red]Error: test-ratio must be between 0 and 1[/red]")
        return
    
    if not 0 <= args.similarity_threshold <= 1:
        console.print("[red]Error: similarity-threshold must be between 0 and 1[/red]")
        return
    
    if not 0 <= args.result_fuzzy_threshold <= 1:
        console.print("[red]Error: result-fuzzy-threshold must be between 0 and 1[/red]")
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
    include_schema = not args.no_schema
    
    system_prompt = load_system_prompt(
        args.system_prompt, 
        include_schema=include_schema
    )
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
        similarity_threshold=args.similarity_threshold,
        enable_result_comparison=not args.no_result_comparison,
        result_fuzzy_threshold=args.result_fuzzy_threshold,
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
        neo4j_database=args.neo4j_database
    )
    
    # Final summary
    model_info = f"{results['llm_provider'].title()} {results['llm_model']}"
    
    if results["generation_errors"] == 0 and results["syntax_valid_queries"] == results["successful_generations"]:
        if results["exact_matches"] == results["total_test_cases"]:
            console.print(Panel(
                Text(f"Perfect evaluation with {model_info}! All queries generated successfully with exact matches! 🎉", style="bold green"),
                title="Perfect Score",
                border_style="green"
            ))
        elif (results["exact_matches"] + results["high_similarity_count"]) == results["total_test_cases"]:
            console.print(Panel(
                Text(f"Excellent evaluation with {model_info}! All queries generated with high similarity! 🎯", style="bold yellow"),
                title="Excellent Score",
                border_style="yellow"
            ))
        else:
            accuracy_rate = ((results["exact_matches"] + results["high_similarity_count"]) / 
                           results["total_test_cases"]) * 100
            console.print(Panel(
                Text(f"All queries generated with valid syntax using {model_info}! Accuracy: {accuracy_rate:.1f}% 📊", style="bold cyan"),
                title="Good Syntax, Mixed Accuracy",
                border_style="cyan"
            ))
    else:
        accuracy_rate = ((results["exact_matches"] + results["high_similarity_count"]) / 
                       max(results["total_test_cases"], 1)) * 100
        console.print(Panel(
            Text(f"Evaluation complete with {model_info}! Accuracy: {accuracy_rate:.1f}% | Avg Similarity: {results['average_similarity_score']:.3f} 📊", style="bold cyan"),
            title="Evaluation Summary",
            border_style="cyan"
        ))

if __name__ == "__main__":
    main() 