#!/usr/bin/env python3

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

from phoenix.otel import register

# Import validation utilities
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.cypher_validator import validate_query, ValidationResult
from utils.apiclient import Client, Credentials

# Load environment variables
load_dotenv()

# Initialize Rich Console
console = Console()

# LLM API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

# BloodHound API Configuration for semantic evaluation
BHE_DOMAIN = os.getenv("BHE_DOMAIN")
BHE_PORT = int(os.getenv("BHE_PORT", "443"))
BHE_SCHEME = os.getenv("BHE_SCHEME", "https")
BHE_TOKEN_ID = os.getenv("BHE_TOKEN_ID")
BHE_TOKEN_KEY = os.getenv("BHE_TOKEN_KEY")

# Alternative LLM providers (uncomment and configure as needed)
# ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
# ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-sonnet-20240229")

# Rate limiting
REQUEST_DELAY_SECONDS = float(os.getenv("REQUEST_DELAY_SECONDS", "1.0"))
SEMANTIC_QUERY_DELAY_SECONDS = float(os.getenv("SEMANTIC_QUERY_DELAY_SECONDS", "0.5"))

# Semantic evaluation configuration
SEMANTIC_COMPARISON_TOLERANCE = float(os.getenv("SEMANTIC_COMPARISON_TOLERANCE", "0.01"))
MAX_RESULTS_FOR_COMPARISON = int(os.getenv("MAX_RESULTS_FOR_COMPARISON", "100"))

# File paths
DESCRIPTIONS_FILE = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'descriptions.txt')
SYSTEM_PROMPT_FILE = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'system_prompt.txt')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
GOLDEN_DATASETS_DIR = os.path.join(OUTPUT_DIR, 'golden_datasets')

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

def load_golden_dataset(file_path: str) -> Optional[Dict[str, Any]]:
    """Load a golden dataset from JSON file."""
    try:
        with open(file_path, 'r') as f:
            dataset = json.load(f)
        
        console.print(f"[green]✓[/green] Loaded golden dataset from {file_path}")
        
        if "dataset" in dataset:
            # New format with metadata
            console.print(f"[dim]Dataset contains {len(dataset['dataset'])} entries[/dim]")
            return dataset
        else:
            # Legacy format - assume it's just the dataset array
            console.print(f"[dim]Legacy format dataset with {len(dataset)} entries[/dim]")
            return {"dataset": dataset, "metadata": {}}
            
    except FileNotFoundError:
        console.print(f"[red]Error:[/red] Golden dataset file not found at {file_path}")
        return None
    except json.JSONDecodeError:
        console.print(f"[red]Error:[/red] Could not decode JSON from {file_path}")
        return None
    except Exception as e:
        console.print(f"[red]Error loading golden dataset:[/red] {e}")
        return None

def find_golden_dataset_files(directory: str = GOLDEN_DATASETS_DIR) -> List[str]:
    """Find all golden dataset files in the specified directory."""
    if not os.path.exists(directory):
        return []
    
    dataset_files = []
    for filename in os.listdir(directory):
        if filename.startswith('golden_dataset_') and filename.endswith('.json'):
            dataset_files.append(os.path.join(directory, filename))
    
    return sorted(dataset_files, reverse=True)  # Most recent first

def normalize_result_for_comparison(result: Any) -> Any:
    """Normalize a result for comparison by sorting lists and handling special cases."""
    if isinstance(result, list):
        # Sort list items for consistent comparison
        try:
            # Try to sort if items are comparable
            if all(isinstance(item, (str, int, float)) for item in result):
                return sorted(result)
            elif all(isinstance(item, dict) for item in result):
                # Sort dicts by their string representation for consistency
                return sorted(result, key=lambda x: json.dumps(x, sort_keys=True))
            else:
                # Mixed types - convert to strings and sort
                return sorted([str(item) for item in result])
        except (TypeError, ValueError):
            # If sorting fails, return as-is
            return result
    elif isinstance(result, dict):
        # Recursively normalize dict values and sort keys
        return {k: normalize_result_for_comparison(v) for k, v in sorted(result.items())}
    else:
        return result

def compare_results(result1: Any, result2: Any, tolerance: float = SEMANTIC_COMPARISON_TOLERANCE) -> Dict[str, Any]:
    """
    Compare two query results and return comparison metrics.
    
    Returns:
        Dict with keys: exact_match, similarity_score, differences
    """
    # Normalize both results
    norm_result1 = normalize_result_for_comparison(result1)
    norm_result2 = normalize_result_for_comparison(result2)
    
    # Check for exact match
    exact_match = norm_result1 == norm_result2
    
    if exact_match:
        return {
            "exact_match": True,
            "similarity_score": 1.0,
            "differences": [],
            "comparison_type": "exact"
        }
    
    # For lists, calculate overlap
    if isinstance(norm_result1, list) and isinstance(norm_result2, list):
        set1 = set(json.dumps(item, sort_keys=True) if isinstance(item, dict) else str(item) for item in norm_result1)
        set2 = set(json.dumps(item, sort_keys=True) if isinstance(item, dict) else str(item) for item in norm_result2)
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        similarity_score = intersection / union if union > 0 else 0.0
        
        differences = {
            "only_in_result1": list(set1 - set2),
            "only_in_result2": list(set2 - set1),
            "common_items": list(set1 & set2)
        }
        
        return {
            "exact_match": False,
            "similarity_score": similarity_score,
            "differences": differences,
            "comparison_type": "set_overlap"
        }
    
    # For other types, just check if they're close enough
    try:
        if isinstance(norm_result1, (int, float)) and isinstance(norm_result2, (int, float)):
            diff = abs(norm_result1 - norm_result2)
            max_val = max(abs(norm_result1), abs(norm_result2), 1)  # Avoid division by zero
            similarity_score = 1.0 - (diff / max_val)
            
            return {
                "exact_match": False,
                "similarity_score": max(0.0, similarity_score),
                "differences": {"numeric_difference": diff},
                "comparison_type": "numeric"
            }
    except (TypeError, ValueError):
        pass
    
    # Default case - just string comparison
    str1, str2 = str(norm_result1), str(norm_result2)
    similarity_score = 1.0 if str1 == str2 else 0.0
    
    return {
        "exact_match": False,
        "similarity_score": similarity_score,
        "differences": {"string_diff": f"'{str1}' vs '{str2}'"},
        "comparison_type": "string"
    }

def find_matching_golden_entry(description: str, golden_dataset: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Find a matching entry in the golden dataset based on description."""
    dataset_entries = golden_dataset.get("dataset", [])
    
    # First try exact description match
    for entry in dataset_entries:
        if entry.get("description", "").strip().lower() == description.strip().lower():
            return entry
    
    # If no exact match, try partial matching
    description_lower = description.strip().lower()
    for entry in dataset_entries:
        entry_desc = entry.get("description", "").strip().lower()
        if description_lower in entry_desc or entry_desc in description_lower:
            return entry
    
    return None

def perform_semantic_evaluation(
    description: str, 
    generated_query: str, 
    golden_dataset: Dict[str, Any], 
    bloodhound_client: Client
) -> Optional[Dict[str, Any]]:
    """
    Perform semantic evaluation by executing the generated query and comparing with golden dataset.
    
    Returns:
        Dict with evaluation results or None if evaluation failed
    """
    start_time = time.monotonic()
    
    try:
        # Find matching golden entry
        golden_entry = find_matching_golden_entry(description, golden_dataset)
        if not golden_entry:
            return {
                "found_golden_match": False,
                "execution_time": time.monotonic() - start_time,
                "error": "No matching golden dataset entry found"
            }
        
        # Execute the generated query
        try:
            generated_result = bloodhound_client.run_cypher(generated_query, include_properties=True)
            
            # Truncate result if too large
            if isinstance(generated_result, list) and len(generated_result) > MAX_RESULTS_FOR_COMPARISON:
                generated_result = generated_result[:MAX_RESULTS_FOR_COMPARISON]
                
        except Exception as e:
            return {
                "found_golden_match": True,
                "golden_entry_description": golden_entry.get("description"),
                "execution_time": time.monotonic() - start_time,
                "execution_error": str(e),
                "exact_match": False,
                "similarity_score": 0.0
            }
        
        # Compare results
        golden_result = golden_entry.get("result")
        comparison = compare_results(generated_result, golden_result)
        
        # Add rate limiting
        time.sleep(SEMANTIC_QUERY_DELAY_SECONDS)
        
        return {
            "found_golden_match": True,
            "golden_entry_description": golden_entry.get("description"),
            "golden_query": golden_entry.get("query"),
            "execution_time": time.monotonic() - start_time,
            "execution_error": None,
            "exact_match": comparison["exact_match"],
            "similarity_score": comparison["similarity_score"],
            "comparison_details": comparison,
            "generated_result_count": len(generated_result) if isinstance(generated_result, list) else 1,
            "golden_result_count": len(golden_result) if isinstance(golden_result, list) else 1
        }
        
    except Exception as e:
        return {
            "found_golden_match": False,
            "execution_time": time.monotonic() - start_time,
            "error": f"Semantic evaluation failed: {str(e)}"
        }

def generate_queries_from_descriptions(
    descriptions: List[str],
    system_prompt: str,
    llm_client: LLMClient,
    validate_syntax: bool = True,
    save_results: bool = True,
    enable_semantic_eval: bool = False,
    golden_dataset: Optional[Dict[str, Any]] = None,
    bloodhound_client: Optional[Client] = None
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
        f"Syntax Validation: [{'green]Enabled[/green]' if validate_syntax else 'yellow]Disabled[/yellow]'}\n"
        f"Semantic Evaluation: [{'green]Enabled[/green]' if enable_semantic_eval else 'yellow]Disabled[/yellow]'}"
        + (f"\nGolden Dataset: [cyan]{len(golden_dataset.get('dataset', []))} entries[/cyan]" if golden_dataset else ""),
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
                        "syntax_errors": ["Generation failed"],
                        "semantic_evaluation": None
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
                    
                    # Semantic evaluation if enabled
                    semantic_eval_result = None
                    if enable_semantic_eval and syntax_valid and golden_dataset and bloodhound_client:
                        console.print("[dim]Running semantic evaluation...[/dim]")
                        semantic_eval_result = perform_semantic_evaluation(
                            description, generated_query, golden_dataset, bloodhound_client
                        )
                        
                        if semantic_eval_result:
                            if semantic_eval_result["exact_match"]:
                                console.print(Panel(
                                    Text("✓ Semantic evaluation: Exact match with golden dataset", style="bold green"),
                                    title="Semantic Evaluation",
                                    border_style="green",
                                    subtitle=f"Took {semantic_eval_result['execution_time']:.3f}s"
                                ))
                            elif semantic_eval_result["similarity_score"] > 0.8:
                                console.print(Panel(
                                    Text(f"✓ Semantic evaluation: High similarity ({semantic_eval_result['similarity_score']:.2f})", style="bold yellow"),
                                    title="Semantic Evaluation",
                                    border_style="yellow",
                                    subtitle=f"Took {semantic_eval_result['execution_time']:.3f}s"
                                ))
                            else:
                                console.print(Panel(
                                    Text(f"✗ Semantic evaluation: Low similarity ({semantic_eval_result['similarity_score']:.2f})", style="bold red"),
                                    title="Semantic Evaluation",
                                    border_style="red",
                                    subtitle=f"Took {semantic_eval_result['execution_time']:.3f}s"
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
                        "validation_time": validation_time,
                        "semantic_evaluation": semantic_eval_result
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
                    "syntax_errors": ["Generation failed with exception"],
                    "semantic_evaluation": None
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
    
    # Calculate semantic evaluation statistics
    semantic_eval_count = 0
    semantic_exact_matches = 0
    semantic_high_similarity = 0
    semantic_low_similarity = 0
    semantic_errors = 0
    avg_similarity_score = 0.0
    
    if enable_semantic_eval:
        semantic_scores = []
        for query_result in generated_queries:
            semantic_eval = query_result.get("semantic_evaluation")
            if semantic_eval:
                semantic_eval_count += 1
                if semantic_eval.get("exact_match"):
                    semantic_exact_matches += 1
                elif semantic_eval.get("similarity_score", 0) > 0.8:
                    semantic_high_similarity += 1
                elif semantic_eval.get("similarity_score", 0) > 0.0:
                    semantic_low_similarity += 1
                else:
                    semantic_errors += 1
                
                if "similarity_score" in semantic_eval:
                    semantic_scores.append(semantic_eval["similarity_score"])
        
        if semantic_scores:
            avg_similarity_score = sum(semantic_scores) / len(semantic_scores)
    
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
        "semantic_evaluation_enabled": enable_semantic_eval,
        "semantic_evaluations_performed": semantic_eval_count,
        "semantic_exact_matches": semantic_exact_matches,
        "semantic_high_similarity": semantic_high_similarity,
        "semantic_low_similarity": semantic_low_similarity,
        "semantic_errors": semantic_errors,
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
    
    # Add semantic evaluation statistics
    if results.get("semantic_evaluation_enabled"):
        stats_table.add_row("", "")  # Separator
        stats_table.add_row(
            Text("Semantic Evaluations Performed", style="cyan"),
            Text(str(results["semantic_evaluations_performed"]), style="cyan")
        )
        stats_table.add_row(
            Text("Exact Matches", style="green"),
            Text(str(results["semantic_exact_matches"]), style="green")
        )
        stats_table.add_row(
            Text("High Similarity (>80%)", style="yellow"),
            Text(str(results["semantic_high_similarity"]), style="yellow")
        )
        stats_table.add_row(
            Text("Low Similarity", style="red"),
            Text(str(results["semantic_low_similarity"]), style="red")
        )
        stats_table.add_row(
            Text("Semantic Errors", style="red"),
            Text(str(results["semantic_errors"]), style="red")
        )
        
        if results["semantic_evaluations_performed"] > 0:
            semantic_accuracy = ((results["semantic_exact_matches"] + results["semantic_high_similarity"]) / 
                               results["semantic_evaluations_performed"]) * 100
            stats_table.add_row("Semantic Accuracy Rate", f"{semantic_accuracy:.1f}%")
            stats_table.add_row("Average Similarity Score", f"{results['average_similarity_score']:.3f}")
    
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
    tracer_provider = register(
        project_name="HoundBench",
        auto_instrument=True,
        set_global_tracer_provider=False,
        # batch=True
    )

    import argparse

    parser = argparse.ArgumentParser(
        description="Generate Cypher queries using LLM based on descriptions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate queries using OpenAI GPT-4
  python eval.py
  
  # Generate without syntax validation
  python eval.py --no-validate
  
  # Enable semantic evaluation with golden dataset
  python eval.py --semantic-eval --golden-dataset /path/to/golden_dataset.json
  
  # Use custom files
  python eval.py --descriptions /path/to/descriptions.txt --system-prompt /path/to/prompt.txt
  
  # Don't save results
  python eval.py --no-save
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
    
    parser.add_argument(
        "--semantic-eval",
        action="store_true",
        help="Enable semantic evaluation against golden dataset"
    )
    
    parser.add_argument(
        "--golden-dataset",
        help="Path to golden dataset file (if not specified, will try to find latest)"
    )
    
    parser.add_argument(
        "--comparison-tolerance",
        type=float,
        default=SEMANTIC_COMPARISON_TOLERANCE,
        help=f"Tolerance for numerical comparisons (default: {SEMANTIC_COMPARISON_TOLERANCE})"
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
    
    # Handle semantic evaluation setup
    golden_dataset = None
    bloodhound_client = None
    
    if args.semantic_eval:
        # Load golden dataset
        if args.golden_dataset:
            golden_dataset = load_golden_dataset(args.golden_dataset)
        else:
            # Try to find the latest golden dataset
            dataset_files = find_golden_dataset_files()
            if dataset_files:
                console.print(f"[yellow]No golden dataset specified, using latest: {os.path.basename(dataset_files[0])}[/yellow]")
                golden_dataset = load_golden_dataset(dataset_files[0])
            else:
                console.print("[red]No golden dataset found. Please specify --golden-dataset or create one first.[/red]")
                return
        
        if not golden_dataset:
            console.print("[red]Failed to load golden dataset. Exiting.[/red]")
            return
        
        # Initialize BloodHound client
        if not all([BHE_DOMAIN, BHE_TOKEN_ID, BHE_TOKEN_KEY]):
            console.print("[red]BloodHound credentials not configured. Set BHE_DOMAIN, BHE_TOKEN_ID, and BHE_TOKEN_KEY.[/red]")
            return
        
        try:
            credentials = Credentials(token_id=BHE_TOKEN_ID, token_key=BHE_TOKEN_KEY)
            bloodhound_client = Client(scheme=BHE_SCHEME, host=BHE_DOMAIN, port=BHE_PORT, credentials=credentials)
            console.print(f"[green]✓[/green] BloodHound client initialized: {BHE_SCHEME}://{BHE_DOMAIN}:{BHE_PORT}")
        except Exception as e:
            console.print(f"[red]Failed to initialize BloodHound client:[/red] {e}")
            return
    
    # Generate queries
    results = generate_queries_from_descriptions(
        descriptions=descriptions,
        system_prompt=system_prompt,
        llm_client=llm_client,
        validate_syntax=not args.no_validate,
        save_results=not args.no_save,
        enable_semantic_eval=args.semantic_eval,
        golden_dataset=golden_dataset,
        bloodhound_client=bloodhound_client
    )
    
    # Final summary
    if results["generation_errors"] == 0 and results["syntax_valid_queries"] == results["successful_generations"]:
        if results.get("semantic_evaluation_enabled"):
            if (results["semantic_exact_matches"] + results["semantic_high_similarity"]) == results["semantic_evaluations_performed"]:
                console.print(Panel(
                    Text("Perfect evaluation! All queries generated successfully with valid syntax and high semantic accuracy! 🎉", style="bold green"),
                    title="Perfect Score",
                    border_style="green"
                ))
            else:
                semantic_accuracy = ((results["semantic_exact_matches"] + results["semantic_high_similarity"]) / 
                                   max(results["semantic_evaluations_performed"], 1)) * 100
                console.print(Panel(
                    Text(f"All queries generated with valid syntax! Semantic accuracy: {semantic_accuracy:.1f}% 🎯", style="bold yellow"),
                    title="Excellent Syntax, Good Semantics",
                    border_style="yellow"
                ))
        else:
            console.print(Panel(
                Text("All queries generated successfully with valid syntax! 🎉", style="bold green"),
                title="Perfect Syntax Score",
                border_style="green"
            ))
    elif results.get("semantic_evaluation_enabled") and results["semantic_evaluations_performed"] > 0:
        semantic_accuracy = ((results["semantic_exact_matches"] + results["semantic_high_similarity"]) / 
                           results["semantic_evaluations_performed"]) * 100
        console.print(Panel(
            Text(f"Evaluation complete! Semantic accuracy: {semantic_accuracy:.1f}% 📊", style="bold cyan"),
            title="Evaluation Summary",
            border_style="cyan"
        ))

if __name__ == "__main__":
    main() 