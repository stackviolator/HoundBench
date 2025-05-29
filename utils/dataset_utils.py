#!/usr/bin/env python3

import json
import random
import re
from typing import List, Dict, Any, Tuple, Optional, Union
from rich.console import Console
from datasets import Dataset, load_dataset, DatasetDict
from datasets.splits import Split

# Initialize Rich Console
console = Console()

def load_queries_dataset(
    dataset_path: Union[str, Dict[str, Any]], 
    split: Optional[str] = None,
    cache_dir: Optional[str] = None
) -> Dataset:
    """
    Load the queries dataset using Hugging Face datasets library.
    
    Args:
        dataset_path: Either a path to local JSON file, a Hugging Face dataset name, 
                     or a dict with dataset configuration
        split: Dataset split to load (e.g., 'train', 'test', 'validation')
        cache_dir: Directory to cache the dataset
    
    Returns:
        Dataset object containing query-description pairs
    """
    try:
        # Handle different input types
        if isinstance(dataset_path, str):
            if dataset_path.endswith('.json'):
                # Load from local JSON file
                console.print(f"[dim]Loading dataset from local JSON file: {dataset_path}[/dim]")
                with open(dataset_path, 'r') as f:
                    data = json.load(f)
                
                # Validate dataset format
                required_fields = ['description', 'query']
                for i, entry in enumerate(data):
                    for field in required_fields:
                        if field not in entry:
                            raise ValueError(f"Entry {i} missing required field: {field}")
                
                dataset = Dataset.from_list(data)
            else:
                # Load from Hugging Face Hub
                console.print(f"[dim]Loading dataset from Hugging Face Hub: {dataset_path}[/dim]")
                dataset = load_dataset(dataset_path, split=split, cache_dir=cache_dir)
        
        elif isinstance(dataset_path, dict):
            # Load with custom configuration
            console.print(f"[dim]Loading dataset with custom configuration[/dim]")
            dataset = load_dataset(**dataset_path, split=split, cache_dir=cache_dir)
        
        else:
            raise ValueError("dataset_path must be a string (file path or HF dataset name) or dict (config)")
        
        # Validate required columns
        required_columns = ['description', 'query']
        missing_columns = [col for col in required_columns if col not in dataset.column_names]
        if missing_columns:
            raise ValueError(f"Dataset missing required columns: {missing_columns}")
        
        console.print(f"[green]✓[/green] Loaded queries dataset")
        console.print(f"[dim]Dataset contains {len(dataset)} query-description pairs[/dim]")
        console.print(f"[dim]Columns: {dataset.column_names}[/dim]")
        
        return dataset
        
    except FileNotFoundError:
        console.print(f"[red]Error:[/red] Dataset file not found at {dataset_path}")
        raise
    except Exception as e:
        console.print(f"[red]Error loading dataset:[/red] {e}")
        raise

def split_dataset(
    dataset: Dataset, 
    train_ratio: float = 0.85, 
    random_seed: int = 42,
    stratify_by_column: Optional[str] = None
) -> Tuple[Dataset, Dataset]:
    """
    Split dataset into train and test sets.
    
    Args:
        dataset: The dataset to split
        train_ratio: Ratio of data to use for training (0 < train_ratio < 1)
        random_seed: Random seed for reproducible splits
        stratify_by_column: Column name to stratify by (optional)
    
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    
    # Calculate test ratio
    test_ratio = 1.0 - train_ratio
    
    # Use datasets library's train_test_split method
    split_dataset = dataset.train_test_split(
        test_size=test_ratio,
        seed=random_seed,
        stratify_by_column=stratify_by_column
    )
    
    train_set = split_dataset['train']
    test_set = split_dataset['test']
    
    console.print(f"[green]✓[/green] Dataset split completed:")
    console.print(f"[dim]  Train set: {len(train_set)} entries ({len(train_set)/len(dataset)*100:.1f}%)[/dim]")
    console.print(f"[dim]  Test set: {len(test_set)} entries ({len(test_set)/len(dataset)*100:.1f}%)[/dim]")
    
    return train_set, test_set

def create_dataset_from_queries_json(json_file_path: str, push_to_hub: bool = False, repo_id: Optional[str] = None) -> Dataset:
    """
    Create a Hugging Face Dataset from a queries JSON file and optionally push to Hub.
    
    Args:
        json_file_path: Path to the JSON file containing queries
        push_to_hub: Whether to push the dataset to Hugging Face Hub
        repo_id: Repository ID for pushing to Hub (required if push_to_hub=True)
    
    Returns:
        Dataset object
    """
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        
        # Validate and clean data
        required_fields = ['description', 'query']
        cleaned_data = []
        
        for i, entry in enumerate(data):
            # Check required fields
            if not all(field in entry for field in required_fields):
                console.print(f"[yellow]Warning:[/yellow] Skipping entry {i} - missing required fields")
                continue
            
            # Clean and standardize entry
            cleaned_entry = {
                'description': entry['description'].strip(),
                'query': entry['query'].strip(),
                'source': entry.get('source', 'unknown'),
                'id': i
            }
            cleaned_data.append(cleaned_entry)
        
        # Create dataset
        dataset = Dataset.from_list(cleaned_data)
        
        # Add metadata
        dataset_info = {
            'description': 'Cypher query dataset for BloodHound evaluation',
            'features': {
                'description': 'Natural language description of the query task',
                'query': 'Corresponding Cypher query',
                'source': 'Source of the query (URL, paper, etc.)',
                'id': 'Unique identifier for the entry'
            },
            'total_examples': len(cleaned_data)
        }
        
        console.print(f"[green]✓[/green] Created dataset with {len(cleaned_data)} entries")
        
        # Push to Hub if requested
        if push_to_hub:
            if not repo_id:
                raise ValueError("repo_id is required when push_to_hub=True")
            
            console.print(f"[dim]Pushing dataset to Hugging Face Hub: {repo_id}[/dim]")
            dataset.push_to_hub(repo_id)
            console.print(f"[green]✓[/green] Dataset pushed to Hub: {repo_id}")
        
        return dataset
        
    except Exception as e:
        console.print(f"[red]Error creating dataset:[/red] {e}")
        raise

def filter_dataset_by_source(dataset: Dataset, sources: List[str]) -> Dataset:
    """Filter dataset to only include entries from specific sources."""
    filtered = dataset.filter(lambda example: example['source'] in sources)
    console.print(f"[green]✓[/green] Filtered dataset: {len(filtered)} entries from sources: {sources}")
    return filtered

def sample_dataset(dataset: Dataset, n_samples: int, random_seed: int = 42) -> Dataset:
    """Sample n random examples from the dataset."""
    if n_samples >= len(dataset):
        console.print(f"[yellow]Warning:[/yellow] Requested {n_samples} samples but dataset only has {len(dataset)} entries")
        return dataset
    
    sampled = dataset.shuffle(seed=random_seed).select(range(n_samples))
    console.print(f"[green]✓[/green] Sampled {n_samples} entries from dataset")
    return sampled

def normalize_cypher_query(query: str) -> str:
    """Normalize a Cypher query for comparison."""
    if not query:
        return ""
    
    # Convert to lowercase
    normalized = query.lower()
    
    # Remove extra whitespace and normalize spacing
    normalized = re.sub(r'\s+', ' ', normalized)
    normalized = normalized.strip()
    
    # Normalize common patterns
    # Remove spaces around parentheses, brackets, and operators
    normalized = re.sub(r'\s*\(\s*', '(', normalized)
    normalized = re.sub(r'\s*\)\s*', ')', normalized)
    normalized = re.sub(r'\s*\[\s*', '[', normalized)
    normalized = re.sub(r'\s*\]\s*', ']', normalized)
    normalized = re.sub(r'\s*:\s*', ':', normalized)
    normalized = re.sub(r'\s*=\s*', '=', normalized)
    normalized = re.sub(r'\s*<\s*', '<', normalized)
    normalized = re.sub(r'\s*>\s*', '>', normalized)
    
    # Normalize quotes (convert single quotes to double quotes)
    normalized = re.sub(r"'([^']*)'", r'"\1"', normalized)
    
    # Remove comments
    normalized = re.sub(r'//.*$', '', normalized, flags=re.MULTILINE)
    
    return normalized.strip()

def tokenize_cypher_query(query: str) -> List[str]:
    """Tokenize a Cypher query into meaningful tokens."""
    if not query:
        return []
    
    # Split on common delimiters while preserving them
    tokens = re.findall(r'\w+|[()[\]{}:=<>!-]|"[^"]*"', query.lower())
    
    # Filter out empty tokens
    return [token for token in tokens if token.strip()]

def calculate_token_similarity(tokens1: List[str], tokens2: List[str]) -> float:
    """Calculate Jaccard similarity between two token lists."""
    if not tokens1 and not tokens2:
        return 1.0
    if not tokens1 or not tokens2:
        return 0.0
    
    set1 = set(tokens1)
    set2 = set(tokens2)
    
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    return intersection / union if union > 0 else 0.0

def extract_query_structure(query: str) -> Dict[str, List[str]]:
    """Extract structural components from a Cypher query."""
    structure = {
        "match_patterns": [],
        "where_conditions": [],
        "return_clauses": [],
        "with_clauses": [],
        "order_by": [],
        "limit": []
    }
    
    if not query:
        return structure
    
    # Normalize query for parsing
    normalized = normalize_cypher_query(query)
    
    # Extract MATCH patterns
    match_patterns = re.findall(r'match\s+([^where^return^with^order^limit]+)', normalized, re.IGNORECASE)
    structure["match_patterns"] = [pattern.strip() for pattern in match_patterns]
    
    # Extract WHERE conditions
    where_conditions = re.findall(r'where\s+([^return^with^order^limit]+)', normalized, re.IGNORECASE)
    structure["where_conditions"] = [condition.strip() for condition in where_conditions]
    
    # Extract RETURN clauses
    return_clauses = re.findall(r'return\s+([^order^limit]+)', normalized, re.IGNORECASE)
    structure["return_clauses"] = [clause.strip() for clause in return_clauses]
    
    # Extract WITH clauses
    with_clauses = re.findall(r'with\s+([^match^where^return^order^limit]+)', normalized, re.IGNORECASE)
    structure["with_clauses"] = [clause.strip() for clause in with_clauses]
    
    return structure

def calculate_structural_similarity(structure1: Dict[str, List[str]], structure2: Dict[str, List[str]]) -> float:
    """Calculate structural similarity between two query structures."""
    total_score = 0.0
    total_weight = 0.0
    
    # Weights for different structural components
    weights = {
        "match_patterns": 0.4,
        "where_conditions": 0.3,
        "return_clauses": 0.2,
        "with_clauses": 0.1
    }
    
    for component, weight in weights.items():
        list1 = structure1.get(component, [])
        list2 = structure2.get(component, [])
        
        if not list1 and not list2:
            component_score = 1.0
        elif not list1 or not list2:
            component_score = 0.0
        else:
            # Calculate similarity for this component
            set1 = set(list1)
            set2 = set(list2)
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            component_score = intersection / union if union > 0 else 0.0
        
        total_score += component_score * weight
        total_weight += weight
    
    return total_score / total_weight if total_weight > 0 else 0.0

def calculate_query_similarity(
    generated_query: str, 
    ground_truth_query: str,
    exact_match_weight: float = 0.4,
    token_similarity_weight: float = 0.3,
    structural_similarity_weight: float = 0.3
) -> Dict[str, Any]:
    """
    Calculate similarity between generated query and ground truth query.
    
    Args:
        generated_query: The LLM-generated query
        ground_truth_query: The ground truth query from dataset
        exact_match_weight: Weight for exact match component
        token_similarity_weight: Weight for token similarity component
        structural_similarity_weight: Weight for structural similarity component
    
    Returns:
        Dict with similarity metrics and overall score
    """
    # Normalize queries
    norm_generated = normalize_cypher_query(generated_query)
    norm_ground_truth = normalize_cypher_query(ground_truth_query)
    
    # Check for exact match
    exact_match = norm_generated == norm_ground_truth
    
    # Calculate token similarity
    tokens_generated = tokenize_cypher_query(norm_generated)
    tokens_ground_truth = tokenize_cypher_query(norm_ground_truth)
    token_similarity = calculate_token_similarity(tokens_generated, tokens_ground_truth)
    
    # Calculate structural similarity
    structure_generated = extract_query_structure(norm_generated)
    structure_ground_truth = extract_query_structure(norm_ground_truth)
    structural_similarity = calculate_structural_similarity(structure_generated, structure_ground_truth)
    
    # Calculate overall score (weighted average)
    if exact_match:
        overall_score = 1.0
    else:
        overall_score = (
            exact_match_weight * (1.0 if exact_match else 0.0) +
            token_similarity_weight * token_similarity +
            structural_similarity_weight * structural_similarity
        )
    
    return {
        "exact_match": exact_match,
        "token_similarity": token_similarity,
        "structural_similarity": structural_similarity,
        "overall_score": overall_score,
        "normalized_generated": norm_generated,
        "normalized_ground_truth": norm_ground_truth,
        "tokens_generated": tokens_generated,
        "tokens_ground_truth": tokens_ground_truth,
        "structure_generated": structure_generated,
        "structure_ground_truth": structure_ground_truth
    } 