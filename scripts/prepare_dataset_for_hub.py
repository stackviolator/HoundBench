#!/usr/bin/env python3

import json
import os
import re
from typing import Dict, List, Any, Tuple
from collections import Counter
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track
from datasets import Dataset, DatasetDict
import hashlib

console = Console()

def validate_query_syntax(query: str) -> Tuple[bool, List[str]]:
    """Basic Cypher syntax validation."""
    errors = []
    
    # Check for basic Cypher keywords
    cypher_keywords = ['MATCH', 'WHERE', 'RETURN', 'WITH', 'OPTIONAL', 'CREATE', 'DELETE', 'SET']
    has_cypher_keyword = any(keyword.upper() in query.upper() for keyword in cypher_keywords)
    
    if not has_cypher_keyword:
        errors.append("No Cypher keywords found")
    
    # Check for balanced parentheses
    paren_count = query.count('(') - query.count(')')
    if paren_count != 0:
        errors.append(f"Unbalanced parentheses (difference: {paren_count})")
    
    # Check for balanced brackets
    bracket_count = query.count('[') - query.count(']')
    if bracket_count != 0:
        errors.append(f"Unbalanced brackets (difference: {bracket_count})")
    
    # Check for balanced braces
    brace_count = query.count('{') - query.count('}')
    if brace_count != 0:
        errors.append(f"Unbalanced braces (difference: {brace_count})")
    
    return len(errors) == 0, errors

def clean_and_validate_entry(entry: Dict[str, Any], index: int) -> Tuple[Dict[str, Any], List[str]]:
    """Clean and validate a single dataset entry."""
    issues = []
    cleaned_entry = {}
    
    # Validate required fields
    required_fields = ['description', 'query', 'source']
    for field in required_fields:
        if field not in entry:
            issues.append(f"Missing required field: {field}")
            return None, issues
    
    # Clean description
    description = entry['description'].strip()
    if not description:
        issues.append("Empty description")
        return None, issues
    
    # Remove common prefixes/suffixes that add no value
    description = re.sub(r'^(Find|Get|Return|Show|List)\s+', '', description, flags=re.IGNORECASE)
    description = description[0].upper() + description[1:] if description else description
    
    cleaned_entry['description'] = description
    
    # Clean query
    query = entry['query'].strip()
    if not query:
        issues.append("Empty query")
        return None, issues
    
    # Normalize whitespace in query
    query = re.sub(r'\s+', ' ', query)
    query = query.strip()
    
    # Validate query syntax
    is_valid, syntax_errors = validate_query_syntax(query)
    if not is_valid:
        issues.extend([f"Syntax error: {error}" for error in syntax_errors])
    
    cleaned_entry['query'] = query
    
    # Clean source
    source = entry['source'].strip()
    if not source:
        source = "unknown"
    
    # Normalize URLs
    if source.startswith('http'):
        # Remove trailing slashes and fragments
        source = re.sub(r'[/#]+$', '', source)
    
    cleaned_entry['source'] = source
    
    # Add metadata
    cleaned_entry['id'] = index
    cleaned_entry['query_length'] = len(query)
    cleaned_entry['description_length'] = len(description)
    
    # Calculate query complexity score
    complexity_indicators = [
        'shortestPath', 'allShortestPaths', 'OPTIONAL MATCH', 'WITH',
        'UNION', 'UNWIND', 'CASE', 'COLLECT', 'COUNT', 'datetime()'
    ]
    complexity_score = sum(1 for indicator in complexity_indicators if indicator in query)
    cleaned_entry['complexity_score'] = complexity_score
    
    # Extract query type
    query_upper = query.upper()
    if 'MATCH' in query_upper and 'RETURN' in query_upper:
        if 'CREATE' in query_upper or 'SET' in query_upper or 'DELETE' in query_upper:
            query_type = 'modification'
        else:
            query_type = 'read'
    elif 'CREATE' in query_upper:
        query_type = 'create'
    elif 'DELETE' in query_upper:
        query_type = 'delete'
    else:
        query_type = 'other'
    
    cleaned_entry['query_type'] = query_type
    
    # Extract domain entities
    entities = []
    entity_patterns = [
        r':User\b', r':Computer\b', r':Group\b', r':Domain\b', 
        r':GPO\b', r':OU\b', r':Container\b', r':CertTemplate\b'
    ]
    for pattern in entity_patterns:
        if re.search(pattern, query, re.IGNORECASE):
            entity = pattern[1:-2]  # Remove : and \b
            entities.append(entity)
    
    cleaned_entry['entities'] = entities
    
    return cleaned_entry, issues

def analyze_dataset_statistics(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze dataset statistics."""
    stats = {
        'total_entries': len(data),
        'sources': Counter(entry['source'] for entry in data),
        'query_types': Counter(entry.get('query_type', 'unknown') for entry in data),
        'entities': Counter(entity for entry in data for entity in entry.get('entities', [])),
        'avg_query_length': sum(entry.get('query_length', 0) for entry in data) / len(data),
        'avg_description_length': sum(entry.get('description_length', 0) for entry in data) / len(data),
        'complexity_distribution': Counter(entry.get('complexity_score', 0) for entry in data)
    }
    
    return stats

def detect_duplicates(data: List[Dict[str, Any]]) -> List[Tuple[int, int, str]]:
    """Detect potential duplicates in the dataset."""
    duplicates = []
    
    # Check for exact query matches
    query_hashes = {}
    for i, entry in enumerate(data):
        query_hash = hashlib.md5(entry['query'].lower().encode()).hexdigest()
        if query_hash in query_hashes:
            duplicates.append((query_hashes[query_hash], i, "exact_query_match"))
        else:
            query_hashes[query_hash] = i
    
    # Check for exact description matches
    desc_hashes = {}
    for i, entry in enumerate(data):
        desc_hash = hashlib.md5(entry['description'].lower().encode()).hexdigest()
        if desc_hash in desc_hashes:
            duplicates.append((desc_hashes[desc_hash], i, "exact_description_match"))
        else:
            desc_hashes[desc_hash] = i
    
    return duplicates

def create_enhanced_dataset(input_file: str, output_file: str = None) -> Dataset:
    """Create an enhanced dataset with validation and metadata."""
    console.print(Panel("🔍 Loading and validating dataset", style="blue"))
    
    # Load original data
    with open(input_file, 'r') as f:
        original_data = json.load(f)
    
    console.print(f"Loaded {len(original_data)} entries from {input_file}")
    
    # Clean and validate entries
    cleaned_data = []
    all_issues = []
    
    for i, entry in track(enumerate(original_data), description="Processing entries", total=len(original_data)):
        cleaned_entry, issues = clean_and_validate_entry(entry, i)
        
        if cleaned_entry:
            cleaned_data.append(cleaned_entry)
        
        if issues:
            all_issues.append({
                'index': i,
                'entry': entry,
                'issues': issues
            })
    
    console.print(f"✅ Successfully processed {len(cleaned_data)} entries")
    if all_issues:
        console.print(f"⚠️  Found issues in {len(all_issues)} entries")
    
    # Detect duplicates
    duplicates = detect_duplicates(cleaned_data)
    if duplicates:
        console.print(f"🔍 Found {len(duplicates)} potential duplicates")
    
    # Analyze statistics
    stats = analyze_dataset_statistics(cleaned_data)
    
    # Display statistics
    display_dataset_statistics(stats)
    
    # Create dataset
    dataset = Dataset.from_list(cleaned_data)
    
    # Add dataset info
    dataset_info = {
        'description': 'Enhanced BloodHound Cypher Queries Dataset with metadata and validation',
        'version': '1.0.0',
        'created': datetime.now().isoformat(),
        'total_examples': len(cleaned_data),
        'validation_issues': len(all_issues),
        'duplicates_found': len(duplicates),
        'statistics': stats
    }
    
    # Save enhanced dataset if output file specified
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(cleaned_data, f, indent=2)
        console.print(f"💾 Saved enhanced dataset to {output_file}")
    
    # Save validation report
    validation_report = {
        'dataset_info': dataset_info,
        'issues': all_issues,
        'duplicates': duplicates,
        'statistics': stats
    }
    
    report_file = input_file.replace('.json', '_validation_report.json')
    with open(report_file, 'w') as f:
        json.dump(validation_report, f, indent=2)
    console.print(f"📊 Saved validation report to {report_file}")
    
    return dataset

def display_dataset_statistics(stats: Dict[str, Any]):
    """Display dataset statistics in a nice format."""
    
    # Main statistics table
    main_table = Table(title="Dataset Statistics", show_header=True, header_style="bold cyan")
    main_table.add_column("Metric", style="dim")
    main_table.add_column("Value", justify="right")
    
    main_table.add_row("Total Entries", str(stats['total_entries']))
    main_table.add_row("Average Query Length", f"{stats['avg_query_length']:.1f} chars")
    main_table.add_row("Average Description Length", f"{stats['avg_description_length']:.1f} chars")
    main_table.add_row("Unique Sources", str(len(stats['sources'])))
    
    console.print(main_table)
    
    # Top sources table
    sources_table = Table(title="Top Sources", show_header=True, header_style="bold green")
    sources_table.add_column("Source", style="dim")
    sources_table.add_column("Count", justify="right")
    
    for source, count in stats['sources'].most_common(10):
        sources_table.add_row(source, str(count))
    
    console.print(sources_table)
    
    # Query types table
    types_table = Table(title="Query Types", show_header=True, header_style="bold yellow")
    types_table.add_column("Type", style="dim")
    types_table.add_column("Count", justify="right")
    types_table.add_column("Percentage", justify="right")
    
    total = stats['total_entries']
    for query_type, count in stats['query_types'].most_common():
        percentage = (count / total) * 100
        types_table.add_row(query_type, str(count), f"{percentage:.1f}%")
    
    console.print(types_table)
    
    # Top entities table
    entities_table = Table(title="Top Entities", show_header=True, header_style="bold magenta")
    entities_table.add_column("Entity", style="dim")
    entities_table.add_column("Count", justify="right")
    
    for entity, count in stats['entities'].most_common(10):
        entities_table.add_row(entity, str(count))
    
    console.print(entities_table)

def push_to_hub(dataset: Dataset, repo_id: str, private: bool = False):
    """Push the dataset to Hugging Face Hub."""
    console.print(Panel(f"🚀 Pushing dataset to Hugging Face Hub: {repo_id}", style="green"))
    
    try:
        dataset.push_to_hub(repo_id, private=private)
        console.print(f"✅ Successfully pushed dataset to {repo_id}")
        console.print(f"🔗 Dataset URL: https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        console.print(f"❌ Failed to push dataset: {e}")
        raise

def main():
    """Main function to prepare and optionally publish the dataset."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare BloodHound Cypher dataset for Hugging Face Hub")
    parser.add_argument("--input", default="data/queries.json", help="Input JSON file")
    parser.add_argument("--output", help="Output JSON file for enhanced dataset")
    parser.add_argument("--push-to-hub", help="Push to Hugging Face Hub (provide repo_id)")
    parser.add_argument("--private", action="store_true", help="Make the dataset private")
    parser.add_argument("--validate-only", action="store_true", help="Only validate, don't create dataset")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        console.print(f"❌ Input file not found: {args.input}")
        return
    
    # Create enhanced dataset
    dataset = create_enhanced_dataset(args.input, args.output)
    
    if args.validate_only:
        console.print("✅ Validation complete")
        return
    
    # Push to hub if requested
    if args.push_to_hub:
        push_to_hub(dataset, args.push_to_hub, args.private)
    
    console.print("🎉 Dataset preparation complete!")

if __name__ == "__main__":
    main() 