#!/usr/bin/env python3
"""
cypher_validator.py – Rich-integrated Cypher syntax validation
------------------------------------------------------------
Provides both offline (ANTLR) and online (Neo4j EXPLAIN) validation
with beautiful Rich console output matching the project's styling.
"""

from __future__ import annotations
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

# Rich imports for consistent styling
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table

# ANTLR validation imports
from antlr4 import InputStream, CommonTokenStream
from antlr4_cypher import CypherLexer, CypherParser
from antlr4.error.ErrorListener import ErrorListener

# Neo4j validation imports (optional)
try:
    from neo4j import GraphDatabase
    from neo4j.exceptions import Neo4jError
    NEO4J_AVAILABLE = True
except ImportError:
    GraphDatabase = None
    Neo4jError = Exception
    NEO4J_AVAILABLE = False

# Initialize Rich Console
console = Console()

class _CypherErrorListener(ErrorListener):
    """Custom ANTLR error listener to capture syntax errors."""
    
    def __init__(self) -> None:
        self.messages: List[str] = []

    def syntaxError(self, recognizer, offendingSymbol, line, col, msg, e):
        """Called by ANTLR when syntax errors are encountered."""
        self.messages.append(f"line {line}:{col} {msg}")

@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Result of cypher query validation."""
    ok: bool
    offline_ok: bool
    online_ok: Optional[bool]  # None if skipped
    errors: List[str]
    execution_time: float

def _antlr_validate(query: str) -> Tuple[bool, List[str], float]:
    """
    Validate Cypher query syntax using ANTLR parser.
    
    Args:
        query: The Cypher query string to validate
        
    Returns:
        Tuple of (is_valid, error_messages, execution_time)
    """
    start_time = time.monotonic()
    
    try:
        stream = InputStream(query)
        lexer = CypherLexer(stream)
        tokens = CommonTokenStream(lexer)
        parser = CypherParser(tokens)

        listener = _CypherErrorListener()
        parser.removeErrorListeners()
        parser.addErrorListener(listener)
        parser.script()  # Entry rule in the OpenCypher grammar

        execution_time = time.monotonic() - start_time
        return not listener.messages, listener.messages, execution_time
        
    except Exception as e:
        execution_time = time.monotonic() - start_time
        return False, [f"ANTLR parsing error: {str(e)}"], execution_time

def _neo4j_validate(query: str, uri: str, user: str, pwd: str, 
                   database: str | None = None) -> Tuple[bool, List[str], float]:
    """
    Validate Cypher query using Neo4j EXPLAIN (semantic validation).
    
    Args:
        query: The Cypher query string to validate
        uri: Neo4j connection URI
        user: Neo4j username
        pwd: Neo4j password
        database: Optional database name
        
    Returns:
        Tuple of (is_valid, error_messages, execution_time)
    """
    if not NEO4J_AVAILABLE:
        return False, ["Neo4j driver not installed"], 0.0

    start_time = time.monotonic()
    cypher = f"EXPLAIN {query}"
    
    try:
        auth = (user, pwd)
        with GraphDatabase.driver(uri, auth=auth) as drv, drv.session(
            database=database) as sess:
            sess.run(cypher).consume()
        
        execution_time = time.monotonic() - start_time
        return True, [], execution_time
        
    except Neo4jError as e:
        execution_time = time.monotonic() - start_time
        return False, [f"{e.code}: {e.message}"], execution_time
    except Exception as e:
        execution_time = time.monotonic() - start_time
        return False, [f"Unexpected Neo4j error: {str(e)}"], execution_time

def validate_query(query: str, 
                  uri: str | None = None,
                  user: str | None = None,
                  pwd: str | None = None,
                  database: str | None = None,
                  show_progress: bool = True) -> ValidationResult:
    """
    Validate a Cypher query with optional Rich console output.
    
    Args:
        query: The Cypher query string to validate
        uri: Optional Neo4j URI for semantic validation
        user: Neo4j username (defaults to 'neo4j')
        pwd: Neo4j password (defaults to 'neo4j')
        database: Optional Neo4j database name
        show_progress: Whether to show Rich console output
        
    Returns:
        ValidationResult with validation details
    """
    total_start_time = time.monotonic()
    
    # Offline ANTLR validation
    if show_progress:
        console.print("[dim]Running offline syntax validation...[/dim]")
    
    offline_ok, offline_errs, offline_time = _antlr_validate(query)
    
    # Online Neo4j validation (if requested)
    online_ok: Optional[bool] = None
    online_errs: List[str] = []
    online_time = 0.0
    
    if uri is not None:
        if show_progress:
            console.print("[dim]Running online semantic validation...[/dim]")
        
        online_ok, online_errs, online_time = _neo4j_validate(
            query, uri, user or "neo4j", pwd or "neo4j", database)

    total_time = time.monotonic() - total_start_time
    everything_ok = offline_ok and (online_ok is not False)
    all_errors = offline_errs + online_errs
    
    if show_progress:
        _display_validation_result(query, offline_ok, online_ok, all_errors, 
                                 offline_time, online_time, total_time)
    
    return ValidationResult(
        ok=everything_ok,
        offline_ok=offline_ok,
        online_ok=online_ok,
        errors=all_errors,
        execution_time=total_time
    )

def _display_validation_result(query: str, offline_ok: bool, online_ok: Optional[bool],
                             errors: List[str], offline_time: float, 
                             online_time: float, total_time: float) -> None:
    """Display validation results using Rich formatting."""
    
    # Create status indicators
    offline_status = "✓" if offline_ok else "✗"
    offline_color = "green" if offline_ok else "red"
    
    if online_ok is None:
        online_status = "⊝"
        online_color = "dim"
        online_text = "Skipped"
    elif online_ok:
        online_status = "✓"
        online_color = "green"
        online_text = "Valid"
    else:
        online_status = "✗"
        online_color = "red"
        online_text = "Invalid"
    
    # Create validation table
    table = Table(show_header=True, header_style="bold blue", box=None)
    table.add_column("Validation Type", style="dim", width=20)
    table.add_column("Status", justify="center", width=8)
    table.add_column("Result", width=15)
    table.add_column("Time", justify="right", width=10)
    
    table.add_row(
        "Syntax (ANTLR)",
        Text(offline_status, style=offline_color),
        Text("Valid" if offline_ok else "Invalid", style=offline_color),
        f"{offline_time:.3f}s"
    )
    
    if online_ok is not None:
        table.add_row(
            "Semantic (Neo4j)",
            Text(online_status, style=online_color),
            Text(online_text, style=online_color),
            f"{online_time:.3f}s"
        )
    
    # Overall result
    overall_ok = offline_ok and (online_ok is not False)
    overall_status = "✓" if overall_ok else "✗"
    overall_color = "green" if overall_ok else "red"
    
    table.add_row(
        "",
        "",
        "",
        "",
        style="dim"
    )
    table.add_row(
        Text("Overall", style="bold"),
        Text(overall_status, style=f"bold {overall_color}"),
        Text("Valid" if overall_ok else "Invalid", style=f"bold {overall_color}"),
        Text(f"{total_time:.3f}s", style="bold")
    )
    
    # Display results
    panel_style = "green" if overall_ok else "red"
    panel_title = "Query Validation Results"
    
    console.print(Panel(table, title=panel_title, border_style=panel_style))
    
    # Display errors if any
    if errors:
        error_text = Text()
        for i, error in enumerate(errors):
            if i > 0:
                error_text.append("\n")
            error_text.append(f"• {error}", style="red")
        
        console.print(Panel(
            error_text,
            title="Validation Errors",
            border_style="red"
        ))

def validate_query_batch(queries: List[dict], 
                        uri: str | None = None,
                        user: str | None = None,
                        pwd: str | None = None,
                        database: str | None = None) -> List[ValidationResult]:
    """
    Validate a batch of queries and return results.
    
    Args:
        queries: List of query dictionaries with 'query' and 'description' keys
        uri: Optional Neo4j URI for semantic validation
        user: Neo4j username
        pwd: Neo4j password
        database: Optional Neo4j database name
        
    Returns:
        List of ValidationResult objects
    """
    results = []
    
    for i, query_item in enumerate(queries):
        query_text = query_item.get("query", "")
        description = query_item.get("description", f"Query {i+1}")
        
        console.print(f"\n[bold blue]Validating Query {i+1}:[/bold blue] {description}")
        
        result = validate_query(
            query_text, uri, user, pwd, database, show_progress=True
        )
        results.append(result)
    
    return results

def get_validation_summary(results: List[ValidationResult]) -> Table:
    """
    Create a Rich table summarizing validation results.
    
    Args:
        results: List of ValidationResult objects
        
    Returns:
        Rich Table with summary statistics
    """
    total = len(results)
    syntax_valid = sum(1 for r in results if r.offline_ok)
    semantic_valid = sum(1 for r in results if r.online_ok is True)
    semantic_tested = sum(1 for r in results if r.online_ok is not None)
    overall_valid = sum(1 for r in results if r.ok)
    
    avg_time = sum(r.execution_time for r in results) / total if total > 0 else 0
    
    table = Table(title="Validation Summary", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="dim", width=30)
    table.add_column("Count", justify="right", width=10)
    table.add_column("Percentage", justify="right", width=12)
    
    table.add_row("Total Queries", str(total), "100.0%")
    table.add_row(
        Text("Syntax Valid", style="green"),
        Text(str(syntax_valid), style="green"),
        Text(f"{(syntax_valid/total*100):.1f}%" if total > 0 else "0.0%", style="green")
    )
    
    if semantic_tested > 0:
        table.add_row(
            Text("Semantic Valid", style="blue"),
            Text(str(semantic_valid), style="blue"),
            Text(f"{(semantic_valid/semantic_tested*100):.1f}%" if semantic_tested > 0 else "0.0%", style="blue")
        )
    
    table.add_row(
        Text("Overall Valid", style="bold green"),
        Text(str(overall_valid), style="bold green"),
        Text(f"{(overall_valid/total*100):.1f}%" if total > 0 else "0.0%", style="bold green")
    )
    
    table.add_row("", "", "", style="dim")
    table.add_row("Average Validation Time", f"{avg_time:.3f}s", "")
    
    return table 