# Chain-of-Thought Trace Generator for Model Distillation

This repository contains tools for generating Chain-of-Thought (CoT) reasoning traces from large language models (like GPT-4/o3) to create training data for model distillation. The system is specifically designed for BloodHound Cypher query generation but can be adapted for other domains.

## Overview

The system works by:
1. Taking query descriptions and their corresponding Cypher queries
2. Prompting a large model to generate detailed reasoning traces
3. Processing the traces into various training data formats
4. Preparing the data for training smaller models through distillation

## Files

- `generate_cot_traces.py` - Main script for generating CoT traces
- `prepare_training_data.py` - Script to process traces into training formats
- `example_usage.py` - Examples showing how to use the system
- `config.yaml` - Configuration file for different models and settings
- `requirements.txt` - Python dependencies
- `data/queries.json` - Input dataset with BloodHound queries

## Installation

1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Quick Start

### 1. Generate CoT Traces

Basic usage with your queries:
```bash
python generate_cot_traces.py \
    --input data/queries.json \
    --output traces_output.json \
    --api-key $OPENAI_API_KEY \
    --model gpt-4 \
    --max-queries 10
```

### 2. Prepare Training Data

Convert traces to training format:
```bash
python prepare_training_data.py \
    --input traces_output.json \
    --output-dir training_data \
    --format all
```

### 3. Run Examples

See the system in action:
```bash
python example_usage.py
```

## Detailed Usage

### Generating CoT Traces

The main script `generate_cot_traces.py` supports various options:

```bash
python generate_cot_traces.py \
    --input data/queries.json \           # Input JSON with queries
    --output output/traces.json \         # Output file for traces
    --api-key $OPENAI_API_KEY \          # Your API key
    --model gpt-4 \                      # Model to use
    --batch-size 5 \                     # Concurrent requests
    --max-queries 100 \                  # Limit number of queries
    --base-url https://api.openai.com/v1 # API endpoint
```

#### Supported Models

- `gpt-4` - Best quality traces
- `gpt-4-turbo` - Faster, good quality
- `gpt-3.5-turbo` - Fastest, lower quality
- `o3` - When available (update base-url as needed)

### Input Data Format

Your input JSON should contain an array of objects with:
```json
[
  {
    "description": "Find all domain admins",
    "query": "MATCH (n:User)-[:MemberOf*1..]->(g:Group {name: \"DOMAIN ADMINS@\"}) RETURN n",
    "source": "optional-source-url"
  }
]
```

### Output Format

Generated traces include:
```json
[
  {
    "description": "Find all domain admins",
    "query": "MATCH (n:User)-[:MemberOf*1..]->(g:Group {name: \"DOMAIN ADMINS@\"}) RETURN n",
    "source": "optional-source-url",
    "reasoning_trace": "Step 1: Identify the goal...",
    "generated_at": "2024-01-01T12:00:00"
  }
]
```

### Training Data Preparation

The `prepare_training_data.py` script converts traces into multiple formats:

#### Available Formats

1. **Instruction Format** - For instruction-following models
2. **Conversation Format** - For chat models
3. **CoT Format** - For chain-of-thought training
4. **QA Format** - Simple question-answer pairs

#### Usage

```bash
python prepare_training_data.py \
    --input traces_output.json \
    --output-dir training_data \
    --format conversation \        # or "all" for all formats
    --train-ratio 0.8 \           # 80% for training
    --val-ratio 0.1 \             # 10% for validation
    --few-shot 3 \                # Number of few-shot examples
    --seed 42                     # Random seed for reproducibility
```

#### Output Structure

```
training_data/
├── train_conversation.json
├── validation_conversation.json
├── test_conversation.json
├── few_shot_examples_conversation.json
└── training_config_conversation.json
```

## Advanced Usage

### Custom Prompts

You can customize the prompting strategy by subclassing `CoTTraceGenerator`:

```python
class CustomCoTGenerator(CoTTraceGenerator):
    def create_cot_prompt(self, description: str, query: str) -> str:
        return f"""Custom prompt for {description}
        
        Query: {query}
        
        Provide reasoning:"""
```

### Different Models

For different API providers or local models:

```python
# For local models
generator = CoTTraceGenerator(
    api_key="dummy",
    model="local-model",
    base_url="http://localhost:8000/v1"
)

# For other providers
generator = CoTTraceGenerator(
    api_key="your-key",
    model="claude-3",
    base_url="https://api.anthropic.com/v1"
)
```

### Batch Processing

For large datasets, use batch processing:

```python
async with CoTTraceGenerator(api_key, model="gpt-4") as generator:
    results = await generator.process_batch(
        queries, 
        batch_size=10  # Adjust based on rate limits
    )
```

## Configuration

Edit `config.yaml` to customize:

```yaml
generation:
  batch_size: 5
  max_retries: 3
  temperature: 0.7
  max_tokens: 2000

filtering:
  min_description_length: 10
  max_queries: null
```

## Training Your Model

After preparing training data, you can train a smaller model:

1. **Choose a base model** (e.g., GPT-2, T5, or a smaller transformer)
2. **Use the generated training data** in your preferred training framework
3. **Follow the suggested parameters** from `training_config_*.json`

Example training configurations are provided in the output files.

## Tips for Better Results

### Prompt Engineering
- Customize prompts for your specific domain
- Include examples of good reasoning in your prompts
- Adjust temperature based on desired creativity vs consistency

### Data Quality
- Filter out low-quality traces before training
- Manually review a sample of generated traces
- Consider multiple models for diverse reasoning styles

### Model Selection
- Use GPT-4 for highest quality traces
- Use GPT-3.5-turbo for faster iteration
- Consider o3 when available for even better reasoning

## Troubleshooting

### Common Issues

1. **Rate Limiting**
   - Reduce batch size
   - Increase delays between requests
   - Use exponential backoff (built-in)

2. **Poor Quality Traces**
   - Improve prompts
   - Use a better model
   - Filter input data more strictly

3. **API Errors**
   - Check API key validity
   - Verify model availability
   - Check rate limits and quotas

### Logging

Check `cot_generation.log` for detailed logs:
```bash
tail -f cot_generation.log
```

## Examples

See `example_usage.py` for complete examples including:
- Basic trace generation
- Multiple model comparison
- Custom prompt strategies
- Error handling

## Contributing

To extend this system:
1. Add new output formats in `TrainingDataProcessor`
2. Implement new model providers in `CoTTraceGenerator`
3. Add domain-specific prompt templates
4. Improve data filtering and quality checks

## License

MIT License - see LICENSE file for details.

## Citation

If you use this system in your research, please cite:

```bibtex
@software{cot_trace_generator,
  title={Chain-of-Thought Trace Generator for Model Distillation},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo}
}
```
