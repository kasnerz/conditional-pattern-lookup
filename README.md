# Conditional Pattern Lookup

Python generator for the Conditional Pattern Lookup task.

## Overview

The goal of the task is to find regex-like patterns in context. 

Each example consists of:
- An input sequence of characters or words.
- A query with a regex pattern and optionally constraints on the context.
- Ground truth span positions matching the query.


## Patterns
Patterns are strings that can include simple regex-like wildcards (`.` for characters, `\w+` for words).


## Constraints
- `preceded_by`: Pattern must be preceded by a specific character/word.
- `not_preceded_by`: Pattern must NOT be preceded by a specific character/word.
- `followed_by`: Pattern must be followed by a specific character/word.
- `not_followed_by`: Pattern must NOT be followed by a specific character/word.

## Quickstart

```bash
pip install -e .
python synthetic_dataset_generator.py
```

### Configuration Options

The `GenerationConfig` class supports the following parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | `GenerationMode` | Required | `CHARACTER` or `WORD` |
| `alphabet_size` | `int` | 4 | Number of characters in alphabet (for character mode) |
| `sequence_length` | `int` | 100 | Length of generated sequences |
| `num_examples` | `int` | 10 | Number of examples to generate |
| `num_queries_per_example` | `int` | 5 | Number of queries per example |
| `language` | `str` | "en" | Language for word mode ("en", "de", "es", "fr") |
| `min_pattern_occurrences` | `int` | 2 | Minimum occurrences of each pattern |
| `max_pattern_occurrences` | `int` | 8 | Maximum occurrences of each pattern |
| `min_pattern_length` | `int` | 2 | Minimum length of patterns |
| `max_pattern_length` | `int` | 3 | Maximum length of patterns |
| `wildcard_probability` | `float` | 0.1 | Probability of using wildcards in patterns |

## Output Format

The generated JSON dataset has the following structure:

```json
[
  {
    "input_sequence": "ABC DEF GHI...",
    "queries": [
      {
        "natural_language": "Find all sequences matching 'AB.' that are not followed by 'C'",
        "pattern": "AB.",
        "constraint_type": "not_followed_by",
        "constraint_pattern": "C"
      }
    ],
    "answers": [
      [[0, 3], [15, 18]]
    ]
  }
]
```

Each answer contains a list of `[start, end]` positions (character indices) where the pattern matches.