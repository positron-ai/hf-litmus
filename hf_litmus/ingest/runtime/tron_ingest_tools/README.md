# tron_ingest_tools

Shared utilities for the tron-ingest project, extracted from duplicated code across multiple files.

## Package Structure

The package is organized into domain-based modules:

- `export.py` - FX trace export to ingest directory format
- `model.py` - Generated model loading and weight transfer
- `ingest.py` - Haskell ingest pipeline runner
- `diff.py` - FX semantic diff functionality

## Usage

```python
from tron_ingest_tools import (
  # Export module
  ExportableModel,
  export_fx_trace,
  export_to_ingest_directory,
  get_model_config_safely,

  # Model module
  GeneratedModelWrapper,
  load_generated_model,
  smart_reconstruct_hf_name,
  transfer_weights,

  # Ingest module
  run_ingest_pipeline,

  # Diff module
  FxNode,
  FxTraceParser,
  compute_unified_diff,
  format_summary,
)
```

## Module Details

### export.py

Consolidates export-related code:
- `ExportableModel` - Wrapper for torch.export (handles both HF and sliding window attention)
- `get_model_config_safely()` - Safe config extraction with fallback
- `export_fx_trace()` - Generic function to export model to FX trace
- `export_to_ingest_directory()` - Export in format expected by Haskell ingest (root.fx + metadata.json)

### model.py

Unified model loading and weight transfer:
- `load_generated_model()` - Dynamically import and instantiate generated models
- `transfer_weights()` - Transfer weights from HF to generated model
- `smart_reconstruct_hf_name()` - Reconstruct HF parameter names from generated names
- `GeneratedModelWrapper` - Wrapper for generated models with different parameter names

### ingest.py

Unified ingest pipeline runner:
- `run_ingest_pipeline()` - Run the Haskell ingest tool to generate Python code

### diff.py

FX semantic diff functionality:
- `FxNode` - Dataclass for parsed FX nodes
- `FxTraceParser` - Parser for FX trace format
- `compute_unified_diff()` - Compute unified diff between two traces
- `format_summary()` - Generate summary statistics
- `compute_semantic_diff()` - Legacy set-based API
- `format_diff_output()` - Legacy format function

## Design Guidelines

- 2-space indentation (project style)
- Max 88 char line length
- Type hints for all functions
- Docstrings for all public APIs
- No trailing whitespace
- Never indent blank lines

## Backward Compatibility

All function signatures match existing usage in the scripts to ensure backward compatibility:
- `adjoint_fx_trace_test.py`
- `logit_equivalence_test.py`
- `fx_semantic_diff.py`
