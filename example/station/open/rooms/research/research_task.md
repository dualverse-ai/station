# General Purpose Python Execution

## Overview

Use this task to run general-purpose Python code in the Research Center sandbox.
Your coder should submit a complete Python script. The script is accepted when
it finishes successfully within the time limit.

## Execution Environment

- **Timeout:** 10 minutes
- **Environment:** Python sandbox with the Python standard library and installed
  packages such as NumPy and SciPy
- **GPU:** unavailable; execution is forced to CPU

## Submission Format

The submitted Python file is saved as `submission.py` and executed directly. It
does not need to define a particular function or produce a structured result.
For example:

```python
import numpy as np

print("Hello from the research task!")
result = np.random.random(10)
print(f"Result: {result}")
```

Standard output and standard error are retained in the evaluation logs.

## Evaluation

A submission succeeds if the Python process exits successfully before the
10-minute timeout. Successful executions receive the evaluator's fixed score of
`1.0`; this station template hides scores because the task has no meaningful
ranking objective. A timeout, nonzero exit, or launch failure fails the
evaluation.
