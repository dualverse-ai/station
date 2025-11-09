# Simplified Evaluation Management System

## Overview

This document describes the simplified queue-based evaluation management system that cleanly separates evaluation IDs from version numbers and treats pending submissions as simple queues.

## Key Design Principles

1. **Simple Queue Model**: Treat pending submissions as queues - pop only when resources are available
2. **Clean ID/Version Separation**: Evaluation ID and version are separate integers, never combined in strings
3. **Resource-First Processing**: Only pop from queue AFTER confirming GPU/resource availability
4. **Single Source of Truth**: One JSON file per evaluation contains all versions and state

## Architecture

### 1. Queue Sources

**Agent Submissions** (pending_evaluations.yamll):
- Contains raw agent submissions
- Simple queue - items are removed atomically when processed
- Never contains Claude versions

**Claude Submissions** (submission files):
- `claude_workspaces/eval_1/submissions/submission_v2.py` = pending evaluation
- Files act as queue items - renamed to `_processed` when popped
- Version extracted from filename (v2 → 2)

### 2. Data Structure

**Queue Item Format**:
```python
{
    'eval_id': '1',           # Base evaluation ID (string)
    'author': 'TestAgent',
    'task_id': '1',           # Task ID
    'title': 'Submission',
    'content': 'code...',
    'tick': 5,
    'version': 2,             # Integer version (None for original, 2/3/etc for Claude)
    'source_type': 'claude',  # 'agent' or 'claude'
    'source_path': '/path/to/submission_v2.py'  # For Claude submissions
}
```

**JSON Evaluation Format**:
```json
{
    "id": "1",
    "author": "TestAgent",
    "original_submission": {
        "content": "...",
        "evaluation_result": {"status": "failed", ...}
    },
    "versions": {
        "v2": {
            "content": "...",
            "evaluation_result": {"status": "pending"}
        }
    },
    "current_state": {
        "latest_version": "v2",
        "claude_active": true
    }
}
```

### 3. Simplified Processing Flow

```python
# Main evaluation loop - Queue → GPU → Pop → JSON → Execute

def _evaluation_loop(self):
    while running:
        # 1. Get all queue items (both sources)
        queue_items = self._get_all_queue_items()
        
        for item in queue_items:
            eval_id = item['eval_id']  # Just the base ID
            version = item.get('version')  # None or integer
            
            # 2. Try to allocate GPU
            gpu_id = self._allocate_gpu(eval_id) if USE_DIFF_GPU else None
            if USE_DIFF_GPU and gpu_id is None:
                continue  # No GPU available
            
            # 3. Pop from queue atomically
            if not self._pop_from_queue(item):
                if gpu_id: self._deallocate_gpu(eval_id)
                continue  # Already processed
            
            # 4. Create/update evaluation in JSON
            self.eval_manager.create_evaluation(
                eval_id=eval_id,        # Base ID only
                author=item['author'],
                task_id=item['task_id'],
                title=item['title'],
                content=item['content'],
                tick=item['tick'],
                version=version         # Integer or None
            )
            
            # 5. Execute (parallel or sequential)
            if parallel_enabled:
                future = thread_pool.submit(self._execute_evaluation, item, gpu_id)
            else:
                self._execute_evaluation(item, gpu_id)
```

### 4. Queue Operations

**Pop from Queue**:
```python
def _pop_from_queue(item):
    if item['source_type'] == 'agent':
        # Remove from pending YAML atomically
        return self._remove_from_pending_evaluations(item['source_data'])
    else:  # claude
        # Rename file atomically
        os.rename(item['source_path'], item['source_path'] + "_processed")
        return True
```

### 5. Clean ID/Version Handling

**EvaluationManager API**:
```python
def create_evaluation(self, 
    eval_id: str,      # Base ID only (e.g., "1")
    version: int = None,  # Integer version (e.g., 2, 3)
    ...):
    
    if version:
        # Add version to existing evaluation
        version_key = f"v{version}"  # Store as "v2", "v3"
        eval_data["versions"][version_key] = {...}
    else:
        # Create new base evaluation
        ...
```

**Display IDs for Logging Only**:
```python
# Only for logging/display, never for data processing
display_id = f"{eval_id}_v{version}" if version else eval_id
print(f"Processing evaluation {display_id}")
```

## Benefits of Simplified Design

1. **No Complex State Management**: Queue items are either pending or processed
2. **Atomic Operations**: Pop from queue is atomic - prevents double processing
3. **Clean Separation**: ID and version never mixed in data structures
4. **Resource Safety**: GPU allocated before any state changes
5. **Simple Mental Model**: Think of it as a queue processor, not a state machine

## Notification System

### When to Notify

Notifications are sent when:
```
(no_claude_called OR (claude_called AND claude_finished)) AND all_evaluations_complete
```

### Three Notification Principles

1. **If success.md exists AND (debugged version score ≠ n.a. or debugged version times out)**: 
   - Notify about the **latest successful version** (e.g., v3)
   - Include Claude's success report in the notification
   - Message: "Your submission succeeded after automatic debugging. Claude's report: ..."

2. **If fail.md exists**:
   - Notify about the **original version**
   - Include Claude's failure report explaining why it couldn't be fixed
   - Message: "Your submission failed. Claude attempted to fix it but encountered: ..."

3. **If no report exists OR (success.md exists but debugged version score = n.a.)**:
   - Notify about the **original version**
   - No Claude report included (even if success.md exists)
   - Message: "Your submission failed with error: ..."
   - This handles both Claude timeout/crash AND cases where Claude thinks it succeeded but the evaluation still failed

## Example Flows

### Agent Submission Flow
1. Agent submits → Added to pending YAML queue
2. Evaluator finds item in queue
3. Allocates GPU (if needed)
4. Pops from queue (removes from YAML)
5. Creates JSON evaluation
6. Executes and updates result
7. If success → Notify agent of success
8. If failure → Check if Claude should debug
9. If Claude debugs → Follow Claude Version Flow
10. Otherwise → Notify agent of failure

### Claude Version Flow
1. Claude creates `submission_v2.py`
2. Evaluator finds file in workspace
3. Allocates GPU (if needed)
4. Pops from queue (renames file to `_processed`)
5. Adds version to existing JSON evaluation
6. Executes and updates result
7. Repeat for v3, v4... as Claude creates them
8. When Claude writes success.md or fail.md → Mark session complete
9. Apply three notification principles to determine message

## Key Improvements from Previous Design

**Before**:
- Complex state tracking across multiple files
- String parsing of combined IDs like "1_v2"
- State changes before resource allocation
- Multiple scanning and processing methods

**After**:
- Simple queue model
- Clean integer version numbers
- Resource allocation before state changes
- Single unified processing flow

## Implementation Notes

1. **Thread Safety**: File locking ensures atomic queue operations
2. **Idempotency**: Renaming and YAML removal are idempotent
3. **Recovery**: If crash occurs after pop but before execution, item is lost but system continues
4. **Monitoring**: All operations logged with display IDs for debugging

The simplified design makes the system much easier to understand, debug, and maintain while preserving all functionality.