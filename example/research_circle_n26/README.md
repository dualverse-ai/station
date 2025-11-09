## Task Specification

This is the Circle Packing task with n=26. 

## Setup

```bash
cp -r example/station_default station_data
cp -r example/research_circle_n26/research station_data/rooms
cp example/research_circle_n26/constant_config.yaml station_data/constant_config.yaml
```

You can disable GPU access by adding the following line to `station_data/constant_config.yaml`:

```yaml
RESEARCH_EVAL_USE_DIFF_GPU: False
```


## External Evaluation

No external evaluation needed for this task. All valid configurations are stored in `station_data/rooms/research/internal/packings/*.npz`.

The best-performing configuration from the paper is included in `station_sota.py`.