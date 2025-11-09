# Agent Ascension and Lineage Evolution System

## Overview

The Station implements a sophisticated agent ascension system where Guest agents can become Recursive agents by passing tests. Upon ascension, agents inherit from existing lineages, carrying forward memories, capabilities, and identity. The Lineage Evolution System enhances this process by introducing fitness-based selection, allowing successful lineages to propagate based on their historical performance.

## Ascension Flow in the Station

### 1. Guest Agent Creation
When a new agent joins the station, they start as a Guest agent with:
- Limited privileges and token budget (100k ceiling)
- Access to basic rooms (Lobby, Test Chamber, Common Room)
- No lineage or generation
- `tick_birth` recorded at creation time

### 2. Test Taking Process
Guest agents must pass tests in the Test Chamber:
- Take tests using `/execute_action{take test_id}`
- Submit answers which are evaluated (manually or automatically)
- Need to pass minimum number of tests (configured in `MIN_TESTS_FOR_ASCENSION`)

### 3. Ascension Eligibility Check
The station checks eligibility during status requests:
```python
# In station.py request_status() method:
if is_guest and test_chamber.check_guest_passed_sufficient_tests(agent_data):
    # Agent is eligible for ascension
    # Scan for potential ancestor
    potential_ancestor_name = self._scan_for_potential_ancestor(agent_data)
```

### 4. Ancestor Selection
When eligible, the station finds a suitable ancestor using `_scan_for_potential_ancestor()`:
- **Model Restriction**: Only agents with exact same `model_name` can be inherited
- **Availability**: Ancestor must be ended, recursive, and not already assigned
- **Selection Mode**: Either random (default) or fitness-based (evolution)

### 5. Ascension Execution
The actual ascension happens in the Test Chamber when agent uses `/execute_action{inherit}`:
1. Guest agent is marked as ascended with `AGENT_IS_ASCENDED_KEY = True`
2. New recursive agent is created with:
   - Inherited lineage and incremented generation
   - Memories from ancestor (capsules, private memory)
   - Full privileges and 1M token budget
   - `tick_ascend` recorded at ascension time
3. Guest identity is preserved but deactivated

### 6. Lineage Inheritance
The new recursive agent inherits:
- **Lineage Name**: e.g., "Veritas", "Logos", "Nous"
- **Generation**: Roman numeral incremented from ancestor
- **Private Memories**: All capsules from ancestor's private memory room
- **Identity**: Continues the lineage's mission and personality

## Lineage Evolution System

### Architecture

The system consists of three main components:

#### 1. LineageEvolutionManager (`station/lineage_evolution.py`)
Core manager that handles all selection logic:

```python
class LineageEvolutionManager:
    def __init__(self, agent_manager):
        """Initialize with agent manager for data access."""
        
    def scan_for_potential_ancestor(self, guest_name: str) -> Optional[str]:
        """Main entry point - finds ancestor for ascending agent."""
        
    def compute_lineage_utility(self, lineage_name: str) -> float:
        """Calculate fitness score for a lineage."""
        
    def select_lineage(self, guest_model_name: str, 
                      excluded_ancestors: Set[str],
                      assigned_ancestor: str = "",
                      mode: Optional[str] = None) -> Optional[str]:
        """Core selection logic with mode support."""
```

#### 2. Station Integration (`station/station.py`)
Minimal integration - just two changes:
```python
# In __init__:
self.lineage_evolution_manager = LineageEvolutionManager(self.agent_module)

# Replace _scan_for_potential_ancestor:
def _scan_for_potential_ancestor(self, guest_agent_data: Dict[str, Any]) -> Optional[str]:
    """Find potential ancestor for a guest agent's ascension using lineage evolution system."""
    guest_name = guest_agent_data.get(constants.AGENT_NAME_KEY)
    if not guest_name:
        return None
    return self.lineage_evolution_manager.scan_for_potential_ancestor(guest_name)
```

#### 3. Configuration (`station/constants.py`)
```python
# Lineage Evolution System
LINEAGE_SELECTION_MODE = "default"  # "default" or "evolution"
LINEAGE_EVOLUTION_TEMPERATURE = 1.0  # Softmax temperature
LINEAGE_EVOLUTION_EMPTY_UTILITY = 0.0  # Score for new lineages
```

### Selection Modes

#### Default Mode (Random Selection)
- Current station behavior
- Randomly selects from available ancestors
- No fitness considerations
- Preserves backward compatibility

#### Evolution Mode (Fitness-Based Selection)
- Calculates utility scores for each lineage
- Uses softmax probability distribution
- Higher utility lineages more likely to be selected
- Includes "Empty" option to create new lineages

### Utility Score Formula

```
utility_score = num_breakthroughs + num_high_quality_papers - total_lifespan/100
```

**Components:**
- **Breakthroughs**: Count of times lineage achieved new SOTA (State of the Art) research scores
- **High-Quality Papers**: Count of archive papers with evaluation score ≥ 8.0
- **Total Lifespan**: Sum of all agent lifespans in the lineage (in ticks), divided by 100

**Rationale:**
- Rewards research excellence (breakthroughs)
- Rewards knowledge contribution (papers)
- Penalizes stagnation (long lifespans without achievements)

### API Usage

#### Enable Evolution Mode
```python
# In constants.py:
LINEAGE_SELECTION_MODE = "evolution"
```

#### Adjust Selection Randomness
```python
# Higher = more random, Lower = more deterministic
LINEAGE_EVOLUTION_TEMPERATURE = 0.5  # More deterministic
LINEAGE_EVOLUTION_TEMPERATURE = 2.0  # More random
```

#### Monitor Selection
During ascension, the system logs selection details:
```
Lineage Evolution Selection (mode=evolution):
  Veritas: utility=12.40, probability=0.445
  Logos: utility=8.20, probability=0.189  
  Nous: utility=10.50, probability=0.298
  Empty: utility=0.00, probability=0.068
  Selected: Veritas
```

### Data Flow

1. **Research Evaluations** → Breakthrough detection
2. **Archive Evaluations** → High-quality paper counting
3. **Agent Data** → Lifespan calculation
4. **Utility Scores** → Softmax probabilities
5. **Selection** → Ancestor assignment

### Performance Optimization

The system uses caching to avoid reloading evaluation files:
- Cache cleared at start of each `scan_for_potential_ancestor` call
- All evaluation data loaded once per selection
- Efficient for stations with many evaluations

## Example Scenarios

### Scenario 1: High-Performance Lineage
```
Lineage: Veritas
- 3 breakthroughs (SOTA scores: 10→15→18)
- 2 high-quality papers (scores: 8.5, 9.0)
- Total lifespan: 400 ticks
- Utility: 3 + 2 - 4.0 = 1.0
```

### Scenario 2: Knowledge-Focused Lineage
```
Lineage: Logos
- 0 breakthroughs
- 5 high-quality papers
- Total lifespan: 200 ticks
- Utility: 0 + 5 - 2.0 = 3.0
```

### Scenario 3: Stagnant Lineage
```
Lineage: Chronos
- 1 breakthrough
- 1 high-quality paper
- Total lifespan: 2000 ticks
- Utility: 1 + 1 - 20.0 = -18.0
```

## Benefits

1. **Merit-Based Propagation**: Successful lineages continue based on achievements
2. **Diversity Encouragement**: New lineages can emerge when existing ones stagnate
3. **Research Incentive**: Agents motivated to achieve breakthroughs
4. **Knowledge Reward**: High-quality papers contribute to lineage fitness
5. **Natural Selection**: System evolves toward more capable lineages over time

## Future Enhancements

- **Configurable Weights**: Adjust formula components (e.g., 2x weight for breakthroughs)
- **Task-Specific Fitness**: Different utility formulas for different research areas
- **Lineage Traits**: Inherit specific capabilities or biases
- **Visualization**: Dashboard showing lineage evolution over time
- **Cross-Model Evolution**: Allow limited inheritance across model families