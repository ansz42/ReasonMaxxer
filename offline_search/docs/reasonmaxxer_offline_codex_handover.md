## AI Generated Experiment

# Codex Handover — Adaptive Offline Search + Entropy-Weighted Distillation

## Project Goal

Implement a practical extension of the ideas in **ReasonMaxxer**:

- Paper: https://arxiv.org/abs/2605.06241
- Reference implementation: https://github.com/farukakgul/ReasonMaxxer

The goal is to test whether **rare but already-existing capabilities** can be discovered through aggressive offline sampling and then compressed back into the model with cheap offline training.

The key target regime is intentionally harder than the original ReasonMaxxer setup.

Example:

```text
1000 rollouts for one task

2 fully correct
10 nearly correct
988 incorrect
```

The original ReasonMaxxer-style mid-difficulty filtering may discard or under-utilize such a task.

This project should instead:

1. Search aggressively for rare successful trajectories.
2. Use multiple sampling configurations rather than one fixed decoding policy.
3. Assign graded rewards rather than only binary correct/incorrect labels.
4. Compute token entropy under the frozen base model.
5. Concentrate signed learning signal around uncertain/high-entropy decisions.
6. Train a small LoRA adapter offline.
7. Measure whether rare pass@k capability becomes substantially better pass@1 capability.
8. Optionally repeat this process for a small number of large offline iterations.

The main hypothesis is:

> Expensive test-time exploration can be compressed into model weights much more cheaply than repeatedly regenerating trajectories throughout online RL.

---

# 1. Design Philosophy

Keep the first implementation simple.

Do **not** begin with:

- MCTS
- token-level tree search
- process reward models
- complicated replay buffers
- GRPO
- PPO
- asynchronous rollout workers
- online model/trainer synchronization
- branch-point search
- learned value models

The MVP should be:

```text
Prompt set
   ↓
multi-configuration offline rollout search
   ↓
verifier / reward function
   ↓
adaptive allocation of remaining rollout budget
   ↓
trajectory dataset with scalar rewards
   ↓
base-model token entropy calculation
   ↓
entropy-weighted signed offline objective
   ↓
LoRA
   ↓
pass@1 / pass@k evaluation
```

The important experiment is whether this simple system works before adding more sophisticated search.

---

# 2. Primary Research Question

Given tasks where:

```text
pass@1 ≈ 0–2%
```

but:

```text
pass@128 or pass@256 > 0
```

can offline search + distillation significantly improve pass@1?

Ideal example:

```text
before:

pass@1     = 0.2%
pass@256   = 25%

after offline distillation:

pass@1     = 15–30%
```

The exact numbers are not assumptions. They are an example of the kind of effect we want to measure.

---

# 3. MVP Search Strategy

For each prompt, define a fixed total rollout budget.

Suggested starting values:

```yaml
initial_rollouts_per_config: 16
number_of_configs: 8
initial_rollouts_total: 128
total_budget_per_prompt: 256  # first cheap experiment
```

Later test:

```yaml
total_budget_per_prompt: 512
total_budget_per_prompt: 1024
```

## Initial decoding configurations

Start with a small, deliberately diverse set.

Example:

```python
SEARCH_CONFIGS = [
    {"temperature": 0.35, "top_p": 0.90},
    {"temperature": 0.50, "top_p": 0.95},

    {"temperature": 0.70, "top_p": 0.95},
    {"temperature": 0.80, "top_p": 1.00},

    {"temperature": 1.00, "top_p": 0.95},
    {"temperature": 1.15, "top_p": 0.98},
    {"temperature": 1.30, "top_p": 1.00},

    {
        "temperature": 0.85,
        "top_p": 0.98,
        "repetition_penalty": 1.05,
    },
]
```

Do not initially perform a giant Cartesian search over:

- temperature
- top_p
- top_k
- min_p
- typical_p
- frequency penalty
- presence penalty
- repetition penalty

That creates unnecessary search complexity.

Temperature + top-p should carry most of the exploratory variation.

Repetition penalty may be included as one or two special arms if the model tends to loop.

---

# 4. Adaptive Rollout Allocation

Stage 1:

Generate the same number of samples from every decoding configuration.

Example:

```text
8 configs × 16 rollouts = 128 initial rollouts
```

Score all rollouts.

Then rank configurations according to observed performance.

A deliberately simple initial scoring formula is acceptable:

```python
config_score = (
    4.0 * correct_rate
    + 1.0 * near_correct_rate
    + 0.2 * mean_reward
)
```

Use the remaining rollout budget preferentially on better-performing configurations.

Possible simple allocation strategies:

### Option A — softmax allocation

```python
allocation_probs = softmax(config_scores / allocation_temperature)
```

### Option B — rank-based allocation

Example:

```text
best config      30%
second           22%
third            16%
fourth           12%
remaining        split remaining 20%
```

### Option C — epsilon exploration

Allocate:

```text
80% according to performance
20% uniformly
```

This avoids prematurely collapsing onto one configuration based on noisy early results.

Use the simplest implementation first.

---

# 5. Reward / Verification

The system should support **scalar rewards**, not only binary rewards.

Required interface:

```python
def score_rollout(prompt, response, reference=None) -> dict:
    return {
        "reward": float,            # preferably 0.0–1.0
        "is_correct": bool,
        "near_correct": bool,
        "metadata": {...},
    }
```

The scorer should be modular because different domains need different verification.

## Example reward interpretation

```text
1.00 = fully correct
0.85 = correct solution with small final error
0.65 = correct approach, later mistake
0.40 = meaningful partial progress
0.15 = coherent but wrong
0.00 = garbage / irrelevant
```

Do not hard-code these semantic categories into the training system.

The trainer should only care about the scalar reward.

---

# 6. Advantage Calculation

Advantages must be normalized **within each prompt/problem**, not globally.

Suggested baseline:

```python
rewards = tensor([...])

advantage = (rewards - rewards.mean()) / (rewards.std() + 1e-6)
```

Alternative robust normalization can be tested later.

The important property is:

```text
high-reward trajectories → positive advantage
low-reward trajectories  → negative advantage
```

The objective must be able to both:

- increase probability of useful decisions
- decrease probability of bad decisions

This negative signal is important.

Do not reduce the method to positive-only SFT.

---

# 7. Do Not Train on Every Generated Rollout

Search may generate hundreds or thousands of trajectories per prompt.

Training does not need all of them.

Example:

```text
1000 searched trajectories

2 correct
10 strong near misses
70 mediocre
918 garbage
```

A better training subset might contain:

```text
2 correct
10 strong near misses
20–40 strategically selected negatives
```

The weakest 900+ examples probably add little useful information and may dominate training.

## Suggested trajectory selection

Per prompt, retain:

1. All correct trajectories up to a cap.
2. Top-K near-correct trajectories.
3. A limited set of hard negatives.
4. Optionally a few low-reward negatives for diversity.

Hard negatives should be prioritized over obvious garbage.

Possible criterion:

```text
reward relatively high
but still incorrect
```

These are likely trajectories where only one or a few decisions separate failure from success.

---

# 8. Token Entropy Calculation

For each retained trajectory, run the **frozen base model** teacher-forced over the generated answer.

The teacher-forced prefix **must** be the same chat-template string used at generation time (`apply_chat_template(..., add_generation_prompt=True)`), then the assistant response tokens appended to that prefix. Encoding `raw_prompt + response` as one string is a different conditional and makes the entropy uninterpretable.

For generated token position `t`:

```text
H_t = -Σ_v p(v | prefix_t) log p(v | prefix_t)
```

Store at minimum:

```python
{
    "input_ids": ...,
    "response_start": ...,
    "reward": ...,
    "advantage": ...,
    "token_entropy": ...,
}
```

Only response tokens should contribute to the training objective.

Prompt/system tokens must be masked.

---

# 9. Entropy Weighting

Do not initially use a brittle hard threshold if avoidable.

Preferred MVP:

```python
weight_t = sigmoid((entropy_t - entropy_threshold) / entropy_scale)
```

This gives:

```text
low entropy  → weak decision-training weight
high entropy → strong decision-training weight
```

Possible later ablation:

```python
weight_t = 1 if entropy_t > threshold else 0
```

to reproduce a more ReasonMaxxer-like hard mask.

Hyperparameters:

```yaml
entropy_threshold: configurable
entropy_scale: configurable
```

The implementation should make it trivial to sweep these.

---

# 10. Training Objective

Core signed decision loss:

```text
L_decision =
    - advantage_i
      Σ_t weight_t log pθ(token_t | prefix_t)
```

Interpretation:

```text
advantage > 0:
    increase probability of high-entropy choices

advantage < 0:
    decrease probability of high-entropy choices
```

Normalize by effective token weight so long generations do not automatically dominate:

```python
decision_loss = (
    -(advantage * token_weights * token_logprobs).sum()
    / (token_weights.sum() + 1e-8)
)
```

Be careful with batch-wise advantage broadcasting.

---

# 11. KL / Anchor Loss

Low-entropy positions should ideally remain close to the base model.

Use a KL anchor or another lightweight preservation loss.

Conceptually:

```text
high entropy:
    learn from signed reward

low entropy:
    stay close to base model
```

Possible implementation:

```text
L_total =
    L_decision
    + kl_coef * L_anchor
```

The reference/base model must remain frozen.

If keeping a full reference model in VRAM is too expensive, investigate:

1. precomputed reference logits for selected tokens
2. CPU/offloaded reference model
3. only computing reference distributions on low-entropy sampled positions
4. omitting KL in the first minimal experiment and adding it as an ablation

Do not overengineer the KL path before the basic signed entropy objective runs.

---

# 12. LoRA

Use LoRA first.

Suggested initial configuration:

```yaml
rank: 16 or 32
alpha: 32 or 64
dropout: 0
```

Candidate target modules:

```text
q_proj
k_proj
v_proj
o_proj
```

Start with QKVO.

Later ablations:

```text
O-only
QKVO r8
QKVO r16
QKVO r32
MLP inclusion
```

The paper's findings suggest relatively small adapters may be sufficient for policy steering.

---

# 13. Dataset Schema

Prefer Parquet.

Suggested rollout-search schema:

```text
problem_id
prompt
reference_answer
sampling_config_id
temperature
top_p
top_k
repetition_penalty
seed
response
reward
is_correct
near_correct
generated_tokens
```

Entropy-processed training dataset:

```text
problem_id
prompt
response
input_ids
labels
response_mask
token_entropy
token_weight
reward
advantage
sampling_config_id
```

If large array columns become inefficient in Parquet, use Arrow-native lists or another compact Hugging Face Dataset representation.

---

# 14. Suggested Repository Layout

```text
project/
│
├── README.md
├── requirements.txt
├── configs/
│   ├── search.yaml
│   ├── train.yaml
│   └── eval.yaml
│
├── src/
│   ├── search/
│   │   ├── generate.py
│   │   ├── sampling_configs.py
│   │   ├── adaptive_allocator.py
│   │   └── search_runner.py
│   │
│   ├── scoring/
│   │   ├── base.py
│   │   ├── exact_match.py
│   │   └── math_verifier.py
│   │
│   ├── data/
│   │   ├── select_trajectories.py
│   │   ├── compute_entropy.py
│   │   └── build_training_dataset.py
│   │
│   ├── training/
│   │   ├── trainer.py
│   │   ├── loss.py
│   │   └── lora.py
│   │
│   ├── eval/
│   │   ├── generate_eval.py
│   │   ├── pass_at_k.py
│   │   └── compare_models.py
│   │
│   └── utils/
│
├── scripts/
│   ├── 01_search.py
│   ├── 02_build_dataset.py
│   ├── 03_train.py
│   ├── 04_eval.py
│   └── run_iteration.py
│
└── outputs/
```

Do not make this layout rigid if the existing repository has a sensible structure.

Integrate cleanly with the current codebase if one already exists.

---

# 15. Search Runtime

Prefer a high-throughput inference backend if practical.

Candidates:

- vLLM
- SGLang
- Transformers generation for the simplest prototype

The search stage is likely the largest compute consumer.

Requirements:

- deterministic seed tracking
- ability to batch prompts
- per-request sampling parameters
- token count logging
- resume support
- save incrementally
- avoid losing completed rollouts if a run crashes

The generated dataset should be appendable / resumable.

---

# 16. Training Framework

Use whichever stack produces the least complexity in the current environment.

Possible choices:

- Hugging Face Transformers Trainer
- TRL custom Trainer
- Unsloth where useful
- plain PyTorch training loop if custom loss integration becomes cleaner

The custom entropy-weighted signed objective matters more than framework choice.

Do not force standard SFTTrainer semantics if they make the loss implementation awkward.

---

# 17. Evaluation

The primary metric is **pass@1**.

Also record:

```text
pass@4
pass@16
pass@64
pass@128
pass@256
```

when affordable.

Why:

The project's purpose is to convert:

```text
rare capability under search
```

into:

```text
reliable default capability
```

If pass@256 improves but pass@1 barely changes, the training method is not accomplishing the main objective.

---

# 18. Critical Baselines

At minimum compare:

## A. Base model

No training.

## B. Successful-rollout SFT

Train only on correct samples.

This tests whether the sophisticated objective is doing more than best-of-N distillation.

## C. Positive-only entropy training

Use entropy weighting but ignore negative advantages.

## D. Binary ReasonMaxxer-style training

Correct / incorrect reward only.

## E. Graded reward + entropy weighting

Main proposed objective.

## F. Graded reward + entropy weighting + adaptive sampling

Full proposed MVP.

Suggested table:

```text
Method                              pass@1  pass@16  pass@256
----------------------------------------------------------------
Base
Successful-rollout SFT
Positive entropy
Binary signed entropy
Graded signed entropy
Graded signed entropy + adaptive search
```

---

# 19. Task Selection for the First Experiment

Do not choose tasks that are:

```text
100% impossible for the base model
```

and do not choose:

```text
already easy at pass@1
```

The most interesting first dataset consists of tasks satisfying approximately:

```text
pass@1 low
pass@128/pass@256 nonzero
```

Example candidate bucket:

```text
pass@1 < 5%
pass@128 > 5%
```

Thresholds should be configurable.

The first experiment should ideally contain roughly:

```text
50–200 difficult tasks
```

rather than a huge benchmark.

This makes iteration fast.

---

# 20. Optional Offline Iteration

After training model `v1`, optionally repeat the full search.

```text
base model
   ↓
search_0
   ↓
offline train
   ↓
model_v1
   ↓
search_1
   ↓
offline train
   ↓
model_v2
```

Limit the first study to:

```text
2–4 major offline iterations
```

Do not recreate an online RL loop with extremely frequent regeneration.

The purpose is to test whether **coarse policy iteration** captures much of the benefit at far lower rollout cost.

Track generation tokens per iteration so cost comparisons are meaningful.

---

# 21. Cost Accounting

Every experiment should log:

```text
number of generated trajectories
number of generated tokens
search wall time
training tokens
training steps
training wall time
GPU-hours if possible
```

The major claim being investigated is economic as well as algorithmic.

We need to be able to compare approximately:

```text
offline search + offline training
```

against:

```text
online GRPO / RL rollout volume
```

Do not claim an arbitrary fixed multiplier such as "200× cheaper" without measured token counts.

Instead report:

```text
generated-token ratio
GPU-hour ratio
wall-clock ratio
```

where available.

---

# 22. Important Failure Modes

## Rare success may be accidental

One correct sample out of 1000 does not necessarily mean the model possesses a stable hidden capability.

Inspect whether the successful trajectory is coherent or merely lucky.

## Correct trajectory can have extremely low model probability

If success requires many individually microscopic-probability decisions, a sparse policy-steering update may not be sufficient.

## Reward quality dominates

A poor graded verifier can make the method worse than binary correctness.

Keep exact/objective verification whenever possible.

## Length bias

Do not allow long trajectories to receive larger gradients merely because they contain more tokens.

Normalize weighted token losses.

## Sampling config overfitting

A config may appear strong because of a few lucky rollouts.

Retain exploration during adaptive allocation.

## Garbage negative domination

Do not train on hundreds of useless zero-reward samples per successful trajectory.

## Distribution shift between base entropy and trained policy

The first implementation computes entropy under the frozen base model.

For later offline iterations, entropy should probably be recomputed under the model that generated that iteration's trajectories.

---

# 23. Things NOT to Implement Yet

Do not add these until the MVP has results:

- entropy-guided prefix branching
- MCTS
- beam search over reasoning states
- PRM
- learned critic
- token-level verifier
- DPO pair construction
- online GRPO
- distributed rollout/training architecture
- automatic curriculum generation
- Bayesian hyperparameter optimization
- elaborate search schedulers

These may be useful later but will obscure the core experiment.

---

# 24. Potential Phase 2 — Local Branch Search

Only after the whole-rollout method works.

For a near-correct trajectory:

```text
good prefix
   ↓
high-entropy decision
   ├── branch A → reward 0.2
   ├── branch B → reward 0.7
   └── branch C → reward 1.0
```

Sibling trajectories sharing the same prefix could provide exceptionally clean local preference signals.

However, entropy alone does not reliably identify the actual reasoning mistake.

This likely requires:

- stronger process verification
- failure localization
- or carefully designed branching heuristics

Therefore this is explicitly **not MVP scope**.

---

# 25. First Implementation Milestones

## Milestone 1 — Search

Implement:

- configurable sampling arms
- initial uniform rollout generation
- scalar scoring
- adaptive allocation
- resumable output
- token accounting

Deliverable:

```text
search_results.parquet
```

---

## Milestone 2 — Training-data builder

Implement:

- per-problem reward normalization
- trajectory selection
- teacher-forced entropy computation
- response masking
- entropy weights
- processed dataset serialization

Deliverable:

```text
train_entropy.parquet
```

or equivalent Hugging Face Dataset.

---

## Milestone 3 — Custom trainer

Implement:

- LoRA
- signed advantage loss
- entropy weighting
- length normalization
- optional KL anchor
- metrics/logging

Verify with unit tests that:

```text
positive advantage increases token probability
negative advantage decreases token probability
zero-weight token contributes no decision gradient
prompt tokens contribute no gradient
```

---

## Milestone 4 — Evaluation

Implement:

- deterministic pass@1 evaluation
- stochastic pass@k evaluation
- base vs trained comparison
- per-problem breakdown
- generated-token accounting

---

## Milestone 5 — Baselines

Run:

```text
base
successful SFT
positive-only
binary signed
graded signed
graded signed + adaptive search
```

---

# 26. Unit Tests Worth Adding

At minimum:

### Advantage sign test

Synthetic two-token model:

```text
advantage > 0
```

must increase probability of the selected token after one optimizer step.

```text
advantage < 0
```

must decrease it.

### Entropy mask test

A token with:

```text
weight = 0
```

must not contribute to decision loss.

### Response mask test

Prompt tokens must never receive the decision objective.

### Length normalization test

Duplicating irrelevant zero-weight tokens must not alter the effective loss.

### Adaptive allocator test

Higher-performing sampling configs should receive more of the remaining budget while preserving the configured exploration fraction.

### Resume test

Interrupted generation should restart without duplicating already completed `(problem_id, config_id, seed)` jobs.

---

# 27. Configuration Example

```yaml
model:
  name: Qwen/Qwen2.5-1.5B-Instruct
  max_seq_length: 4096

search:
  initial_samples_per_config: 16
  total_samples_per_problem: 256
  exploration_fraction: 0.20
  allocation_temperature: 0.5

  configs:
    - temperature: 0.35
      top_p: 0.90

    - temperature: 0.50
      top_p: 0.95

    - temperature: 0.70
      top_p: 0.95

    - temperature: 0.80
      top_p: 1.00

    - temperature: 1.00
      top_p: 0.95

    - temperature: 1.15
      top_p: 0.98

    - temperature: 1.30
      top_p: 1.00

    - temperature: 0.85
      top_p: 0.98
      repetition_penalty: 1.05

selection:
  max_correct_per_problem: 8
  max_near_correct_per_problem: 16
  max_hard_negatives_per_problem: 32
  max_low_reward_negatives_per_problem: 4

entropy:
  threshold: null       # determine sensible initial default experimentally
  scale: 0.25
  mode: sigmoid

training:
  lora_rank: 32
  lora_alpha: 64
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj

  learning_rate: 2.0e-5
  batch_size: 2
  gradient_accumulation_steps: 4
  epochs: 1
  max_grad_norm: 0.1

  kl_coef: 0.0          # start simple; add KL once core loss is validated

evaluation:
  pass_k:
    - 1
    - 4
    - 16
    - 64
    - 256
```

These are starting points, not sacred hyperparameters.

---

# 28. Acceptance Criteria for the MVP

The implementation is successful if it can:

1. Generate rollouts using multiple decoding configurations.
2. Adaptively spend additional rollout budget.
3. Score every trajectory with a scalar reward.
4. Select useful positives, near-positives, and hard negatives.
5. Compute token entropy under the frozen generating model.
6. Build entropy weights.
7. Train LoRA with signed per-trajectory advantages.
8. Evaluate pass@1 and pass@k.
9. Run the main baselines from the same data.
10. Report rollout-token and training-cost statistics.
11. Resume both search and preprocessing after interruption.
12. Reproduce a complete run from configuration files.

---

# 29. Most Important Experimental Result

The single most informative plot/table is:

```text
x-axis:
base-model pass@256 or search success rate

y-axis:
post-training pass@1 improvement
```

This tells us whether rare capabilities discoverable by search can actually be compressed into reliable behavior.

Another useful quantity:

```text
compression ratio =
post-training pass@1 / pre-training pass@1
```

but handle near-zero denominators carefully.

Also compare against:

```text
number of successful trajectories found
```

and:

```text
maximum reward found during search
```

---

# 30. Codex Priorities

When making implementation decisions, optimize in this order:

1. **Correctness of the custom training loss**
2. **Reproducibility**
3. **Reliable search/resume behavior**
4. **Accurate reward + metric tracking**
5. **Generation throughput**
6. **Training throughput**
7. **Code elegance**

Do not sacrifice experimental correctness for premature optimization.

---

# 31. Summary

Build the simplest possible system that tests this hypothesis:

> If a model can occasionally solve a difficult task under sufficiently diverse search, we may be able to identify the useful trajectories and uncertain decisions, then distill them into a cheap LoRA so the same capability appears much more often at ordinary inference time.

The MVP is:

```text
adaptive multi-temperature/top-p search
        +
graded scalar rewards
        +
per-problem signed advantages
        +
base-model token entropy
        +
entropy-weighted LoRA training
        +
pass@1-focused evaluation
```

Do this before implementing tree search or online RL.

The highest-value result is not beating GRPO on a broad benchmark immediately.

The first goal is demonstrating, cleanly and reproducibly, that:

```text
very low pass@1
+
nonzero pass@k
        ↓
offline search + distillation
        ↓
much higher pass@1
```

with substantially less rollout generation than a comparable online-RL training regime.
