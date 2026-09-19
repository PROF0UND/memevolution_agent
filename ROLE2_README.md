# Role 2 — AI Agent / Evolution

This is the evolutionary decision-maker described in the main [README.md](README.md):
it generates and mutates meme genomes, asks Role 1's historical model how
promising each one looks, picks one to actually run, and later updates its
beliefs from what really happened.

```
Role 1 (historical prior)  --predicted fitness-->  Role 2 (this module)
                                                          |
                                                    generate / mutate / select
                                                          |
                                                    one real experiment
                                                          |
                                              Role 4 (TikTok) --actual engagement-->
                                                          |
                                                    Role 2 updates beliefs
```

Role 2 never trains or replaces Role 1's model, and never talks to TikTok directly.

## Architecture

```
memevolution/
├── models/
│   ├── genome.py       MemeGenome — the meme's DNA
│   ├── prediction.py   FitnessPrediction (= Prediction)
│   ├── experiment.py   Experiment, Mutation, Deployment, Observation, MemeConcept
│   └── agent.py         AgentState — persistent beliefs
├── evolution/
│   ├── mutation.py      mutate(), generate_hypothesis()
│   ├── population.py    generate_population()
│   ├── selection.py     select_candidate()
│   └── learning.py      update_state()
├── prediction/
│   ├── interface.py     FitnessPredictor protocol (Role 1 plugs in here)
│   ├── mock.py           MockFitnessPredictor — DEV STUB, not Calcifer
│   └── role1.py          Role1FitnessPredictor — adapter around Role 1's real model
├── llm/
│   └── gemini.py         GeminiConceptGenerator + MockConceptGenerator
├── persistence/
│   └── json_store.py     data/agent_state.json, data/experiments.json
├── agent/
│   └── orchestrator.py   run_generation(), record_observation(), apply_observation()
└── cli.py                `python -m memevolution {generate,status,observe}`
```

## How Role 1 connects to Role 2

Role 2 only depends on this Protocol (`memevolution/prediction/interface.py`):

```python
class FitnessPredictor(Protocol):
    def predict_fitness(self, genome: MemeGenome) -> FitnessPrediction: ...
```

`Role1FitnessPredictor` (`memevolution/prediction/role1.py`) implements it as
a thin adapter around Role 1's deployed model in `model_deployment_package/`.
It's the default for `python -m memevolution generate` (pass `--predictor
mock` to use the heuristic stub instead). No other Role 2 code — `evolution/`,
`agent/` — knows or cares which predictor is plugged in.

**Two gaps in that integration, both documented in `role1.py`'s module
docstring, and worth fixing with Role 1 before trusting this for a real
demo of the learning loop:**

1. **Feature mismatch.** Role 1's model was trained on post-metadata
   (`duration`, `caption_length`, hashtags, mentions, upload timing) and has
   no signal for `absurdity`/`irony`/`relatability`/`trend_relevance` — the
   traits Role 2 actually evolves. Only `video_length`, `caption_length`,
   and `audio_strategy` currently move the prediction; you can see this
   directly in a demo run, where candidates that only differ in
   absurdity/irony/relatability/trend_relevance come back with an identical
   predicted fitness. Closing this needs Role 1 to add features (or
   labeled data) that capture those tone/content qualities.
2. **Scale mismatch.** The model's raw output isn't a 0–1 fitness score —
   empirically it's ~1.2–4.0 for realistic inputs (and can swing to
   -1000/+2800 for feature combinations outside its training
   distribution, which is why the adapter always sends fixed, safe
   defaults for fields the genome has no signal for). `role1.py` linearly
   rescales an empirically-observed range into `[0, 1]` to satisfy
   `FitnessPrediction`'s contract; this is a guess, not a calibration Role
   1 has confirmed. Revisit once Role 1 documents what the training target
   actually represents.

## How to run the demo

```bash
pip install -r requirements.txt   # or: pip install pydantic (gemini/pytest optional)

python -m memevolution generate --seed 42     # run one generation, print the full trace
python -m memevolution status                 # show current beliefs + experiment history
python -m memevolution observe exp_001 --fitness 0.30 --views 50000 --likes 3000
                                                # record real engagement, agent updates beliefs
```

State lives in `data/agent_state.json` and `data/experiments.json` and
survives restarts. Delete `data/` to reset the agent to generation 0.

Run tests with:

```bash
pytest
```

## Swapping predictors (`role1` vs `mock`)

`python -m memevolution generate --predictor {role1,mock}` (default: `role1`)
picks which `FitnessPredictor` implementation gets passed to
`run_generation`. If Role 1 ships a new/retrained model, the model artifact
lives entirely in `model_deployment_package/` — update it there and
`Role1FitnessPredictor` picks it up automatically. If the model's input
features or output scale change, update `_genome_to_features` /
`_rescale` in `role1.py` accordingly. Nothing in `evolution/` or `agent/`
ever needs to change: they only see the `FitnessPredictor` protocol.

## How actual observations update the agent

1. Role 4 (or a human, for the demo) calls
   `apply_observation(state, experiment_id, Observation(fitness=..., views=..., ...))`
   — or `python -m memevolution observe <id> --fitness ...` from the CLI.
2. This records the observation on the persisted experiment, then computes
   `error = actual_fitness - predicted_fitness` and nudges the belief for
   each trait that was mutated in that experiment: a mutation that raised a
   trait and then beat its prediction increases belief in that trait;
   underperforming decreases it (and mirrored for traits that were lowered).
   See `evolution/learning.py::update_state` for the exact rule.
3. `Experiment.prediction_error` is available directly on any experiment
   once both `prediction` and `observed.fitness` are set — this is the
   number a future dashboard would plot as "predicted vs. actual."
4. Genomes whose observed fitness clears `SUCCESS_THRESHOLD` (0.6) are
   added to `state.successful_genomes`, becoming the parent for the next
   generation's population.

## Gemini integration

`llm/gemini.py::get_concept_generator()` returns a real
`GeminiConceptGenerator` when `GEMINI_API_KEY` is set in the environment,
and falls back to a deterministic `MockConceptGenerator` (clearly marked as
a dev stub) otherwise — so the CLI demo runs end-to-end with no API key.
Gemini only ever turns a genome into a concrete concept (title/opening/
visual/punchline/caption/audio); it never sees or changes the genome's
numeric traits, and it is never used for mutation, selection, or learning.

## Assumptions / open integration points for other roles

- **Role 1**: the real model is wired in (`Role1FitnessPredictor`), but see
  the two gaps documented above and in `role1.py` — it can't see the
  genome's tone/content traits, and its output scale is an empirical guess
  rather than a confirmed calibration. `MockFitnessPredictor` remains
  available (`--predictor mock`) as a heuristic that *does* react to
  absurdity/irony/relatability/trend_relevance, useful for demoing the
  evolutionary-loop concept independent of Role 1's current feature gap.
- **Role 4 / TikTok**: `Deployment.timestamp`/`post_id` and all of
  `Observation` are populated by whatever Role 4 builds; Role 2 exposes
  `record_observation` / `apply_observation` as the two integration points
  and makes no assumption about how the real numbers arrive (webhook,
  scraper, manual entry — any of them can call these functions).
- **Gemini schema**: `MemeConcept`'s six fields (title/opening/visual/
  punchline/caption/audio_strategy) are a reasonable MVP shape for a
  short-form video concept; extend `MemeConcept` if the content/creative
  team needs more structure (e.g. shot list, on-screen text timing).
- **Success threshold** (0.6) and **learning rate** (0.15) in
  `evolution/learning.py` are placeholders picked to be interpretable, not
  tuned against any real data — worth revisiting once real TikTok
  engagement numbers come in.
