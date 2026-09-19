"""Adapter around Role 1's deployed model (model_deployment_package/predictor.py).

KNOWN GAPS -- read before trusting this predictor's numbers:

1. Feature mismatch. Role 1's model was trained on post-metadata features
   (duration, hashtags, mentions, upload timing, ...). It has NO signal for
   `absurdity`, `irony`, `relatability`, `trend_relevance`, `topic`, `humor`,
   `format`, or `hook` -- the exact traits Role 2's evolutionary loop
   mutates and forms hypotheses about. In practice this means predicted
   fitness only reacts to `video_length`, `caption_length`, and
   `audio_strategy`; mutating tone/content traits will not move the
   prediction at all. This is a data-contract gap for Role 1 to close
   (e.g. engineered absurdity/irony features in Calcifer), not a bug here.

2. Scale mismatch. The model's raw output is NOT a 0-1 fitness score --
   empirically it lands roughly in [1.2, 4.0] for realistic genome-derived
   inputs (see `_RAW_SCORE_LOW`/`_RAW_SCORE_HIGH` below), and can swing far
   outside that (observed: -1000 to +2800) for feature combinations outside
   its training distribution. This adapter (a) always sends fixed, safe
   defaults for fields the genome has no signal for, to avoid the
   extrapolation blowups, and (b) linearly rescales the realistic range
   into [0, 1] so it satisfies Role 2's FitnessPrediction contract. The
   rescaling bounds are an empirical guess, not a calibration Role 1 has
   confirmed -- revisit once Role 1 documents what the training target
   actually represents (raw engagement? log(views)? a composite score?).
"""

from __future__ import annotations

import sys
from pathlib import Path

_MODEL_PACKAGE_DIR = Path(__file__).resolve().parents[2] / "model_deployment_package"
if str(_MODEL_PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(_MODEL_PACKAGE_DIR))

from memevolution.models.genome import MemeGenome
from memevolution.models.prediction import FitnessPrediction

# Empirically observed range of raw model output for realistic
# genome-derived inputs (duration swept 3-60s, audio_strategy toggled,
# caption_length pinned near the seed genome's value, everything else at
# its safe default). See the module docstring, gap #2.
_RAW_SCORE_LOW = 1.0
_RAW_SCORE_HIGH = 4.5


def _genome_to_features(genome: MemeGenome) -> dict:
    """Map the fields Role 2 actually has onto Role 1's feature schema.

    Fields Role 2 has no signal for (ads, hashtags, mentions, upload
    timing) get fixed safe defaults rather than being left to the
    predictor's own fallback -- deliberately, so behavior doesn't change
    if that fallback logic changes, and to stay inside the range this
    adapter was calibrated against.
    """
    return {
        "duration": genome.video_length,
        "is_video": 1,
        "is_ad": 0,
        "caption_length": genome.caption_length,
        "has_hashtags": 0,
        "mentions_count": 0,
        "hashtags_count": 0,
        "is_original_sound": 1 if genome.audio_strategy == "original_sound" else 0,
        "upload_hour": 12,
        "upload_day_of_week": 3,
    }


def _rescale(raw_score: float) -> float:
    span = _RAW_SCORE_HIGH - _RAW_SCORE_LOW
    normalized = (raw_score - _RAW_SCORE_LOW) / span
    return min(1.0, max(0.0, normalized))


class Role1FitnessPredictor:
    """Wraps Role 1's deployed XGBoost model behind the FitnessPredictor protocol.

    This is the real historical-data model (unlike MockFitnessPredictor),
    but see the module docstring for two known gaps: it can't see most of
    the genome's traits, and its output scale is only approximately
    normalized. Treat its numbers as directional, not precise, until Role 1
    resolves gap #1 and confirms gap #2.
    """

    def __init__(self) -> None:
        import predictor as _role1_predictor  # Role 1's deployment package

        if _role1_predictor.model is None:
            raise RuntimeError(
                "Role 1's model failed to load. Check that "
                "model_deployment_package/memetic_fitness_xgb.json exists "
                "and that xgboost/pandas are installed."
            )
        self._predictor = _role1_predictor

    def predict_fitness(self, genome: MemeGenome) -> FitnessPrediction:
        features = _genome_to_features(genome)
        raw_score = self._predictor.predict_fitness(features)
        return FitnessPrediction(fitness=round(_rescale(raw_score), 3), confidence=None)
