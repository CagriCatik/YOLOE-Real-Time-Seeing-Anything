"""Frame-level perception ground truth and explicit N/A semantics."""

from types import SimpleNamespace

import numpy as np

from adascope.perception_ground_truth import (
    ExpectedPerceptionFrame, PerceptionAcceptance, score_perception,
)


def analysis(**changes):
    base = dict(
        image=np.zeros((100, 200, 3), np.uint8),
        driving_area_src=np.float32([[50, 90], [150, 90], [120, 10], [80, 10]]),
        boundaries=[20.0, 60.0, 100.0], lanes_rel=[object(), object()],
        ego_lane_pos=1, vehicles=[], occupancies=[],
    )
    base.update(changes)
    return SimpleNamespace(**base)


def expected(**changes):
    base = dict(frame=4,
                driving_area=((50, 90), (150, 90), (120, 10), (80, 10)),
                boundaries_bev=(20, 60, 100), lane_count=2,
                ego_lane_position=1)
    base.update(changes)
    return ExpectedPerceptionFrame(**base)


def test_perfect_geometry_passes_all_annotated_thresholds():
    score = score_perception((expected(),), {4: analysis()}, PerceptionAcceptance())
    assert score.perfect
    assert score.means()["driving_area_iou"] == 1.0
    assert score.means()["lane_count_accuracy"] == 1.0
    assert score.means()["vehicle_lane_accuracy"] is None


def test_missing_metrics_are_na_not_automatic_successes():
    truth = ExpectedPerceptionFrame(frame=4, lane_count=2)
    score = score_perception((truth,), {4: analysis()}, PerceptionAcceptance())
    assert score.perfect                         # the one measured metric passes
    assert score.checks()["driving_area_iou"] is None
    assert "N/A" in score.as_text()


def test_wrong_lane_count_fails_acceptance():
    score = score_perception((expected(),),
                             {4: analysis(lanes_rel=[object()])},
                             PerceptionAcceptance())
    assert not score.perfect
    assert score.checks()["lane_count_accuracy"] is False


def test_unprocessed_annotated_frame_is_a_failure():
    score = score_perception((expected(),), {}, PerceptionAcceptance())
    assert not score.perfect and score.missing_frames == [4]


def test_boundary_matching_is_one_to_one_and_tolerance_limited():
    score = score_perception(
        (expected(boundaries_bev=(20, 60, 100), driving_area=(),
                  lane_count=None, ego_lane_position=None),),
        {4: analysis(boundaries=[22, 58, 160])},
        PerceptionAcceptance(boundary_tolerance_px=5,
                             boundary_recall_min=0.5,
                             boundary_mae_max_px=3))
    assert score.means()["boundary_recall"] == 2 / 3
    assert score.means()["boundary_mae_px"] == 2.0
    assert score.perfect
