"""The frozen pilot's train/dev/test episode lists must remain disjoint."""

from research.rae.make_episode_splits import episode_splits


def test_frozen_beat_block_assignment():
    splits = episode_splits("beat_block_hammer", "openwam-rae-20261023:")
    assert splits["dev"] == [30, 16, 19, 7, 28]
    assert splits["test"] == [20, 5, 36, 31, 8]
    assert len(splits["train"]) == 40
    assert set(splits["train"]) | set(splits["dev"]) | set(splits["test"]) == set(range(50))
    assert not (set(splits["train"]) & set(splits["dev"]))
    assert not (set(splits["train"]) & set(splits["test"]))
    assert not (set(splits["dev"]) & set(splits["test"]))
