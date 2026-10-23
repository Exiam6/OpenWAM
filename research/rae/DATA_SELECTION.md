# Bounded RoboTwin training subset

Dataset: `TianxingChen/RoboTwin2.0`, revision
`3dc3b798668feb99ac61cc9086d84cbcc3d79186`. From its 50
`dataset/<task>/aloha-agilex_clean_50.zip` archives, sort task names by
`SHA256("openwam-rae-20261023:" + task)` and take the first 10. This rule was
fixed before inspecting RAE results. The selected archives total
6,734,092,000 bytes compressed; the full clean corpus is 23,780,715,316
bytes and clean plus randomized is 414,977,048,251 bytes compressed.

| Task | Archive bytes | Archive SHA256 |
|---|---:|---|
| stack_blocks_three | 950507257 | d8b3de21979e1a89b1f8b4dbc9d95ea9622e66c1db62c4d6736be996976b0a90 |
| place_burger_fries | 689932579 | 23ebeffcb668e55ef6d560a169c3a68032017a080a8228d97da57538844de0b1 |
| hanging_mug | 735324234 | d854cbe1e70563417a05bffc9295e3b46e993287e43294f9575a5d4cbf8d2b09 |
| shake_bottle_horizontally | 483114791 | 605c099234ba3a1f471fce6d5339adc8ff816e926cac2ce782579af1f3ab555d |
| stack_bowls_two | 622234585 | e3fbb78738e195387a14e80cdfd7270db4743271a951400be2a21e8ecddb9f9a |
| open_microwave | 982663234 | 8cf14879560849a76738fb05a1842608fe79831efac625b81aa4c886edeff132 |
| beat_block_hammer | 229401324 | a135ad233bdcffff65fb636780f95ec34abc36ccf2bd0ff26ab07c9c464cc6af |
| stack_bowls_three | 1018303066 | 46ecebaaa1828c7f4c941a6761324cf445c7a66e91161d6fbf13c43a214c26ea |
| place_bread_skillet | 394729396 | f0570822c1286b7610a3bec1018f421548df0002c98b9392a0f809cf5245cfb3 |
| place_dual_shoes | 627881534 | 0cdb0e5c03a30332c928cce1c430a21da8248adc01608706487709283168f703 |

This subset supports a bounded method pilot and matched VAE/RAE training.
It cannot replace the full 50-task released-checkpoint reproduction or support
a claim about all RoboTwin tasks. Episode assignment is specified below;
demonstration hashes, normalization statistics, and evaluation scene seeds
must be finalized before any GPU run.

OpenWAM's RoboTwin `split=train` and `split=val` enumerate the same episode
files; they differ only in window starts. `make_episode_splits.py` therefore
assigns each task's 50 episodes by SHA256 of
`"openwam-rae-20261023:<task>:episode<index>"` to 40 train, 5 dev, and 5 test
episodes. It creates separate symlink-only dataset roots, verifies all archive
hashes, and writes a manifest with every episode file hash. All normalization
statistics must be fitted from the train root and reused unchanged for dev/test.

The complete episode manifest was materialized at
`/root/zifan/openwam-assets/robotwin-splits/split_manifest.json` with SHA256
`d492addcdd94f41b0c5819561e38bc0f3b0b39cfb650de15161142cd0339a1f2`.
It contains 400 train, 50 dev, and 50 test episode links with no broken
links. The archive SHA256 values above were rechecked before extraction and
again before splitting.
