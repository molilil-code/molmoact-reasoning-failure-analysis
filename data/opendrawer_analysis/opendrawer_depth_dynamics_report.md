# OpenDrawer depth-change dynamics classification

This retrospective classification uses only the complete `feat_depth_change_ratio_prev` trajectory. Operational onset is not used.

Low-change threshold: `0.1`.

| class | episodes | low-change fraction | longest low run | turning rate | total variation |
| --- | ---: | ---: | ---: | ---: | ---: |
| oscillation/burst-like | 4 | 0.301 | 0.100 | 0.647 | 18.48 |
| stagnation-like | 4 | 0.540 | 0.165 | 0.651 | 9.73 |

The stagnation-like group is characterized primarily by a larger low-change fraction and longer low-change runs. The oscillation/burst-like group is characterized primarily by larger total variation. Turning rate overlaps between the groups and should not be treated as a standalone separator. These thresholds are exploratory and were not cross-validated.
