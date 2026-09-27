# Twelve-hour window closed; outcomes remain mixed

The original window ended **2026-09-23 01:15:12 CDT**. The final experiment had
already exited successfully at00:24:31. Checks after the deadline only verified
terminal files/processes/resources, archived findings and attempted report lookup;
no training, data collection, inference, rescoring or budget extension occurred.

- The preregistered layer candidate did not pass: K4-SVAE48 increased noisy state
  error by45.675% versus L12-SVAE48 on the complete60scene confirmation. Its
  secondary goal error was lower by36.265%, a tradeoff rather than a joint win.
- Goal-auxiliary confirmation remains incomplete at56/60 (20/20/16), with original
  timeout/failed exit records intact and no scores. It is not a successful result.
- The added official-weight Wan comparison and equal-noise controls completed
  and passed independent saved-artifact checks. Noise training largely repairs
  Wan's previous noisy readout collapse, so that collapse alone is not evidence
  of irretrievable information loss.
- After equal-noise training, compact DINO PCA48 has43.88% lower noisy short-state
  E than native Wan in the exposed60scene cohort. SVAE48 is close. Raw DINO has
  the lowest aggregate E but63.91% worse gripper error than native Wan. Compact
  features' gripper improvements remain uncertain; all regressions are retained.
- The DINO path has no trained pixel decoder, so this is not a complete RAE
  reproduction. No new closed-loop policy benefit has been established.

The minimal Wan input-range doc correction and independent-confirmation draft
are prepared locally. The draft proposes a new complete60scene cohort, all
frozen routes and separate translation/gripper endpoints. It estimates about
three hours, subject to environment/resource checks and a new fixed budget.
It has not launched. The expired56/60 and previously completed80 are not reused.

Public report lookup at01:17CDT again failed: **Sites project not found**.
The current report is local only; no replacement site or successful deployment
is claimed. Future timer ticks do not extend the expired experiment window.
