# Step 8b Findings: Interference-Aware Mitigation Autoencoder (Fig. 13-15)

## Summary
Trained the mitigation AE WITH tone interference present during training
(BASE-PAPER FACT: N_SYMBOLS=1024, SIR range -10 to 30dB), the paper's
proposed fix for denoising-only training's failure (Step 8a).

## Result: mitigation performs WORSE than doing nothing, and worsens with Eb/N0
| Eb/N0 | No mitigation | Mitigated |
|---|---|---|
| 0dB | 0.339 | 0.419 |
| 3dB | 0.335 | 0.468 |
| 6dB | 0.304 | 0.496 (~chance) |

## Diagnostic: ruled out overcorrection, confirmed MSE/BER disconnect
Direct MSE check (OUR EXPERIMENTAL RESULT): AE output is ~7x closer to
the true clean signal than raw input, at BOTH Eb/N0=0dB (17.95->2.62)
and Eb/N0=6dB (12.08->1.71) -- reconstruction genuinely improves with
Eb/N0 as expected. Yet BER gets WORSE with higher Eb/N0. This rules out
overcorrection and confirms a direct disconnect between MSE and BER.

## Interpretation
This closely parallels the paper's OWN documented finding for the
DENOISING-ONLY case (Fig.12: "despite achieving a low MSE... the
mitigation was ultimately ineffective... functioned similarly to a
matched filter"). The paper treats interference-aware training as the
fix for this phenomenon. Our result suggests the same MSE/BER
disconnect may persist even with interference-aware training, at least
for our (unspecified-by-paper) architecture, window length, and mixed
SIR/SNR training distribution. Root cause hypothesis (UNTESTED): MSE
does not penalize errors in the specific decision-critical component
(sign of the derotated imaginary part) more than any other error,
so overall energy reduction does not guarantee correct bit decisions.

## Status
Reproduction: NOT SUCCESSFUL for this specific configuration. This is
treated as a genuine, evidence-backed finding rather than an
unresolved bug -- three specific, diagnosed issues were found and
either fixed (mode collapse) or characterized (window-length
sensitivity, MSE/BER disconnect) across Step 8a/8b, none dismissed
without investigation.

## Candidate for Research Opportunity Report
Strong candidate: "MSE-based training objectives for interference
mitigation may not reliably optimize the downstream metric that
actually matters (BER), even when training includes interference" --
a more general and more strongly-evidenced version of a limitation the
paper itself only partially acknowledges.