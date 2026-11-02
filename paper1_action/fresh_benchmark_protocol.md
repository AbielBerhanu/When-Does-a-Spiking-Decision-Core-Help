# Prospective Speech-MASSIVE French replication protocol

Recorded before opening or evaluating any Speech-MASSIVE example or label.

Access amendment, also recorded before opening any example: the official test repository returned HTTP 401 to unauthenticated access, and the authors' direct archive required SharePoint login. Use the official `validation` partition as the untouched final evaluation set. Derive an internal development subset from the official `train` partition with a fixed random seed of 20261101, stratified by intent, holding out 10% of training items. Train on the other 90%. Do not evaluate or tune on official validation before all model runs are fixed. Report this as a held-out validation benchmark, never as the official test split.

## Purpose and dataset

Use the official French (`fr-FR`) Speech-MASSIVE full train and validation partitions as a new speech collection for an independent acoustic-input ranking check. Its intent ontology descends from MASSIVE/SLURP, so this is a new-language and new-recording replication, not a fully independent semantic ontology. Use only recordings marked valid by the dataset. Report exact included counts and any excluded items.

## Fixed model choices

- Frozen multilingual Whisper `tiny` audio encoder, int8 CPU; process each clip at its actual duration and uniformly pool its valid encoder states to 32 vectors of width 384. Normalize with training-split statistics only.
- Compare the same 64-unit spiking LIF, plain tanh, leak-matched tanh, and 47-unit GRU decision cores used in the SLURP study. Keep recurrent initialization gains, spike activity penalty, and readout unchanged. Use the dataset's official 60 intent IDs as the bounded actions.
- Seeds 11, 13, 17, 19, and 23. AdamW, learning rate 0.0015, batch size 128, weight decay 0.001, norm-one clipping, at most 24 epochs, early stopping after five validation NLL non-improvements. Select each seed's checkpoint by minimum validation NLL. Do not change these choices after viewing test results.
- Fit each model's temperature and 10% empirical selective-error threshold on validation data only. Compare unthresholded test accuracy and macro F1, calibrated NLL and ECE, selective coverage/error, and area under the test risk-coverage curve. Report sample standard deviations across seeds and a paired bootstrap that resamples both seeds and utterances. Do not claim a guaranteed 10% test error.

## Interpretation

The replication checks model ranking across a new speech collection. It does not establish live response latency, energy savings, a clinical benefit, or independence from SLURP-derived semantics. A finding that the SNN does not beat leak-matched tanh or GRU will be reported as such.

## Reviewer-requested French recognized-word extension (fixed before ASR outcomes)

Recorded on 2026-11-01 before running French ASR or inspecting its action scores. Keep the 10,333/1,149/2,031 fit/development/held-out split above and the same five seeds. Use the corpus `utt` field as the reference prompt; it is an intended utterance, not an independently verified transcription of the recording. Transcribe the validated internal-development and official-validation recordings using frozen multilingual Faster Whisper `tiny`, int8 CPU, forced French, beam size one, no VAD, and no previous-text conditioning. Do not run ASR on the fit recordings for this primary extension.

Train the same four text decision cores on the 10,333 fit-set reference prompts with the SLURP text recipe: lowercase Unicode word tokenization, maximum 40 tokens, vocabulary capped at 5,000, 48-dimensional embeddings, AdamW at 0.002, batch size 128, at most 18 epochs, and internal-development reference-prompt NLL checkpoint selection. Evaluate the fitted checkpoints on actual ASR hypotheses from internal development and held-out official validation. Fit a new scalar temperature and empirical 10% deferral threshold on internal-development ASR predictions only. Report held-out action accuracy, macro F1, ECE, selective risk/coverage, AURC, and five-seed paired hierarchical intervals for spiking minus leak-matched tanh and GRU. Include a word/bigram TF--IDF logistic baseline trained on the same fit prompts and calibrated on internal-development ASR hypotheses. Report prompt-referenced WER as a noisy recognition diagnostic. No hyperparameter tuning or model selection uses the official-validation action outcomes.
