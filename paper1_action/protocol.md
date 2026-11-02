# Paper 1 action-selection study protocol

Status: working protocol for an offline algorithm study. Initial single-seed SLURP and EasyCall test results were inspected while developing the pipeline and choosing the frozen speech encoder. These pilot results cannot be represented as a strictly untouched final holdout. Subsequent multi-seed results use the stated fixed choices, but the paper must disclose this development history. The old Speech Commands experiments are exploratory work only.

## Research question

Can a compact recurrent spiking decision network choose among bounded actions from an input state with useful accuracy, calibrated confidence, and selective execution? Compare the same model family against a parameter-matched conventional recurrent network. Speech is the principal input example, evaluated through (a) direct acoustic features and (b) text recognized by ASR; corpus transcripts are an explicitly labeled oracle ceiling. Assistive phone control is a secondary stress test of the decision core and its no-action behavior.

## Benchmarks

1. **SLURP** (English, multi-domain assistant): official utterance-level train/development/test partitions, with `scenario_action` as the action label. Use one deterministic real recording per utterance for the main paired audio/text comparison; do not treat multiple recordings of one utterance as independent test samples. Exclude synthetic audio. 60 actions occur in training; all test action labels occur in training. Report accuracy, macro-F1, NLL, Brier, ECE, and selective risk. Classifier confidence can trigger `defer` but is not a ground-truth SLURP action.
2. **EasyCall, secondary stress test** (Italian, voice-controlled contact app): original data from [the dataset authors](http://neurolab.unife.it/easycallcorpus/). Participants include people with dysarthria and controls. Map the documented executable command phrases to phone actions; map the 30 close-sounding noncommands to `no_action`. Publish the mapping and its rationale. Partition by speaker, stratifying on dysarthria/control and severity where feasible; no speaker may cross splits. Report dysarthric and control performance separately. A false execution on a `no_action` sample is a prespecified safety error. A command missed by abstention/no-action is also reported. This benchmark supports a robustness discussion; it does not change the paper's central claim to medical or clinical performance.

## Decision model and comparators

- Matched spiking LIF and nonspiking tanh recurrent cores receive the same preprocessed input, action catalog, splits, training budget, and calibration method. Include a near-parameter-matched GRU and a nonrecurrent linear probe as stronger conventional controls. The spiking and tanh initial recurrent gains and the spiking activity penalty differ; a favorable result cannot be attributed solely to spiking dynamics.
- The decision head returns logits over allowed actions, then a validation-calibrated probability distribution. A validation-chosen threshold maps low-confidence predictions to `defer`; EasyCall's `no_action` is an ordinary explicit action. Distinguish the two.
- Choose architecture and threshold from training/development data only. Following the disclosed pilot inspection, the multi-seed comparisons keep these choices fixed. Report three seeds, seed variation, and bootstrap confidence intervals grouped by utterance or speaker as appropriate; do not describe the test set as strictly untouched.
- Profile CPU software inference, parameter count, and spiking activity. Report model-only and full-pipeline time separately. Spike count is an activity proxy, not a physical energy measurement.

## Boundaries

- Neither dataset records a live user interacting with this model. EasyCall's prompt labels are intended commands, not verified exact transcriptions of what participants pronounced.
- Direct audio versus transcript input is not automatically a fair end-to-end comparison: the text pipeline must account for ASR computation and errors. Oracle transcript results are an upper-bound diagnostic, not a deployable text-system result.
- No claim about edge hardware, energy savings, clinical efficacy, or performance for blind users. No claim that Jev's unpublished internal model was reproduced. The similarity is the state-to-bounded-action decision interface.

## Primary sources

- Bastianelli et al., [SLURP](https://aclanthology.org/2020.emnlp-main.588/), EMNLP 2020; [data/annotations](https://github.com/pswietojanski/slurp), [real audio](https://zenodo.org/records/4274930).
- Turrisi et al., [EasyCall corpus](https://www.isca-archive.org/interspeech_2021/turrisi21_interspeech.pdf), Interspeech 2021.
- Neftci et al., [Surrogate Gradient Learning in Spiking Neural Networks](https://doi.org/10.1109/MSP.2019.2931595), IEEE Signal Processing Magazine 2019.
- Guo et al., [On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html), ICML 2017.
