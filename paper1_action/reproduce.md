# Reproducing the Paper 1 action-decision study

The paper's target is **action selection**. `paper1_draft.tex` and the top-level `results/` directory contain an earlier exploratory keyword study and are not the Paper 1 results.

## Inputs and integrity

- SLURP annotations: [official repository](https://github.com/pswietojanski/slurp), `data/slurp_repo/dataset/slurp/{train,devel,test}.jsonl`.
- SLURP real-audio archive: [official Zenodo record](https://zenodo.org/records/4274930), `data/slurp_real.tar.gz`; verified MD5 `9efc0f058ced47bf5131c7cb2cade513` against the record. The scripts deterministically use the first listed real recording per utterance.
- EasyCall: [author corpus](http://neurolab.unife.it/easycallcorpus/), `data/EasyCall.zip`; local SHA-256 `fbdc6fa95d747e9f394b951f525e60817822878b904e28358323d64893c40a06`. The 67 prompt ontology is in `prepare_easycall.py` and was checked against the original 21,386 files. The command/noncommand distinction follows Table 4.2 in the [author thesis](https://iris.unife.it/retrieve/e309ade5-9c83-3969-e053-3a05fe0a2c94/3.pdf), with the archive's `Salva` spelling replacing the table's `Salve`.
- Frozen English audio encoder and ASR: Faster Whisper `tiny.en`, int8 CPU, with local model download in `data/asr_models`. It is an upstream pretrained component, separate from the 36,476-parameter recurrent decision core.
- Fresh acoustic benchmark: [Speech-MASSIVE](https://github.com/hlt-mt/speech-massive) French `fr-FR` train and validation Parquet shards from `FBK-MT/Speech-MASSIVE`. The public test files required authentication during this study, so the official validation set is the untouched final evaluation set. `fresh_benchmark_protocol.md` records this choice before any outcome was inspected. The intent ontology derives from MASSIVE and SLURP; the recordings, speakers, and language differ.

## Split and label rules

SLURP uses official 11,514/2,033/2,974 utterance train/development/test splits and 60 `scenario_action` choices. Six test utterances have a train sentence identical after lowercasing and trimming; no utterance ID or selected recording crosses splits. This is an utterance split, not a speaker-disjoint claim.

EasyCall uses a fixed 33/11/11 speaker train/development/test split, giving 13,115/4,118/4,153 recordings and 36 choices: 35 phone actions plus explicit `no_action`. The split is speaker-disjoint and covers both dysarthric and control speakers. Initial single-seed test results were inspected during development; the paper discloses that the holdout is not strictly blind. Speech-MASSIVE uses only recordings marked valid; its official train is divided 90/10 for fit/internal development with stratification and seed 20261101, and its official validation is reserved for final evaluation.

## Commands

Run from the project root using the local `.venv` with dependencies in `paper1_action/requirements.txt`:

```sh
.venv/bin/python paper1_action/prepare_slurp_audio.py
.venv/bin/python paper1_action/prepare_slurp_embeddings.py
.venv/bin/python paper1_action/prepare_easycall.py
.venv/bin/python paper1_action/transcribe_slurp.py
.venv/bin/python paper1_action/transcribe_slurp.py --split train
.venv/bin/python paper1_action/prepare_speech_massive.py
.venv/bin/python paper1_action/run_speech_massive.py
.venv/bin/python paper1_action/transcribe_speech_massive.py
.venv/bin/python paper1_action/train_speech_massive_text.py
.venv/bin/python paper1_action/evaluate_speech_massive_asr.py
.venv/bin/python paper1_action/run_bow.py
.venv/bin/python paper1_action/run_extra_slurp_seeds.py
.venv/bin/python paper1_action/run_asr_trained_batch.py
```

Both ASR steps write resumable JSONL files. For SLURP seeds `11`, `13`, and `17`, run `run_text.py`, `run_slurp_audio.py --representation whisper --front plain`, and `evaluate_asr_text.py` for each of `ann`, `leaky_ann`, `snn`, and `gru`. `run_extra_slurp_seeds.py` repeats those fixed comparisons for seeds `19` and `23`; run it only after the first three are complete. `run_asr_trained_batch.py` then trains all four text cores across five seeds on actual ASR training hypotheses rather than reference words. It requires the completed training ASR file. `run_bow.py` is the nonrecurrent word/phrase TF-IDF logistic comparator. `run_speech_massive.py` executes the prospectively fixed five-seed French comparison. The same actions, train/development/evaluation divisions, and calibration method are used within each comparison. SNN and tanh models have exactly equal trainable parameter counts; GRU counts differ by at most 3%. `run_linear_probe.py` is a nonrecurrent acoustic control. For the secondary accessibility stress test, run `run_easycall.py --front conv --epochs 35` for both `--kind ann` and `--kind snn`, with seeds `11`, `13`, and `17`, then run `analyze_easycall.py`. `analyze_results.py` produces paired utterance and hierarchical intervals; `analyze_calibration.py` produces before/after ECE and area under the risk-coverage curve. `validate_study.py` checks paired labels, split integrity, and main result files. `profile_pipeline.py` should run alone after training and transcription complete to avoid shared-CPU timing distortion.

## Interpretation

SLURP oracle transcripts are a diagnostic ceiling. The recognized-text path uses actual ASR hypotheses from the same real audio as the acoustic path. In EasyCall, the prompt filename is an intended phrase label, not a manually verified transcription. Mean hidden spike rate is a neuron-activity statistic, not energy. Timing from loaded waveforms omits file I/O and the time needed to hear the full utterance.

The per-recording encoder seconds stored during feature-cache generation were measured while other jobs used the CPU; they are marked `unisolated` in result JSON and must not be cited as latency. The isolated 100-recording measurement in `slurp_pipeline_cpu_timing.json` supplies the paper's limited CPU timing comparison.

## Paper figures and citations

`plot_accuracy_comparison.py` creates the color action-accuracy chart from the five-seed summary. `plot_risk_coverage.py` creates the vector risk--coverage chart and a PNG preview from held-out prediction files. `analyze_calibration.py` creates a before/after reliability diagram. The manuscript also contains self-contained LaTeX versions of all three charts so the desktop document editor can render them when it cannot load companion files. `refresh_inline_charts.py` updates the LaTeX accuracy and risk figures from the result files. Include the three figure files with the manuscript source in any external build.

The two Banad-group references in the paper are Ahmadvand et al. (2024, doi:10.1088/2634-4386/ad8c79) and Ahmadvand et al. (2025, doi:10.1038/s41598-025-28344-4). Berhanu et al. (2026, doi:10.1109/AIxDKE67294.2026.00035) is the author's previous hardware survey; it is cited only for scope and does not supply an energy result for this software model.

## Reviewer-response leak control

Run `run_slurp_audio.py`, `run_text.py`, and `evaluate_asr_text.py` with `--kind leaky_ann` for seeds 11, 13, and 17, using the same epoch, batch, representation, and front-end options as the existing tanh runs. This equal-parameter control uses an analog state leak of 0.9 and the spiking core's recurrent initialization gain of 0.25. `analyze_results.py` now includes it in the summary and reports both utterance-only and seed-plus-utterance hierarchical intervals. `plot_accuracy_comparison.py` regenerates the four-core color chart. The new control was added after test-set inspection, so its scores remain exploratory rather than an independent confirmation.

The frozen-audio encoder does not pool across a 30-second padded window: `prepare_slurp_embeddings.py` passes each variable-length waveform directly to Faster Whisper feature extraction and pools only the returned encoder states. A synthetic 2.72-second waveform yielded 273 feature frames and 137 encoder states before the 32-step pooling operation. This audit clarified the original feature extraction; it did not change the acoustic cache or linear-probe fit. That is why the earlier and revised linear-probe numbers are identical.

## French recognized-word extension

The three additional French commands above transcribe the 1,149 internal-development and 2,031 held-out official-validation recordings, fit five reference-prompt models of each recurrent architecture plus a TF--IDF logistic control, then evaluate real ASR words. Development hypotheses fit temperature and rejection threshold; held-out hypotheses supply final metrics. French word error rate uses intended prompts as references, which are not independently verified transcriptions. The exact extension was recorded in `fresh_benchmark_protocol.md` before ASR outcomes were inspected. The TF--IDF baseline has the same calibration, selective-risk, and action-cost analyses as the recurrent models.
