# Paper 1 research and claim boundaries

## Venue

The [AIxDKE submission page](https://www.aixdke.org/submission) specifies a six-page regular paper in double-column IEEE format, including figures and references. The [conference overview](https://www.aixdke.org/) explicitly includes decision making and multimodal interaction. The [dates page](https://www.aixdke.org/dates) lists the extended submission deadline as 3 November 2026 PT. The paper should foreground the action-selection model and its measured comparison, with accessibility as a secondary demonstration.

## Position relative to prior work

- [SLURP](https://aclanthology.org/2020.emnlp-main.588/) already provides scenario/action annotated spoken requests, paired real audio and text, and published spoken-language-understanding baselines. The paper must not present action labeling of SLURP as a new dataset or imply that direct speech-to-intent is a new idea.
- [Villatoro-Tello et al., ICASSP 2023](https://www.fit.vut.cz/research/group/speech/public/publi/2023/villatoro-tello_icassp2023_10095168.pdf) already compare text, acoustic, and lattice representations on SLURP (doi:10.1109/ICASSP49357.2023.10095168). The distinguishing experiment here is a **controlled spiking versus matched nonspiking decision core**, including calibration, rejection, paired acoustic/ASR input paths, and an assistive robustness check.
- [Neftci et al., IEEE SPM 2019](https://doi.org/10.1109/MSP.2019.2931595) provides the established surrogate-gradient method; LIF neurons and surrogate gradients should be described as established components, not invented by this paper.
- [A Spiking Neural Network Action Decision Method Inspired by Basal Ganglia (2024)](https://doi.org/10.1016/j.procs.2024.11.017) studies spiking action decision in reinforcement-learning environments. Our study is supervised bounded action choice from speech representations, with calibration and abstention; no general first-ever SNN decision claim is supportable.
- [Radford et al., ICML 2023](https://proceedings.mlr.press/v202/radford23a.html) is the Whisper source. The [official model card](https://github.com/openai/whisper/blob/main/model-card.md) lists tiny at about 39 million full-model parameters. Its frozen encoder is a substantial upstream model; decision-core parameter counts must be identified separately from full pipeline size and computation.
- [Guo et al., ICML 2017](https://proceedings.mlr.press/v70/guo17a.html) motivates temperature scaling. [Geifman and El-Yaniv, NeurIPS 2017](https://papers.neurips.cc/paper_files/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html) motivates selective classification. Our development-set threshold is an empirical operating point, **not** a guaranteed test-error bound.
- [Turrisi et al., Interspeech 2021](https://www.isca-archive.org/interspeech_2021/turrisi21_interspeech.pdf) describes EasyCall: 21,386 Italian recordings from 31 dysarthric and 24 control speakers, with phone commands and near-sounding noncommands. It is a legitimate voice-accessibility stress test but does not establish live assistive performance.

## Honest contribution statement

The defensible contribution is an offline, reproducible comparison of a recurrent spiking action-selection core with an equal-parameter conventional recurrent core for 60 bounded actions; a near-parameter-matched GRU and linear probe test stronger alternatives. It tests both acoustic and recognized-text input paths, quantifies confidence and selective execution, and checks the action interface in a speaker-disjoint voice phone-control task. Results may favor the spiking core, the conventional core, or neither on different measures. The study must present that empirical tradeoff rather than force a favorable efficiency claim.

## Key limitations to retain

The primary dataset is utterance-level SLURP, not continuous streaming or a timed conversational interaction. Decision computation after a complete clip is distinct from time needed to hear the clip. The acoustic branch uses a frozen pretrained Whisper encoder, so its classifier is not a stand-alone tiny audio system. The EasyCall branch has fixed prompts and labels of intended phrases, not independent verification of spoken content or a live device trial. No physical energy, edge hardware, medical benefit, or performance for blind users has been measured. Initial single-seed held-out scores were inspected during development, so a strictly blind test claim would be inaccurate.
