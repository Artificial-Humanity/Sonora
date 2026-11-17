# Step 2c QC floor implementation plan

> Execute with `subagent-dev` and `code-review` under WORKFLOW.md.

**Goal:** Retain all 90 Arm Q speakers with at least 233 hard-QC-passing clips each.
**Architecture:** Change the existing admission floor. Preserve hard QC, all passing clips,
the deterministic split, and ten validation clips per speaker.
**Tech stack:** Existing Python corpus tools and pytest.
**Spec:** Owner accepted MIN_QC=233 after all 20 listening-probe clips received core verdicts.

## Constraints and review focus

Natural vocal buzz is accepted, not a reason to discard these voices. Hard clip QC remains.
The minimum implies 223 training clips plus ten validation clips. Do not downsample speakers
to that minimum. No training run is authorized. Execute source from the repository.

## Task 1: Change the floor and prove its boundary

- [ ] In tests/test_render_step2_corpus.py, exercise cmd_build_corpus with the real floor:
      233 passing clips per speaker succeeds, writes all 90 speakers, 223 train/10 val each,
      and provenance min_qc=233; one speaker at 232 refuses before output writes.
- [ ] Observe the new boundary test fail against the old floor.
- [ ] Set MIN_QC=233 in scripts/tools/render_step2_corpus.py; update the obsolete docstring
      to point to the constant. Update the existing refusal assertion. Keep unrelated 310s.
- [ ] Run focused tests and the repository CI pytest command, commit with Sonya's trailer.
- [ ] Fresh subagent review, remediate findings through fresh subagents, then PR/JARVIS review.

## Task 2: Assemble and measure the accepted corpus

- [ ] After merge, assemble q_synth_22k from the existing raw corpus with build-corpus.
- [ ] Recompute mel statistics and speaker HNR with the registered settings; pin outputs.
- [ ] Replace the Q config's placeholder statistics through review and rerun staged gates.
- [ ] Record corpus counts, accepted natural buzz, and the amended density in FerroState
      and private notes. Prepare the next step without starting training.
