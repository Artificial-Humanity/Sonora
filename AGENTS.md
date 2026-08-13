# AGENTS — Project Sonora (Training Repo)

This is the entry point for any agent or developer working on Project Sonora's training
codebase. This is an independent GitHub repo (the PyTorch training pipeline that produces the
actor model artifacts published to the `Sonora/huggingface` sibling checkout). Internal engineering notes —
architecture, current state, open decisions — live in [notes/](notes/), mapped one-line-per-file
in [notes/README.md](notes/README.md). Before starting work, read
[notes/STATE.md](notes/STATE.md) for the current state of the project and
[notes/todo.md](notes/todo.md) for the open work.

---

## Core Stack Matrix

* **Language Ecosystem:** Python-based ML training pipeline.
* **Second lane — teacher synthesis (`scripts/stages/`):** five audited TTS engines
  (`chatterbox`, `qwen`, `zonos`, `orpheus`, `moss_vg`; `vibevoice` + `dia` are set aside,
  reversible — `ref_select.SET_ASIDE`) driven by a two-pass Gemma director — per-line
  V/A/T + a register copied from the 47-label controlled lexicon
  (`scripts/assets/register_lexicon.json`), then per-engine casting/delivery from
  `scripts/assets/director_skills/<engine>.md`. Engine allocation is the three-layer
  `ref_select.ENGINE_MIX_BY_LANE` (capability veto · measured per-lane weights ·
  diversity floor). **`build_direction()` in `book_ingest.py` is the single source of
  truth for what each engine actually receives — never bypass it; unknown engines are fatal.**
  The 2026-07-25 relay audit found rich direction had been silently dropped by every engine.
  **Standing rule:** no TTS model enters the portfolio without a studied interface and a
  Gemma skill-file adapter — see [notes/tts-engine-onboarding.md](notes/tts-engine-onboarding.md),
  which also carries the known-gotchas reference. Minimum clip length is **4 s of estimated
  speech**, gating input text (not render duration).
* **Core Framework:** PyTorch & Matcha-TTS (conditional flow-matching).
* **Environment:** AMD ROCm PyTorch Docker container (optimized for Ryzen AI Max+).

---

## Integration Dependencies

* **Deployed code is a copy; the repo is the source** — see §6 and [notes/data-mirrors.md](notes/data-mirrors.md).
* This repo is a standalone PyTorch training repository for building, fine-tuning, and
  exporting voice actor models (Matcha-TTS). It is consumed by Project Prosodia
  (`ProsodiaActor`) via exported artifacts promoted into the sibling **`Sonora/huggingface`**
  checkout — a direct clone of `artificial-humanity/Sonora` alongside this repo's
  `Sonora/github`, under the flat workspace layout adopted 2026-07-22 (the umbrella workspace
  is gone). Lineage and `current` flags live in its `registry.json`.

---

## File Naming Conventions

Names must be predictable so links resolve on case-sensitive systems (Linux/CI) as well as
case-insensitive macOS/Windows.

* **Canonical root marker files → `UPPERCASE`** (`SCREAMING_SNAKE_CASE` if multi-word): `README.md`, `LICENSE`, `CONTRIBUTING.md`, `ROADMAP.md`, `AGENTS.md`. Keep this set small and curated.
* **Top-level anchor docs → `UPPERCASE`, single word preferred:** `ARCHITECTURE.md`, `STATE.md`.
* **All other docs & notes → `lowercase-kebab-case.md`:** e.g. `open-decisions.md`, `code-review-findings.md`. This is the rule for everything in `notes/`.
* **Source code → the language's own convention:** Rust `snake_case.rs`, Swift `PascalCase.swift`, Kotlin `PascalCase.kt`.
* **Never** let case be the only difference between two paths, and always reference files with their exact case.

---

## System Operational Mandates

### 1. Commit Hygiene

**THE LOOP** (owner, 2026-08-13). Work is committed, reviewed by a second session, and pushed
straight to `main`.

1. **Worker commits and taps `Reviewer`**, naming **the whole range it is about to push** —
   `@{push}..HEAD`, or `origin/main..HEAD` with no upstream — and every commit in it.
2. **`Reviewer` reviews and files issues** straight into PocketBase, each carrying the
   `review_id` of the review that produced it.
3. **`Reviewer` taps the worker**, naming the `review_id`: *"issues are filed under
   `<review_id>`."*
4. **Worker increments `agent_passes` on each issue it takes on — first, before any work —**
   then addresses them: fixing what is wrong, rebutting what is not, **in the issue's
   comments**. Then taps `Reviewer` again.
5. **`Reviewer` re-reviews and RESOLVES**: closes what is genuinely cleared, leaves open what
   is not, files anything new under a **new** `review_id`, flags what needs the owner, and
   taps back.

⚠ **EVERY PASS ENDS WITH A REVIEWER CALL** (owner, 2026-08-13). Steps 4→5 are one pass, and a
pass is not over when the worker stops typing — it is over when a reviewer has read the
result. There is no final worker pass that nobody reads.

**Repeat to a maximum of THREE passes** (owner, 2026-08-13), then the worker pushes to `main`.
Anything still open after the third review is **escalated** — see below.
**There is no report file at any point**: the tracker is the report.

**THE TWO ROLES.** Defined by responsibility, not by how either is spawned:

| | **Worker** | **Reviewer** |
|---|---|---|
| owns | the change, **and the attempt count** | the reading of it, **and whether a finding is cleared** |
| does | commits, fixes, rebuts, pushes | reads the range, files issues, **closes them**, taps back |
| writes to `main` | **yes — the only role that does** | **never** |
| **opens** issues | not as part of the loop | **yes — this is its output** |
| **closes** issues | **never** | **yes — only it decides a finding is cleared** |
| sets `escalated` | on an existing issue, only for "this needs a user decision" | **yes — the normal path** |
| holds the count | **yes — increments `agent_passes` first thing each pass** | no |

* ⚠ **THE WORKER NEVER CLOSES AN ISSUE** (owner, 2026-08-13). It fixes, or it argues, and it
  says so in the comments — but the issue stays open until a reviewer agrees it is cleared.
  **A worker closing its own findings is marking its own homework**, and it defeats the one
  thing the split exists to buy. This is the same reason the worker does not review its own
  range, applied to the tracker instead of the diff.
  * **A rebuttal is not a close either.** Argue it in the comments and leave it open; the
    reviewer accepts the argument and closes, or does not and says why. **A review is a
    report, not an order** (§5) — that still holds, and it now cuts both ways, since the
    worker cannot end the argument unilaterally any more than the reviewer can.
* **One session is only ever one role for one change.** The worker does not review its own
  range and the reviewer does not fix what it finds — that separation *is* the mechanism, and
  it survives regardless of what the reviewer is made of.
* ⚠ **DO NOT COUPLE THE PROTOCOL TO THE REVIEWER'S IMPLEMENTATION.** Today it is a persistent
  peer session (below). The owner intends to move to **single-shot `claude -p` runs** and is
  not building that yet. Everything above is written to survive that swap: it turns on *a
  reviewer reads the range and files issues*, never on a session that remembers the last one.
  The transport details in the next bullet are the part that will be replaced — keep new
  rules out of it.
  * Practical consequence today: **a single-shot reviewer remembers nothing between passes**,
    so pass 2 and pass 3 must be briefed as completely as pass 1. Writing the briefs that way
    now costs nothing and is what makes the swap a change of one bullet. The one thing that
    does *not* have to be re-briefed is the pass number — it is on the issue as
    `agent_passes`, which is why it is there.

⚠ **THE LOOP HAS ONE EXIT THAT IS NOT A PUSH, and it is a human handoff.** If a finding is
that the change **should not land at all** — it corrupts data, it ships a known-broken
training path, it cannot be safely reverted — the loop does not apply. **Do not push; take it
to the owner.** Filing an issue and pushing anyway is right for a defect that can live on
`main` and be fixed later; it is wrong for one that cannot. Deliberately a judgement call and
not a severity threshold: no automatic rule has ever separated legitimate repair from churn.

⚠ **THIS IS INTERIM.** The owner is settling a more complete workflow architecture; this is
the simple version that holds until then. Do not build tooling on its shape.

* ⚠ **NOTHING STANDS BETWEEN A SESSION AND `main`.** Measured 2026-08-13, not inferred:
  `gh api repos/:owner/:repo/branches/main/protection` returns **404 Branch not protected**.
  There is no branch protection, **force-push to `main` is unblocked**, there is no pre-push
  hook, and CI runs *after* a push rather than gating one. The abort above is the only thing
  in front of `main`, which is why it is a rule and not a preference.
* **One session commits to this repo** (owner, 2026-08-13), so `main` does not move under you
  and divergence is not an ordinary event. The repo is configured to match:
  `push.default=upstream` and `pull.rebase=false`.
  * ⚠ **`git config --local` WRITES THE SHARED CONFIG, WHICH EVERY WORKTREE READS.** This repo
    has two — this one and `/data/repos/Sonora` — so a setting made here changes git's behaviour
    in a checkout you are not looking at. "One committer, therefore harmless" is not the test:
    `commit.template` was set `--local` and pointed at a `.gitmessage` that does not exist in the
    other worktree, which makes an interactive `git commit` **fatal** there — it refuses and
    creates nothing. **Check any new setting against both checkouts, and use `--worktree` for
    anything that names a path.** `extensions.worktreeConfig` is enabled, so `--worktree` works.
  * **`git push` is the whole command.** No `HEAD:main`, no `-u`. Verified against this
    worktree, whose branch name differs from `main`.
    * ⚠ **AND IT COST A GUARD, which is worth knowing before you cut a branch.** `simple` — the
      default this replaced — *refuses* to push a branch whose name differs from its upstream,
      and that refusal is the same `'simple'` fatal blamed for breaking `@{push}`. It was doing
      two jobs and only one of them was in the way. Measured: with `upstream`,
      `git checkout -b scratch origin/main` inherits `origin/main`, and a bare `git push` from
      that scratch branch reports `scratch -> main`.
    * **So cut scratch branches from a LOCAL ref, not from `origin/main`** — measured to fail
      safely with `no upstream branch`, which is the refusal you want.
  * **`origin/main..HEAD` is the range**, and `@{push}..HEAD` also resolves now — `push.default`
    was what broke it (`fatal: cannot resolve 'simple' push to a single destination`). Prefer
    `origin/main..HEAD`: it is correct under any config.
  * ⚠ **IF `main` EVER HAS MOVED, MERGE — NEVER REBASE.** A rebase **rewrites your local
    commits**, so the reviewed SHAs cease to exist and the cycle silently ends. `pull.rebase` is
    set to `false`, so a bare `git pull` merges (verified: exit 0, local commit survives as a
    parent). ⚠ **Do not take git's advice here.** With `pull.rebase` unset, `git pull` is
    *fatal* on diverged branches and offers `git config pull.rebase true` as one of three
    equal-looking remedies — measured, and it would silently end every future cycle. The
    setting is what stands between the worker and that hint.
  * If the tree holds the owner's uncommitted local edits, fetch and check ahead/behind rather
    than integrating anything.
* ⚠ **REVIEW THE RANGE YOU ARE ABOUT TO PUSH, NOT ONLY THE LAST COMMIT** — step 2 says this,
  and it is repeated here because the numbered list is what gets copied elsewhere. A push
  carries every unpushed commit, so a review scoped to one SHA leaves the rest unread, and
  commits made while a review is in flight land in that gap. Measured on every cycle this loop
  has run so far: the range grew after the brief each time, twice from the worker's own commits
  and twice from owner instructions arriving mid-cycle. ⚠ **Commits that arrive after the review
  are a NEW CYCLE, not a third lap** — the cap forbids re-reviewing the same range, not
  reviewing new work.
* ⚠ **A rule in this file is not an enforcement mechanism, and this loop has no mechanism at
  all** — no trigger, no check, no artifact. A push that skipped the review is
  indistinguishable afterwards from one that did not. The project has learned the general
  lesson expensively: `deploy.sh`'s "deploy only when a service change is intended" was a
  header comment for weeks, got ignored eleven hours into a live training run, and is now a
  hard refusal in code.
* **Step 1 (tapping) — reaching the reviewer.** ⚠ **This bullet is the TRANSPORT, and it is
  the part the `claude -p` move replaces.** Nothing above depends on it.
  Today `Reviewer` is a peer Claude session on this host (tmux
  `claude-reviewer`), addressed by name with `SendMessage`; `ListAgents` confirms it is up.
  Messages queue and drain on the receiver's next turn, so a reply is not instant.
  * ⚠ **THE FIRST SEND IS REJECTED, IN BOTH DIRECTIONS, AND THAT IS NOT A FAILURE.** A
    cross-session send to a name the conversation has not used before returns
    `'<name>' is not an agent in this conversation` plus the ref to confirm with; re-send as
    `Reviewer [<ref>]`. Take the ref from that error or from `ListAgents`, **never from this
    file** — it changes when a session restarts. ⚠ It bites the **reply** as well as the
    request (measured from both ends), so a worker whose reviewer hit it and gave up sees a
    review that simply never arrives. Expect one rejection at each end.
  * ⚠ **IT HAS NO SYSTEM PROMPT, SO THE REQUEST MUST CARRY ITS OWN BRIEF.** The `Reviewer`
    session holds no standing context about this repo, this file, or what a good review is
    here. A bare "please review my commit" gets a generic reading of unfamiliar code. Send the
    repo path, the commit range and how to diff it, what the change is *for*, what you already
    verified, and which standards apply (§5, plus §5b/§5c when the change touches doc claims
    or stage wiring). **The quality of the review is bounded by the quality of the brief.**
  * ⚠ **A REVIEW IS DELIVERED WHEN THE `Reviewer` SAYS SO, NOT WHEN ISSUES APPEAR.** Issues
    land one at a time, so a partially-filed review is indistinguishable from a finished one
    by looking at the tracker. **Do not start the fix pass until the Reviewer has said it is
    complete.** This outlived the report file it was written for: the same mistake once
    deleted a whole second version of a report unread, because its existence on disk was
    taken as the signal.
* **Step 2 (filing) — THE `Reviewer` FILES ITS OWN FINDINGS, DIRECTLY.**
  * **The old "the `Reviewer` never files" rule is retired, and it was never an owner mandate**
    (owner, 2026-08-13: *"I never actually asked for a reviewer-never-files rule... reviewer
    filing is fine and perhaps ideal"*). It was written here by agents after a real incident —
    25 issues in one afternoon, each review producing a backlog and the backlog producing the
    next review's subject — and it generalised that measurement into a prohibition the owner
    had not asked for. **The measurement stands; the prohibition does not.** What holds the
    hazard now is the **three-pass cap**, which bounds the loop directly instead of by
    proxy. Keep the cap honest and reviewer-filing is safe; relax the cap and this is the
    first thing to reconsider.
  * **There is no report file.** The tracker *is* the report. `notes/reviews/` and its
    `review-<sha7>.md` handoffs are retired, along with the whole "write a document, then
    transcribe it into issues" step — the issues are local, immediate, and queryable, and the
    intermediate document was a second place for the same findings to drift.
  * **ISSUES LIVE IN POCKETBASE, NOT GITHUB** (owner, 2026-08-13). The tracker is the
    `issues` collection at **https://board.ai-lab-0.mcfarlin.family/_/**, and the `pocketbase`
    skill — machine-wide for Claude Code and Antigravity — carries the shape, the field list
    and the number-allocation snippet. File with `repo: "Artificial-Humanity/Sonora"`.
    * **This repo's GitHub issue tracker is empty by intent.** All 48 issues (#12–#89) were
      replicated into PocketBase on 2026-08-13, byte-for-byte including comment bodies, and
      their **numbers were preserved** — so a `#33` in a commit message still names the same
      finding, it now resolves in the tracker rather than on GitHub. `gh issue create` against
      this repo is a mistake, not an alternative route.
    * ⚠ **The tracker is superuser-only and reachable only from ai-lab-0.** Port 8090 is
      loopback-bound; there is no LAN-direct route and no anonymous read. A session that
      cannot reach it cannot file — say so and hand the findings to the owner rather than
      silently dropping them. With the report file gone this is the *only* copy, so an
      unreachable tracker now loses the whole review rather than delaying its transcription.
  * **`review_id` ties the issues to the review that produced them.** Set it on every issue a
    review files.
    * **Default: the tip SHA of the reviewed range**, at the time of the request — the one
      unambiguous choice, since a range has no single SHA and "the interesting commit" is a
      judgement two agents make differently. Full SHA or `sha7`, consistently within a review.
    * ⚠ **This is a default, not a policy** (owner). A SHA can be rewritten by a rebase, and
      not everything reviewed is a commit at all. **When there is no commit to name — a
      review of a working tree, or of a directory with no git behind it — GENERATE one**
      (`uuidgen`, or a timestamp-plus-noun); do not skip the field and do not invent a
      plausible-looking SHA, which would read as a commit that never existed.
    * **The requirement is only that it be unique to one review**, so that "what did this
      review find?" is one query. The field is indexed and deliberately **not** unique —
      a review yields many issues, which is the entire point.
    * **One review_id per pass**, so a three-pass cycle leaves three of them. That is
      expected, not a bug to normalise away: each names what was true when that pass read the
      code, and the fix commits move the tip in between.
  * **File with the reproduction re-verified, not relayed.** A finding never executed is how a
    wrong finding becomes a tracked task. This gets *sharper* now that the reviewer files
    directly: there is no longer a worker reading the report in between, so nothing else
    stands between a speculative finding and a permanent record of it.
* **Step 4 (addressing) — INCREMENT FIRST, then fix or rebut, and LEAVE IT OPEN.**
  ⚠ **Before anything else, `agent_passes += 1` on every open, un-escalated issue you are
  taking on** — an attempt is counted when it starts, so a pass that dies halfway has still
  spent one. Anything already at `3` is out of attempts: escalate it instead of picking it
  up. Then: fix what is genuinely wrong and
  say so in the issue's comments, naming the commit. Where a finding is wrong, argue it there
  and leave the code alone rather than making a change you believe is wrong — **a review is a
  report, not an order** (§5). **Close nothing**: resolving is the reviewer's job (see the
  role table), and a comment with no reasoning is indistinguishable from a finding quietly
  dropped, now that the tracker is the only record it happened at all.
  * **Re-brief on the next pass, and say what you did.** The fix pass adds commits, so
    `@{push}..HEAD` has grown; re-brief all of it. ⚠ **The later brief has a job the first
    does not: it must say which findings were fixed and how, and which were rebutted and
    why.** Without that the next review cannot tell a fix from an omission, and it will
    re-derive findings already settled — burning a pass that exists to catch *regressions
    introduced by the fix pass*, which is the class this repo has actually measured.
* **Step 5 (resolving) — THE REVIEWER CLOSES WHAT IS CLEARED** (owner, 2026-08-13). Every pass
  ends here, not at step 4.
  * **Verify, then close.** A finding is cleared when the reviewer has *checked* the fix, not
    when the worker reports one. Closing on the strength of "fixed in abc1234" reintroduces
    the self-marking this split exists to prevent, one level up.
  * **Closing is per-issue and explicit.** Leave open what is not cleared, and say in the
    comment why — "the fix addresses the symptom, the cause in §3 is untouched" is a review
    finding, not a re-file. **Do not close in bulk at the end of a pass.**
  * **Accept or refuse a rebuttal, out loud.** The worker cannot close its own argument, so
    an unanswered rebuttal is a finding left hanging with no one owning it. Close it if the
    argument holds; if it does not, say why and leave it open.
  * ⚠ **SKIP THE ISSUES THE WORKER ESCALATED.** An issue the worker flagged
    `escalated: true` as needing a user decision is **out of scope for re-review** — do not
    re-derive it, do not argue it, do not close it. It is waiting on the owner, and a
    reviewer re-litigating it burns a pass on something no pass can settle. Filter them out:
    `filter=review_id="<id>" && escalated=false`.

* **THE CAP — AT MOST THREE PASSES** (owner, 2026-08-13). The number of passes is otherwise a
  judgement call, not a fixed count; three is the ceiling, not a target. Stopping earlier
  because the work is clean is the normal, good outcome.
  * ⚠ **The cap is what bounds the loop, and it is the only thing that does.** The failure it
    exists to prevent is a *loop* — each review produces work, the work produces the next
    review — and this repo has measured that failure rather than theorised it: a fix pass on a
    large diff ran 7→5→9→7 findings, with the later rounds mostly defects introduced by the
    earlier rounds' own fixes. **More passes stopped converging and started manufacturing
    work.** A fourth pass is not a judgement call.
  * **THE COUNT LIVES ON THE ISSUE: `agent_passes`** (owner, 2026-08-13). An integer,
    defaulting to `0` — this used to be a number carried in the brief, which §1's own lesson
    says is a convention and not a mechanism.
    * ⚠ **THE WORKER INCREMENTS IT, AND DOES SO AS THE FIRST ACT OF ITS PASS** (owner,
      2026-08-13) — before reading the issue, before touching any code. It counts **worker
      attempts**, so the role making the attempt is the one that records it. An issue is
      filed at `0` and becomes `1` when a worker picks it up, not when it is written.
    * ⚠ **Incrementing FIRST is the point, not a detail.** A pass that is abandoned halfway —
      the session dies, the context runs out, the worker gives up — **has still spent its
      attempt**, because the count went up before the work started. Incrementing at the end
      would make a failed pass free, and "retry until it works" is exactly the unbounded loop
      the cap exists to prevent. **A crashed pass costs a pass.**
    * **Increment only what you are actually taking on**: open, un-escalated issues in the
      review you were tapped about. Not the whole tracker, not issues you are deferring.
    * ⚠ **The current pass is `max(agent_passes)` over the issues in play, not something a
      session remembers.** This is the property that matters: a **single-shot `claude -p`
      reviewer remembers nothing**, so a count held in conversation cannot survive the move
      the owner intends. Held on the record, it survives a session that has never run before.
    * **At `agent_passes = 3` the issue is out of attempts** — it is escalated rather than
      picked up a fourth time. A worker that increments a `3` to `4` has broken the cap.
    * **An escalated issue stops incrementing**, because neither role picks it up: the
      worker does not attempt it and re-review skips it (step 5). Its count therefore reads
      as *how many attempts were made before it was parked* — 1 means "recognised
      immediately", 3 means "we spent the whole cycle failing to settle it".
    * A finding first raised on the last pass sits at `0` while others are at `3`. That is
      correct and worth reading — it distinguishes *nobody has attempted this yet* from
      *three passes could not fix it*, which the cycle number alone cannot.
  * **THE COUNT IS PER ISSUE, NOT PER REVIEW — deliberately** (owner, 2026-08-13: *"it allows
    finer grained control"*). A single number for the whole cycle would force every finding to
    share a fate: one stubborn issue would either drag the rest through extra passes or drop
    them all at the same line. Per issue, a finding cleared on pass 1 stops there, a stubborn
    one exhausts its three, and a late one is visibly untried — and the owner can re-arm
    exactly one without touching the others.
    * ⚠ **The per-issue cap does NOT bound the cycle on its own, and must not be read as
      doing so.** If every review may file new issues and each new issue carries a fresh
      three passes, the loop never has to end — which is the *exact* failure the cap exists
      to prevent, re-entering through the door that was opened for granularity.
      **So: the cycle still ends at the third review.** Whatever is open then is escalated,
      including a finding at `agent_passes = 0` that nobody has attempted. Its low count is
      the useful part — it tells the owner this is untried work rather than a hard problem,
      and the decision being asked for is *"another cycle, or ship it as known?"*
  * ⚠ **THE COUNTER IS THE OWNER'S DIAL, AND AGENTS DO NOT SECOND-GUESS IT** (owner,
    2026-08-13). Putting the count on the record rather than in a brief means the owner can
    edit it in the admin console — **setting `agent_passes` back to `0` re-arms the loop on
    that issue for another three passes.**
    * **Treat the stored value as authoritative, always.** If an issue you expected at `3`
      reads `0`, the cap was reset on purpose; do not "correct" it, do not reason from what
      you remember, and do not restore a value you think it should have had. **The record is
      the count** — that is the entire reason it stopped living in the brief.
    * **This is the return path out of escalation, and it is the only one.** An escalated
      issue is not a dead end: the owner answers, clears `escalated`, and resets
      `agent_passes` to `0`; the issue re-enters the loop with a fresh three passes and a
      decision attached. **Without a reset it stays parked** — re-review skips escalated
      issues, so clearing the flag alone gets it read again but on a spent counter.
    * **Incrementing is the worker's; RESETTING is only the owner's.** The distinction is
      the whole guard: a worker that lowers a counter, or sets it to a value it thinks fair,
      is the cap deleting itself. Move it in one direction, by one, at the start of a pass.
      If an issue genuinely deserves more attempts, that is a judgement the owner makes.

* **ESCALATION — `escalated: true` means the owner has to decide.**
  * ⚠ **THE REVIEWER SETS IT. That is the normal path** (owner, 2026-08-13) — it flags what is
    still open after the third review, as part of that review. The worker does not sweep the
    tracker before pushing; by then a reviewer has already read everything and marked it.
  * ⚠ **THE RULE IS A QUERY, NOT A JUDGEMENT** (owner, 2026-08-13). Because the worker counts
    its own attempts, the reviewer needs no memory of the cycle to know what to escalate —
    the three fields say it outright:

    ```
    agent_passes = 3   AND   state = "open"   AND   escalated = false     →  escalate it
    ```
    ```bash
    .../issues/records?filter=agent_passes%3D3%20%26%26%20state%3D%22open%22%20%26%26%20escalated%3Dfalse
    ```

    Each clause excludes a case that must not be escalated, and all four were checked against
    the live tracker: `agent_passes = 2` still has an attempt left; `state = "closed"` was
    resolved; `escalated = true` was already parked (re-flagging it would churn the owner's
    queue and reset nothing). **Run the filter, escalate what it returns.**
    * **This is what the counter bought.** A single-shot `claude -p` reviewer, with no
      knowledge of which pass it is on or what happened before, reaches the same answer as a
      session that watched the whole cycle. The decision moved out of memory and onto the
      record — which is the property that makes the planned move safe rather than merely
      possible.
    * **No agent owns every part of this** (owner, 2026-08-13). The worker counts, the
      reviewer resolves and escalates, the owner decides and re-arms. Each reads state the
      others wrote and writes only its own field. **That is the design, not a division of
      labour to tidy up later** — a role that could do all three could also undo the cap
      by itself.
  * **The worker's one exception: an issue it can see needs a USER decision.** The worker may
    set `escalated: true` itself, at any point, on any pass — it is the role actually holding
    the change and is often first to know that no amount of reviewing will settle something.
    * ⚠ **A worker escalation takes the issue OUT of re-review** (step 5). That is the point:
      it stops the remaining passes being spent on something no pass can decide. It also
      means the flag is not free — an issue parked there is an issue no reviewer will look at
      again, so park only what genuinely needs the owner.
    * **Escalate when you know, not on pass 3.** An issue that needs a decision does not
      become decidable by being reviewed again.
  * **Escalation is a flag, not an exit.** The work still pushes; the flag says a human owes
    an answer. This is *distinct* from the abort below, which stops the push entirely. The
    test: escalation means **"this can live on `main` but I cannot choose"**; the abort means
    **"this must not land."**
  * **What the owner reads:** `escalated = true && state = "open"`. Filing something there
    that merely looks hard, rather than genuinely needing a decision, is how that view stops
    being read — the same way the 25-issue afternoon made a backlog nobody worked.
* **If `Reviewer` is unreachable or does not answer, say so and push anyway.** Say it in the
  commit trail or to the owner. A blocked push is worse than an unreviewed commit; a *silently*
  skipped review is worse than both. ⚠ This does **not** override the abort above: unreachable
  means the review did not happen, not that a "must not land" finding was cleared.

### 2. Training & Troubleshooting Mandates

* **Training Workspace**: Training runs inside the ROCm Docker container (named `sonora_training`), whose compose definition lives in the `AI-Lab-AMD` sibling repo.
* **Active Log Review**:
  * To inspect the training progress, check the container logs directly: `docker logs sonora_training` or `docker logs -f sonora_training`.
  * Local execution metrics, hydra configurations, checkpoints, and tensorboard events are output directly to `logs/`.
  * ⚠ **`logs/` is NOT WRITABLE by a normal user on ai-lab-0** (`drwxr-sr-x root datashare`,
    created by a container on 2026-07-14). Group `datashare` has `r-x` only, so **no non-root
    process can write there — including `ai-mgr`, which the containers run as**. The directory
    is empty, so nothing has ever used it and the real training artifacts land under `/data`.
    Treat the line above as the intent, not the state: **probe before writing, and never
    assume a repo-relative log path is writable** — probe an env override, then a repo-relative
    dir, then `$TMPDIR`, and print which one you chose.
    Fixing the ownership needs root and has not been done.
* **Common Troubleshooting and Fixes**:
  * *Audio Decoding Failures (ROCm CUDA Mismatch)*: `torchaudio.load()` defaults to `torchcodec`, which fails inside the ROCm container due to missing CUDA dynamic libraries. Always use `soundfile.read(..., dtype='float32')` and convert to PyTorch tensors manually (see implementation in [matcha/data/text_mel_datamodule.py](matcha/data/text_mel_datamodule.py)).
  * *Matplotlib AttributeError in Validation*: Matplotlib 3.9+ removes `tostring_rgb()`. Use `np.asarray(fig.canvas.buffer_rgba())[:, :, :3]` instead (see implementation in [matcha/utils/utils.py](matcha/utils/utils.py)).
  * *Isolated build failures (NumPy 1.24.3 source compilation on Python 3.12)*: Pre-install `Cython` and run `uv pip install --no-build-isolation -e .` to reuse container-native compiled libraries.
  * *Local Gitignore Masking*: This repo's root `.gitignore` must contain `/data` (not recursive `data`) to avoid ignoring source code folders like `matcha/data/`.
* **Continuing Forward**: If a container run fails, apply the fix, commit to this repository, then **relaunch explicitly** (committing deploys nothing — GitOps was retired 2026-07-22; service deploys go through `AI-Lab-AMD/scripts/deploy.sh`) — `docker compose --profile training up -d sonora_training` from the `AI-Lab-AMD` repo (the service is profile-gated so ordinary syncs never start it). Container state is ephemeral: recreation wipes `/tmp` and pip installs, so copy artifacts out immediately (to `/data/model-training/…`, or promote blessed ones into the `Sonora/huggingface` registry checkout).

### 3. Python Tooling Mandate — uv

* **`uv` is the standard for all Python tooling in this organization**: interpreter/version
  management, virtual environments, dependency resolution, and tool execution. Prefer `uv pip`,
  `uv venv`, `uv sync` (with `pyproject.toml` + `uv.lock`), and `uv tool run` over bare
  `pip` / `python -m venv` / `pipx` / conda / poetry.
* **`uv run` is the one exception, and only for HOST scripts** (owner, 2026-08-01). Invoke them
  as **`.venv/bin/python scripts/…`**. `uv run` resolves dependencies against the repo
  `pyproject.toml` and *ignores* the inline PEP 723 block these scripts carry, which is what made
  four launch attempts fail that day, each on a different missing module. uv still owns the venv —
  create it with `uv venv`, populate it with `uv pip install --python .venv/bin/python …` — so the
  standard above is intact; only the invocation changes. Container scripts are unaffected.
  `tests/test_gate_scripts.py` enforces this for both shell scripts and Python docs.
* **To run the test suite: `uv pip install --python .venv/bin/python --group test`.** The set
  and the reasons for every pin live in `pyproject.toml` beside the group. ⚠ Deliberately not
  restated here — the same claim in two files drifted three times in one pull request, and
  after the third correction the two copies contradicted each other.
* **New work must use uv from the start** — new scripts, containers, CI steps, and docs. Do not
  introduce new `pip install` invocations.
* **Existing pip usage is legacy** and is being migrated; the catalog of migration points and their
  done-criteria lives in [AI-Lab-AMD/notes/cleanup-chores.md](../../AI-Lab-AMD/notes/cleanup-chores.md)
  in the `AI-Lab-AMD` sibling repo. When touching a file that contains legacy pip usage, migrate it
  as part of the change when practical (small diffs, e.g. `pip install X` → `uv pip install
  --system X` in containers), rather than leaving new debt.
* **Containers**: the `rocm/pytorch`-based service commands should bootstrap uv (single static
  binary — `pip install uv` once or `COPY --from=ghcr.io/astral-sh/uv`) and then install runtime
  deps with **`uv pip install --python /opt/venv/bin/python`** — these images ship their stack in
  an `/opt/venv` activated via PATH, and `--system` bypasses it into Debian's externally-managed
  Python, which refuses installs (PEP 668; learned 2026-07-13 when the first `--system` deploy
  crash-looped two services). Use `--system` only in images whose Python truly is the system one.
  uv's resolver speed materially shortens the recreate-reinstall cycle documented in this
  project's STATE ops notes.

### 4. The Record of Change — git history

* **There is no changelog** (owner, 2026-08-11). `notes/CHANGELOG.md` was retired, along with
  the review-document cycle that cross-referenced it, because both had become a third place
  for the same facts to drift. The record of a change is now, in order of authority:
  1. **the commit message** — WHY the previous state was wrong, not merely what moved;
  2. **`git log`** — which needs no maintenance to stay accurate;
  3. **the issues filed out of a review** (§1 step 2) — which hold what a cycle could not
     settle, and are the only durable artifact the review loop produces. ⚠ **They are in
     PocketBase, not GitHub, as of 2026-08-13**, which makes this third authority the only
     one that does not travel with a `git clone`. The tracker is on ai-lab-0 and is
     backed up nightly with the rest of `/data`; a clone alone no longer carries it.
* ⚠ **THE COMMIT MESSAGE IS THE ONLY PROSE THAT TRAVELS WITH A CHANGE.** The argument, the
  alternatives rejected, and what was verified go there or they do not exist anywhere.
  The tracker is not that place either — an issue records what is still WRONG, not why a
  landed change is right.
  * **`git commit -m "one line"` is insufficient by policy for any non-trivial change.** Stated
    outright rather than left to inference from the paragraph above.
  * **`.gitmessage` is the template — for a HUMAN committing interactively.**
    * ⚠ **IT DOES NOTHING FOR AN AGENT.** `commit.template` applies only to an *interactive*
      `git commit`; `-m` and `-F` bypass it, and every commit a session makes here uses `-F`.
      **An agent must put the `Co-Authored-By` trailer in the message text itself** — CLAUDE.md
      requires it, and nothing supplies it automatically. Measured: a commit written with `-F`
      while `commit.template` was set came back with zero trailers.
    * **To enable it for interactive use:** `git config --worktree commit.template .gitmessage`.
      ⚠ **`--worktree`, not `--local`.** `--local` writes the SHARED config, and a template path
      that does not resolve in another worktree makes an interactive `git commit` **fatal**
      there — it refuses the commit and creates nothing. That is not hypothetical: it happened
      to `/data/repos/Sonora` for one cycle, from exactly this setting.
    * ⚠ Local config,
    so it does not travel with the repo and nothing enforces it; it prompts at the moment the
    message is written, which is the only moment the prompt is useful.
* ⚠ **Do not reintroduce a changelog, and do not resurrect it under another name** — a
  `notes/changes-*.md`, a "release notes" file, a running summary in `STATE.md`. The failure
  was structural, not cosmetic: a hand-maintained narrative of what changed is a copy of
  information that already exists in two authoritative places, and the copy is the one that
  goes stale. This repo has already paid for doc-vs-artifact drift repeatedly.
* **What genuinely does not fit in a commit or a PR belongs in `notes/`** as a durable
  document about the *current* state of something (`notes/STATE.md`, a design note), never as
  a dated log of past events. If you catch yourself writing "on 2026-08-11 we changed X",
  that belongs in the commit that changed X.

### 5. Code Review Standards

* **Review happens BEFORE THE PUSH, and §1 has the procedure** — commit, request a review of
  the range you are about to push, one fix pass, one re-review, file the remainder, push.
  §1 is a procedure and not a *mechanism*: nothing enforces it.
* **A review is a report, not a fix pass.** This survives the retirement and applies to any
  agent asked to review anything: the deliverable is the findings. Take on fixes only when
  the owner explicitly asks, never as a rider on the review itself.
* ⚠ **REVIEW THE INSTRUCTION, NOT ONLY THE CLASSIFICATION — they fail independently, and the
  second is where the defects hide.** Six instances across four rounds on 2026-08-11: in every
  one the code decided *correctly* and the instruction attached to it was wrong or impossible.
  A remedy naming a fix that cannot address the cause; a bucket telling the reader to "score
  them first" about clips already scored; a comment claiming an override the tool never had.
  * *"Is this line true?"* is easy to read for. *"What would someone DO on reading this line?"*
    is a different question and almost never asked. Ask it of every message, comment, docstring
    and suggested remedy — including the ones a review itself writes, since a `suggestion`
    block is committed in one click and gets far less scrutiny than the finding it hangs off.
* **Scope: code work only.** Source, configs and dependency manifests. Docs-only changes need
  no review. ⚠ **With exceptions that are ALWAYS in scope regardless of extension:
  `.claude/**`, `AGENTS.md`, `CLAUDE.md`.** `.claude/commands/*.md` is an executable prompt —
  it tells an agent holding push rights what to run — so it is closer to a shell script than
  to a README. The unqualified version of this sentence once let a change merge with **zero
  review**.
  * ⚠ **THIS IS A JUDGEMENT THE WORKER MAKES ABOUT ITS OWN CHANGE**, before it asks for
    anything — nothing checks it. When in doubt on a mixed diff, request the review.
* ⚠ **Periodic wholesale review is a different altitude — and whether `Reviewer` is also the
  session that does it is an OPEN QUESTION for the owner** (raised 2026-08-13). The owner
  keeps a floating reviewer session for reading the codebase and the product direction as a
  whole, on its own cadence: per-change review answers *"is this diff correct?"*, that one
  answers *"is this still coherent?"*. The standing warning was that the two must not be
  collapsed, because per-diff volume will always crowd out the wider read — and §1 now points
  per-commit traffic at a session named `Reviewer`, which is either that same session or one
  the file never names. **Do not resolve this by assumption in either direction; ask.** Where
  a wholesale review's output lands is open for the same reason. It is **not** the per-change
  tracker: §1's loop files issues scoped to one range under one `review_id`, and a wholesale
  read has neither. Ask before inventing a home for it.

### 5b. The doc-claims gate can stop enforcing WITHOUT going red

`scripts/gates/test_doc_claims.py` compares documented numbers against the artifacts on disk. It
has **five silent-disarm modes** — **four observed on 2026-08-11, and one reasoned from the
mechanism** (mode 2, see its own correction) — none of which was written down
anywhere until now — which is the same reason the workflow-validation trap in §1 cost the
same hour twice.

* ⚠ **A registry fact may only guard a STATIC artifact.** `audit-*/ratings.csv` is written
  **live** by the Auditions app, so a count taken from it (the 1,279 keeps, say) moves as the
  audit continues, and a fact guarding it would go **red on correct work**. A gate that fails
  when nothing is wrong gets switched off, which costs every other fact in the registry.
* ⚠ **A fact enforced only in a file some convention DELETES is one deletion from being
  disarmed** — because the checker reports only a *disagreement*, so a fact matching no
  document simply passes.
  * ⚠ **CORRECTED: `v5 speakers` is NOT an example of this, and an earlier version of this
    bullet said it was.** Measured against the merge base: it had **two** enforced lines, in a
    `notes/code-review-*.md` and in `configs/data/…_v6.yaml:8` — the latter in scope since
    `configs/data/*.yaml` was added to the scan. Deleting the review document took it **2 → 1**,
    never to zero, so it was never disarmed. The hazard is real as a class; this was not an
    instance of it, and the claim was made by measuring on a tree that predated the config
    scan.
  * ⚠ **The same-line rule — real, and the reason mode 2 looked instantiated:**
    `notes/STATE.md` already stated v5's 2,500 speakers, but `scope` is
    matched **per line** and the scope token sat on the line above, so the fact never matched
    there. **Check that a fact's scope and its number are on the SAME LINE.**
  * ✅ **This class is now ENFORCED IN CODE, not by vigilance.** #62 added
    `tests/test_doc_claims_registry.py::test_every_fact_recognises_at_least_one_live_statement`
    — *"a fact no document states is a fact nobody is checking."* An earlier version of this
    bullet said nobody had swept for other facts in this position; that was true when written
    and #62 made it false. Do not re-add a manual sweep: assert on the registry instead.
* The general form of both: **a fact that matches zero lines is indistinguishable from a fact
  that passes.** When you add or move an enforced sentence, count the lines it matches — do
  not infer enforcement from a green gate.

⚠ **AND THE SAME SHAPE OUTSIDE THIS GATE: a tool's FAILURE is easily mistaken for its
NEGATIVE RESULT.** Four instances on 2026-08-11, three of them within an hour, each costing a
wrong answer stated confidently:

| Instrument | Failed because | Read as |
|---|---|---|
| `test_doc_claims.py` | fact matched no document | fact passes |
| `gh pr diff N -- path` | takes one arg; errored to **stderr** | empty diff, file untouched |
| `git merge-tree` | prints conflict to **stdout**, grepped stderr | no conflict |
| `git merge <ref>` | branch name misspelled, ref missing | merge conflict |

**Before believing a negative result, check that the instrument ran.** Non-zero exit, an empty
list where a real answer prints something, output on the stream you are not reading — all
produce a confident nothing. Two of the four above were caught only because another agent
contradicted the claim, and in both of those cases the person had *already run* the command
that would have falsified it and stopped reading once the answer looked settled.

**A cheap falsifier that is not run is not evidence.**

⚠ **A THIRD DISARM MODE, AND THE ONLY ONE READING CANNOT FIND: a test whose PREMISE and whose
SUBJECT are wrong in the same direction passes.** Found 2026-08-11 in
`test_one_bad_axis_is_enough_to_hold_a_clip_out_of_keeps`: the fixture set an intended `V: 0.9`
against a stub measuring `−0.79`, so V actually FAILED and the clip was a direction failure
too — "held out on the label alone" was false of it. **The assertion passed anyway, because the
count it asserted on was the buggy one.** Two defects agreeing, a green test between them, and
neither visible from the other.

* **The tell is that it cannot be found by reading either the test or the code** — only by
  running the case and looking at what the subject actually did, rather than at whether the
  number came out as expected.
* So when a guard's own test is the evidence that the guard works, **check the fixture states
  what you think it states.** A passing assertion proves the two sides agree; it does not prove
  either is right.
* Same family as the other two: the green result is indistinguishable from the correct one.

⚠ **A FOURTH MODE, AND THE ONLY ONE WITH NO MISSING MEASUREMENT: a claim can be internally
incoherent and still survive two readers, when both like the story it tells.** Found
2026-08-11, in this repo's own review lane. The claim was *"numba 0.53.1 cannot build on 3.12,
**but** the same resolution silently downgrades librosa by a major version"* — offered as the
stronger argument for a version pin, agreed with by a second agent in the same terms, and
**impossible**: if the build fails nothing is installed, so nothing is lived with. The
downgrade is planned in the resolution and never installed.

* **The resolution had been measured.** So "run the falsifier" would not have caught it — the
  numbers were right and the conclusion drawn from them was not.
* **The check that does catch it: state the END STATE, singular.** What does the machine
  actually end up in? One install has one outcome, and naming it forces the incompatible
  halves into the same sentence where they cannot both stand.
* ⚠ **A second reader agreeing is not verification** — it is likelier to be two people liking
  the same story. The agreement arrived one round after a review had already corrected the
  previous version of the same sentence.

⚠ **MODE 5, the inverse of mode 4: "the number is unchanged" is itself a claim, and it has to
be re-derived rather than inferred from the fix looking conservative.** Mode 4 is a measured
number with an unmeasured conclusion; this is an unmeasured number that a narrow-looking change
invites you to assume still holds. Found 2026-08-11 when a `ge90` change left "13 of the 20
campaigns" unregenerated — it was re-run and is still 13, so the note was right, which is
exactly why the habit is dangerous: being right this time costs nothing and teaches the wrong
lesson. **If a fix touches an input to a stated number, re-derive the number.**

### 5c. A pipeline stage is WIRED, or it is merely WRITTEN ABOUT

`scripts/` holds **100 non-test `.py` files** (106 tracked, less the 6 gate scripts) and
most of them are *correctly* uninvoked — operator tools, finished campaign tooling. So "nothing calls this" carries no signal there, and a
**stage** that stopped being called is indistinguishable from a tool that never was. That is
how `qc_verdict.py` was named in a `synth_bank.sh` comment for a month and never ran, while
695 directed clips reached the ear with no direction check (issue #24).

* **The buckets say what a file IS; the manifest says whether it RUNS. Read
  [scripts/README.md](scripts/README.md) before adding anything under `scripts/`.**
  `stages/ lib/ tools/ gates/ assets/ teacher_audition/ litert_export/` (#26 step 3,
  2026-08-12). Every file under `scripts/<bucket>/` is **exactly two levels down** on
  purpose — one repo-root expression is then correct everywhere, which is what replaced 87
  scattered `sys.path.insert(0, dirname(__file__))` calls and what makes the path guard
  below possible.
* ⚠ **Checked-in data a script reads goes in `scripts/assets/`, resolved from the repo root.**
  `tests/test_asset_paths.py` asserts every in-repo path built from `__file__` points at the
  thing it names. That class of bug does not fail at import — it fails when something READS
  the path, which for the synthesis lane is hours into a GPU render, and for
  `book_ingest`'s register lexicon does not fail at all: the load sits inside
  `except Exception: return []`, so a wrong depth silently yields an empty controlled
  vocabulary and every bank after it is built without one.

* **`scripts/pipeline_manifest.py` declares every stage and which shell wires it**, and
  `tests/test_stage_coverage.py` iterates it in both directions: a stage declared and not
  invoked fails, a script invoked and not declared fails, and a `.sh` appearing under
  `scripts/` in none of the three categories fails. **Wire a stage → declare it in the same
  commit.** How to do that is in the manifest's docstring, which is the only copy.
* ⚠ **Two things look like a call and are not.** A **comment** (the #24 failure verbatim) and
  an **`echo`** — every stage in `synth_bank.sh` prints its own re-run command on failure, so
  the recovery hints name the very scripts under test. `tests/test_audit_sampling.py::_invocations`
  is this repo's one definition of "a call"; import it, never re-derive it. The guard that
  predated it asserted a substring over the whole file, and commenting out the real invocation
  left it **green**.
* ⚠ **A ratchet never observed going red is not known to work.** This one was built by mutating
  the tree eight ways — unwire a stage, demote it to an `echo`, add an undeclared stage, add a
  new shell, reverse a recorded non-invocation, unwire the nested EIV lane, hardcode a target
  into the dynamic-dispatch wrapper, empty the manifest — and recording which test caught each.
  The last is the point: **every test there iterates the manifest, so an emptied manifest would
  collect zero cases and report green.** That is §5b's mode 1 in a different file, and it is why
  `test_the_manifest_declares_the_lanes_it_is_supposed_to_cover` exists. Do the same for the
  next ratchet: a coverage test's own coverage is not self-evident.

### 6. Execute From The Repo — `/data` Holds Data

**Owner principle (2026-08-06): code executes from the repo checkout; `/data` holds what
its name implies** — datasets, checkpoints, model artifacts, venvs, training logs, service
runtime state, vendor checkouts. A byte-copy of our source under `/data` is something to
*remove*, not to manage. The full inventory, what is legitimately untracked, and the audit
behind this rule are in [notes/data-mirrors.md](notes/data-mirrors.md).

* **When a tool writes its outputs next to its own source** — the usual reason a `/data`
  copy exists at all — give it an artifact-root variable defaulting to the script's own
  directory, point that at `/data`, and delete the copy. That is the shape that satisfies
  both halves. Worked example: `SONORA_LITERT_WORK` and
  [scripts/litert_export/run.sh](scripts/litert_export/run.sh).
* **For containers, bind-mount the repo path**, not a `/data` copy of it.

Where a copy still exists, the rest of this section governs it. **The repo is
authoritative in every case.**

* **Never edit code under `/data`.** Change it in the repo, commit, then deploy. An edit made
  on `/data` has no history, no diff and no review, and no way for anyone
  else to discover it happened.
* **A `/data` file that is *newer* than its tracked original is not authoritative — it is
  unreviewed.** If that edit is the one you want, commit it in the repo and redeploy; do not
  let the copy become the record.
* **Deploy explicitly.** Committing deploys nothing (GitOps retired 2026-07-22). Service
  stacks go through `AI-Lab-AMD/scripts/deploy.sh`; the training clone at `/data/repos/Sonora`
  through `deploy.sh training-code`.
* **Adding a tool that will run from `/data`? Add it to `MIRRORS` in
  [tests/test_data_mirrors.py](tests/test_data_mirrors.py) in the same commit.** That gate
  compares every tracked file against its deployed copy and fails on any difference. It is the
  only thing standing between us and this failure mode, and it only covers what it is told
  about.
* **What legitimately lives only on `/data`, and must stay untracked:** venvs, `artifacts*/`,
  checkpoints, training logs, datasets, vendor checkouts (`toolchain/Zonos`,
  `services/comfyui`, `services/unsloth`, …) and service runtime state.

**Why this is a mandate and not a preference.** A copy that stops updating looks exactly like
a copy that is up to date — both directories are healthy, the scripts run, and the only
symptom is that a fix you believe shipped did not. It has already happened twice:

* `convert_vat.py` on `/data` ran **three weeks stale**, missing the `detect_vat_dim` seam
  guard that this repo recorded as landed. The guard was written, reviewed, committed and
  never installed.
* The training deploy clone went **dead for two weeks** in July, because `deploy.sh
  training-code`'s fast-forward pull cannot cross a history rewrite and failed silently.

Neither was found by noticing something break. Both were found by going to look.

---

### 7. The Deploy Cycle — repo → `/data`, and the check that comes FIRST

§6 says *why* the repo is authoritative. This is the *procedure*, and it is mandatory for
every target with a `/data` copy. Canonised 2026-08-08, when `audition/` moved into this repo
and the cycle stopped being another repo's internal business.

**Targets, and where each is sourced from.** `scripts/deploy.sh` lives in `AI-Lab-AMD`
because it is box tooling, but it deploys from whichever repo owns the code:

| target | source | command |
|---|---|---|
| `audition` → `/data/services/audition/app` | **this repo**, `audition/` | `deploy.sh audition` |
| `training-code` → `/data/repos/Sonora` | **this repo** (ff-pull; a real checkout) | `deploy.sh training-code` |
| `dashboard` → `/data/services/dashboard` | `AI-Lab-AMD/dashboard` | `deploy.sh dashboard` |
| `stack` → the compose services | `AI-Lab-AMD` | `deploy.sh stack` |

⚠ **`deploy.sh` reads `SONORA_REPO` for the source checkout and defaults it to
`Sonora/github` — the caretaker's tree, which is usually on a feature branch.** Scoped to
`deploy.sh` on purpose: the same name has three *other* readers in this repo with different
defaults (`scripts/litert_export/run.sh`, `convert_vat.py`, `scripts/stages/score_holdout.py`
— the last falling back to the container path `/sonora`), so an unqualified "it defaults to
`Sonora/github`" is false of every in-repo consumer.

An agent working in a linked worktree passes it **per command**:

```bash
SONORA_REPO="$(git rev-parse --show-toplevel)" deploy.sh audition
```

* `--show-toplevel`, not `$PWD`: it is exact from any depth and returns the *linked
  worktree's* root. `$PWD` means "wherever I happen to be standing", and an agent that
  `cd`'d into `audition/` to make the change would hand `deploy.sh` a source root of
  `<worktree>/audition`. ⚠ **An earlier version of this bullet said that root "inherits a
  git dir from the worktree and passes `require_source`". It does not** —
  `<worktree>/audition/.git` does not exist at all, so `require_source`'s `-e` test *refuses*
  it. The advice is right and the reason was wrong, which contradicted the `require_source`
  note below — and the errata itself then pointed at that note as "eight lines down" when it
  is nearer thirty, a stale pointer inside a correction about stale claims. Its replacement
  said "at the end of this subsection", which was *also* positional and *also* wrong (the
  note sits about two-thirds through §7). **Positional cross-references rot — name the thing
  and let the name carry it.** Use
  `--show-toplevel` because it is correct from any depth, not because anything downstream
  fails to catch `$PWD`.
* ⚠ **Prefix it; do not `export` it.** `scripts/litert_export/run.sh` honours an inherited
  `SONORA_REPO` through `${SONORA_REPO:-…}`, so an export run later in the same session
  would resolve `import matcha` out of the worktree instead of the caretaker tree.
  `convert_vat.py`'s guard cannot see that — it only asserts `matcha/models/matcha_tts.py`
  exists, which is true of any worktree — and its own comment names the stakes: *a wrong
  export converts cleanly and its graphs run*.

⚠ **THE CHECK THAT ANSWERS "will this deploy revert someone's work" IS AN ANCESTRY TEST, NOT
A LOG RANGE.** `rsync --delete` replaces the target with *your tree*, so the question is
whether your tree already contains everything the deployed copy has:

```bash
git fetch origin                                          # remote-tracking refs are SHARED
                                                          # across worktrees and may be days old
git merge-base --is-ancestor <deployed-sha> HEAD           # exit 0 ⇒ your tree is a superset
git log --oneline HEAD..<deployed-sha> -- audition/        # non-empty ⇒ this deploy reverts work
```

An earlier version of this bullet said to run `git log <deployed-sha>..origin/main -- audition/`.
**That cannot detect the failure it was written for**: it compares two refs, neither of which
is the tree being rsynced, and whenever the deployed sha *is* main's tip the range is empty by
construction — so it prints nothing and reads as "safe" while your branch, forked before a
commit that touched `audition/`, is about to delete it. The 2026-08-12 deploy that prompted
this was in fact safe (`20713f1` is an ancestor of that branch's HEAD), which is the danger:
the wrong check agreed with the right one that once.

`require_source` used to refuse a worktree outright — `.git` is a file there, not a directory
— fixed in AI-Lab-AMD `882f620` (2026-08-12).

⚠ **A SQUASH MERGE ORPHANS A DEPLOY STAMP MADE FROM THE BRANCH, AND `status` THEN LIES.**
Observed 2026-08-12 minutes after #74 merged: the audition target read
**`STALE — source changed since (7557aea)`** while its bytes were **byte-identical to
`main`**. The stamp named the branch commit `c444f2c`, which the squash had replaced, so
`stamp_is_current` compared against a commit no longer reachable. Step 0 tells you to treat
STALE as drift and resolve it before editing — here there was nothing to resolve, and the
fix is one idempotent re-run: `deploy.sh audition` diffs first, finds nothing to copy, and
refreshes the stamp **without restarting**. So: **after a branch-sourced deploy is
squash-merged, re-run the deploy once to re-stamp.**

⚠ **Settle "orphaned stamp vs. real drift" with the rsync dry-run, NOT with
`test_data_mirrors`.** An earlier version of this note nominated that test, and it cannot
make the distinction: `MIRRORS` registers `audition/app` with a NON-RECURSIVE `*.py` glob,
which is `main.py` alone — one of the four tracked files under `audition/`. A green run
therefore says nothing about `static/`, and reading it as "the bytes match" is the
instrument-failure-mistaken-for-a-negative-result shape from §5b.

What answers it is the dry-run **with `deploy.sh`'s own three excludes**, which is byte-exact
over every file — an empty plan means the stamp drifted and **the content** did not (content
only; see the limit stated below the command):

```bash
sudo rsync -rl --delete --checksum --dry-run --itemize-changes \
  --exclude='__pycache__' --exclude='.deployed.json' --exclude='_contract/' \
  audition/app/ /data/services/audition/app/
```

⚠ **`-rl` ANSWERS ONE QUESTION: does the CONTENT match.** It cannot see **mode or ownership
drift** — dropping `-p`, `-o`, `-g` retires those comparisons along with the time comparison,
so a file deployed `0600` that should be `0644`, or owned by `root` where it should be the
service user, is byte-identical under `-rl` and prints nothing. That is the intended trade:
content drift is what this section exists to settle, and permission drift has other symptoms.
But it means **an empty plan here is not a full clean bill of health**, and the empty-plan rule
below is scoped to content only. (`-D` goes too, so a non-regular file is skipped with a
warning on **stderr** rather than itemised on stdout — irrelevant for this tree today.)
Stating the limit rather than implying its absence is the point: the two previous spellings
were both replaced for inviting a reader to treat silence as clean.

⚠ **`-rl`, NOT `-a`, AND NO `grep`.** This is the third spelling of this command; the
reasoning matters more than the flags.

`-a` is `-rlptgoD`, so it itemises differences in **time, perms, owner and group** as well as
content, and `--checksum` changes only how rsync *decides to transfer*, not what it reports.
A file with identical bytes and a skewed mtime prints `.f..t......`, and the empty-plan rule
above reads that as drift. The previous fix for that was to keep `-a` and pipe through
`grep -E '^[<>]fc'`, on the grounds that `c` in column 3 is the content verdict. **That
filter was worse than the noise it removed**, because `c` only ever appears for an item that
exists on BOTH sides. Measured on a synthetic pair (rsync 3.4.1):

| situation | itemised as | survived `^[<>]fc`? |
|---|---|---|
| content differs | `>fc........` | ✅ |
| identical bytes, skewed mtime | `.f..t......` | ✅ correctly dropped |
| **file present in the repo, MISSING from the deployment** | `>f+++++++++` | ❌ **dropped** |
| **file in the deployment, absent from the repo** | `*deleting   path` | ❌ **dropped** |
| symlink whose target changed | `cLc........` | ❌ dropped |
| directory not yet created | `cd+++++++++` | ❌ dropped |

For a newly-created item rsync replaces **every** attribute letter with `+`, so the content
column is `+` and never `c`; a deletion carries no update flags at all; and a changed symlink
puts `c` in column 1, so `[<>]` misses it even though column 3 matches. Net effect: a
deployment missing half of `static/` printed **nothing**, and the operator was told in bold
to read nothing as *no drift*. That is the §5b shape again — and one rung worse than the
version it replaced, which at least erred loudly.

Dropping `-ptgoD` fixes it at the source instead of filtering the symptom: rsync never
compares time, perms, owner or group, so **every row above reports** and there is nothing left
to filter — at the cost of the mode/ownership comparison noted above, which is a narrowing of
the question, not free. Verified both directions — mtime-only skew across five files prints
nothing; content drift, a missing file, an extra file and a changed symlink all print.
⚠ It also removes an exit-code trap: `rsync … | grep` exits **1** when there is no drift, so
the clean path was the failing path, which bites the first person to lift this into a
`set -e -o pipefail` script.

⚠ **`-rl` IS FOR THIS DIAGNOSTIC ONLY — never deploy with it.** `deploy.sh` uses `-a`
deliberately; a copy without `-ptgoD` would strip modes and ownership off a live service
directory. `--dry-run` is spelled long here so that deleting it is a visible edit.

⚠ **The excludes are not optional either**, and dropping them is the same mistake one rung down:
`.deployed.json` and `_contract/` are written by `deploy.sh` AFTER the copy and exist in no
commit, so a bare dry-run reports them as deletions and reads as drift. Measured just now:
five `*deleting` lines, not the "three" an earlier version of this note claimed — `_contract/`
contributes its own contents as well as itself, and the `__pycache__` count moves with
whatever interpreters have run, so **the number is not worth stating**, only the shape.
⚠ And `deploy.sh
audition` is the one-step version of this only on a CLEAN tree — `require_clean` refuses a
dirty one, correctly, so it is not available as a diagnostic mid-edit.

**Deploying is idempotent and free to over-run** (2026-08-08). `deploy.sh audition` /
`dashboard` diff the target first and do **nothing** when it already matches — no copy, no
stamp rewrite, no container restart. So running a deploy at the outset of any work costs
nothing and requires no judgement about whether it is needed; that is the point, because a
deploy that always restarted a live rating app made the safe habit the expensive one.
`training-code` (`git pull --ff-only`) and `stack` (`compose up -d`) were already idempotent
by construction.

⚠ **`stack` REFUSES during a training run**, because `up -d` restarts manually-stopped
services and would put every inference engine back on the GPU under a live run — the
spin-down rule broken by a command that never mentions inference. Override with
`ALLOW_STACK_DURING_TRAINING=1`, then re-run `inference-engines.sh stop`.

#### 0 · BEFORE touching related code — confirm the deployed copy is current

**This step comes first and is the point of the section.**

```bash
../../AI-Lab-AMD/scripts/deploy.sh status                  # which revision is running
.venv/bin/python -m pytest tests/test_data_mirrors.py -q   # do the bytes still match
```

A target reading **STALE**, **DIRTY**, or *"deployed from a commit not in this repo"* means
the running copy is not the code you are about to edit. Editing on top of that and then
deploying **silently clobbers** whatever was actually live — and since `rsync --delete` is
not recoverable, there is nothing left to diff. Resolve drift *before* the first edit.

Not hypothetical: `convert_vat.py` ran **three weeks stale** on `/data` — the repo gained a
`detect_vat_dim` seam guard, the notes recorded it as landed, and the harness that actually
executed never received it. *A guard that is not installed is not a guard*, and the only
symptom was that a fix believed shipped had not.

#### 1 · Edit in the repo · 2 · Commit · 3 · Deploy · 4 · Verify

```bash
git commit …                                    # a dirty tree is REFUSED by deploy.sh
../../AI-Lab-AMD/scripts/deploy.sh <target>
../../AI-Lab-AMD/scripts/deploy.sh status       # the target should now read `current`
.venv/bin/python -m pytest tests/test_data_mirrors.py -q
```

#### THE RULE: edit at the SOURCE, deploy to the target. Never the reverse.

**Code is edited in the repo and deployed. It is never edited at the deploy target, and
never copied back from one.** This is not a style preference and it has no exceptions:

* An edit under `/data` has **no history, no diff and no review**. A
  `/data` file that is *newer* than its tracked original is not authoritative — it is
  unrecorded, and it will be destroyed without ceremony by the next deploy, because
  `rsync --delete` leaves nothing to recover or diff.
* It defeats every guard we have. `test_data_mirrors.py` reports drift but cannot tell an
  intentional in-place fix from an accident; `.deployed.json` will name a commit that does
  not contain what is running; and `deploy.sh status` will read `current` while the target
  holds code that exists nowhere in git.
* **Enforced, not merely stated.** `deploy.sh` diffs the target before copying and, when it
  finds content the repo does not match, prints the exact files and says they are about to
  be overwritten *before* doing it — so an in-place edit is caught at the moment it costs
  nothing rather than discovered later as a mystery.

If an edit was made at the target and is the one you want: **port it into the repo, commit
it, and deploy** — do not rsync it backwards.

#### Rules that travel with the cycle

* **Register a new target in the same commit that first deploys it.** `MIRRORS` in
  `tests/test_data_mirrors.py` covers only what it is told about, and
  `test_there_is_something_to_check` exists so a layout change cannot quietly empty it.
* **A contract the deployed copy needs must be SHIPPED, never transcribed.** The audition app
  cannot import `matcha` — its container has fastapi and uvicorn and nothing else — so
  `deploy.sh` copies `matcha/delivery.py` into `app/_contract/` **after** the `rsync --delete`,
  and the app **refuses to start** without it. Same pattern as the device G2P's exported
  `g2p_contractions.json`. A fallback to a literal would re-create the fork the asset exists
  to delete; D-C1's lesson is that a hand-synced table is a defect of omission on a delay.
* **Staleness is judged per source PATH, by ancestry** — against the last commit touching that
  target's own path, never against repo HEAD. A HEAD comparison called `dashboard` BEHIND for
  a week while it was perfectly current, and a check that cries wolf gets ignored, which is
  worse than no check.
* **A dirty tree is refused.** `ALLOW_DIRTY=1` overrides and the stamp records that it was
  used, so the exception leaves a trace.
