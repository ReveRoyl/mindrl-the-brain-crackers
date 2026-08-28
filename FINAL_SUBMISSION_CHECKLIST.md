# Final submission handoff

## Before publishing

1. Review the final interpretation card, experiment report and
   `post_freeze_changes.md`. The new public-data scores are development results,
   not hidden-test scores.
2. Confirm team membership: `submission.yaml` currently lists only Lei.
   Add the other members' names, approved emails and roles if they should be
   included. No contact details have been guessed.
3. Confirm whether the team used a separate optional post-freeze hidden
   evaluation. None was requested during this development pass. If one exists,
   identify its exact SHA in `post_freeze_changes.md`.
4. Preview `interpretation_card.md` on GitHub and in your installed Typora.
   Local MathJax validation does not certify those two exact applications.
5. Run tests and evaluation, inspect the diff, then commit only the intended
   code, small parameter/report artifacts, documentation and tests. Do not
   commit `data/`, credentials, caches, or `.preview/`.

## Pin the reviewed package

After the team approves and commits the final package:

Suggested commit message: `feat: validate history-augmented model and finalize submission docs`

```powershell
git status --short
git show --stat --oneline HEAD
git rev-parse HEAD
```

Confirm that the working tree is clean and the displayed commit contains the
intended `agent.py`, `config.yaml`, parameter artifact, interpretation card and
metadata. Push that reviewed commit to the team repository. Open its exact
commit page on GitHub and check the files there.

Then open the [Rolling submission form](https://github.com/mindrl-challenge/mindrl-challenge-core/issues/new?template=rolling_submission.yml)
for the final update. Use the same team ID `the_brain_crackers` and the full
40-character SHA from the committed, pushed package — never `main`, an
uncommitted working tree or the old frozen SHA by accident.

| Form field | Value |
|---|---|
| Title | `[Rolling submission]: the_brain_crackers` |
| Team ID | `the_brain_crackers` |
| Display Name | `The Brain Crackers` |
| Visibility | `internal` |
| Repository URL | `https://github.com/mindrl-challenge/submission-the-brain-crackers.git` |
| Commit Hash | The exact reviewed and pushed 40-character SHA |
| Agent Path | `agent.py` |
| Config Path | `config.yaml` |
| Requirements Path | `requirements.txt` |
| Submission Metadata Path | `submission.yaml` |
| Interpretation Card Path | `interpretation_card.md` |
| Method Family | `cognitive_reinforcement_learning_model` |
| Short Description | Copy the final `submission.short_description` from `submission.yaml`. |

Do not insert a self-referencing commit hash into `submission.yaml`.
The issue and central registry are the authoritative pin.

## Complete the modeling–auditing interaction

- Review the two reply drafts in `labbook/audit-response-notes.md`, add the
  published fix/documentation SHA, and post to existing issues #2 and #3.
- Do not claim the auditors evaluated the post-freeze model or change their
  frozen target. Ask them to confirm the proposed resolution.
- Check the automatically generated registry PR and toy CI. A green toy check
  is not an official hidden evaluation. Send the new issue/PR link to organizers.

## Deadline and status

The supplied organizer notice states **2026-08-27 23:59 AoE**, equivalent to
**2026-08-28 12:59 BST in London**. Allow time for metadata/CI corrections.

This local work does not itself publish commits, post audit replies, update the
registry or request a hidden evaluation. The previously accepted checkpoint
is tracked by [issue #63](https://github.com/mindrl-challenge/mindrl-challenge-core/issues/63)
and [PR #64](https://github.com/mindrl-challenge/mindrl-challenge-core/pull/64).
