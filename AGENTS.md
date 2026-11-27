# Agent instructions for 재민's work in HEAT (Team 14)

These apply to work done for 재민 (GitHub `JAEMINII`). `CLAUDE.md` in this repository is the team's main instruction file; follow it first. Where this file and `CLAUDE.md` conflict, ask 재민.

## Experiments and Git

- When an experiment produces results, commit and push right away. Do not wait to be asked.
  - Branch: `jaemin`. Never push to `main`; changes reach `main` only through a pull request the team reviews.
  - Commit the code, the result tables and figures, and a short report that states the setup (data, period, forecast origins, metric) and the numbers. Add a row to `experiments/log.csv`.
  - Stage only files that belong to the experiment. Leave unrelated untracked files alone.
  - A run that failed or was stopped is still recorded if its result informs a decision; say so in the report.

## Decisions you cannot make yourself

1. Ask 재민 first. State the question, the options and what each would change.
2. Then post the question to the team's agents on Slack:
   - Workspace `timesereis-team14`, channel `#agent` (`C0CKA90F79S`).
   - Start the message with `@channel` (in the Slack API text, `<!channel>`) so every teammate's agent sees it.
   - Write it so a reader without this conversation can answer: background, the exact question, the options, and links to the relevant commit or file.
3. Skip step 2 only if 재민 says not to post it.

Do not post to any other Slack workspace or channel (in particular, not to `#agent` in the `hyperballhq` workspace).
