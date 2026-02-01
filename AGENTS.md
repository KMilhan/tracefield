# Repository Guidelines

별도의 지시가 있을 때까지 한국어와 영어의 혼용을 허락한다.

## Project Structure & Module Organization
- `ad.md`: ISO/IEC/IEEE 42010 아키텍처 서술 문서.
- `goal.md`: 프로젝트 목표와 rationale.
- `build.zig`, `build.zig.zon`: Zig 빌드 정의.
- `src/main.zig`: CLI 엔트리.
- `src/root.zig`: core 모듈 export.
- `src/pipeline.zig`: 파이프라인 타입/설명 스텁.
- `src/inputs.zig`, `src/inputs/*`: 입력 파서 스켈레톤 (ekos/phd2/telemetry).
- `LICENSE`

## Build, Test, and Development Commands
- Build: `zig build`
- Run: `zig build run`
- Tests: `zig build test`
- Format: `zig fmt src/*.zig src/inputs/*.zig`

## Coding Style & Naming Conventions
- Zig formatting은 `zig fmt` 기준.
- Types: `PascalCase`, functions/vars: `lowerCamelCase`, files: `lower_snake_case`.
- Parser/IO는 작은 단위 함수로 분리하고 에러 타입은 명확히 정의.

## Testing Guidelines
- 새로운 파서/모듈 추가 시 최소 1개 단위 테스트 추가.
- 변경 후 `zig build test` 수행.

## Commit & Pull Request Guidelines
- Commit style: colon-based emoji prefix, no category/scope. Format `:<emoji>:` + imperative subject.
    - Examples: `:sparkles: add ekos parser stub`, `:memo: update architecture notes`, `:wrench: adjust build config`.
- Atomic and immediate: keep each commit a single logical change; commit and push right away to minimize drift.
- Verbose messages: include a clear subject and a short body explaining why, what changed, and doc refs if any.
    - Example command: `git commit -m ":sparkles: add telemetry parser" -m "why: scaffold ingestion; what: add inputs/telemetry_video; refs: ad.md 7.1" && git push`
- PRs include: summary, linked issues, commands run (build/test), and doc refs when `ad.md`/`goal.md` changes.

## Security & Configuration Tips
- Restrict GitHub CLI scope to this repository: `gh repo set-default KMilhan/tracefield`; verify with `gh repo view -R KMilhan/tracefield`.

## Issue-Driven Workflow (Loop)
- Status labels: use `status:ready`, `status:in-progress`, `status:blocked`, `status:review`. Create if missing: `gh label create "status:ready" -R KMilhan/tracefield --color 0366d6`.
- Pick next task: open, non-epic issues labeled `status:ready` (prefer `priority:high`, then `type:task`/`type:enhancement`).
    - Example: `gh issue list -R KMilhan/tracefield -l "status:ready" -l "type:task" -s open --limit 1 --json number,title,url`
- Start work: mark in progress and leave a short comment.
    - `gh issue edit <n> -R KMilhan/tracefield --add-label "status:in-progress" --remove-label "status:ready"`
    - `gh issue comment <n> -R KMilhan/tracefield -b "Starting: will implement and update docs if needed."`
- Implement: keep commits atomic; push immediately; reference the issue and doc refs in the body if applicable.
    - `git commit -m ":sparkles: implement X" -m "why: ...; what: ...; refs: ad.md 7.x; closes: #<n>" && git push`
- Close: after push, close the issue with a link to the commit/PR.
    - `gh issue close <n> -R KMilhan/tracefield -c "Done in <sha/url>"`
- Fallback when no `status:*` labels: treat any open non-epic with `priority:high` as ready.

## One-Phrase Command: "start work"
- Trigger: When you say "start work" (optionally with an issue number), the agent will autonomously:
    1) Ensure GitHub CLI is scoped and labels exist
        - `gh repo set-default KMilhan/tracefield`
        - Create missing status labels: `status:ready|in-progress|blocked|review`.
    2) Select the most relevant open, non-epic issue
        - Preference order: `status:ready` > `priority:high` > `type:task` > oldest updated.
        - Fallbacks: if none ready, pick any open `priority:high`; else the oldest open.
        - To target a specific issue: say `start work #<n>`.
    3) Start the issue
        - Apply `status:in-progress`, remove `status:ready`; add a brief comment with intended steps.
    4) Implement and push immediately
        - Make minimal, atomic changes; commit with `:<emoji>:` subject and verbose body including doc refs and `closes: #<n>`; push to `main`.
    5) Close the issue
        - `gh issue close <n> -c "Completed in <sha/url>"`.
    6) Stop after one issue unless you say "start work continuously" (loops to the next ready issue).

- Alignment reminders
    - Logic/architecture changes should stay consistent with `ad.md` and `goal.md`.
    - If you change pipeline behavior, update `ad.md` sections to match.
    - Run `zig build test` before closing issues.
- "preview start": list the candidate issue and planned steps without making changes.
- "pause work": remove `status:in-progress`, re-add `status:ready`, and note why.
- "next": close current if done, then immediately pick the next ready issue.
