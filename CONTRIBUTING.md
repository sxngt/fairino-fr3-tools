# Contributing

Thanks for helping. Reports from other firmware versions, other grippers and other ROS
distributions are especially welcome.

기여 환영. 다른 펌웨어·그리퍼·ROS 배포판에서의 결과 보고가 특히 반갑다.

## Development

```bash
uv sync --extra dev
uv run ruff format src tests examples
uv run ruff check src tests examples
uv run pytest -q
```

CI runs the same on Python 3.10-3.12 without ROS, the SDK or a robot. New code needs tests that
also run without them: use `FakeArm`, `FakeRobot`, `ScriptedPad`, `calib.simulate.Rig` and
`SimClock` (see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)).

## Style

* Docstrings and comments in English first, then Korean (`English. / 한국어.` on one line, or a
  short English paragraph followed by a Korean one).
* Readable multi-line code, ruff line length 110.
* Comments explain why, not history: no "fixed on ..." or changelog notes.
* No personal paths, hostnames or site-specific values; defaults must be safe and generic.
* Anything that moves the arm is dry run unless `--execute`, goes through the servo stream and
  takes the motion lock.

## Reporting robot behaviour

Please include the controller firmware (`fr3-check` prints it), the SDK tag, the command line,
and the driver log. If a code is unknown to `fr3-errors`, the text shown by the teach pendant or
the web UI is very useful: the lookup table grows from such reports.

## Commits and pull requests

Small, focused commits with plain messages. Make sure `ruff format --check`, `ruff check` and
`pytest` pass before opening a pull request.
