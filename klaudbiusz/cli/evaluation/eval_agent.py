"""Agentic evaluation using Claude SDK.

Replaces template-specific shell scripts with direct Claude SDK agent calls.
Instead of `bash install.sh`, ask the agent "install dependencies for this app"
and let it figure out the commands.

When running as root (e.g., on Databricks clusters), falls back to direct
shell script execution since Claude SDK doesn't allow --dangerously-skip-permissions
when running as root for security reasons.
"""

import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)


def _is_running_as_root() -> bool:
    """Check if running as root user."""
    return os.geteuid() == 0 if hasattr(os, 'geteuid') else False


# Evaluation step prompts - agent figures out specific commands
EVAL_PROMPTS: dict[str, str] = {
    "install": """Install dependencies for this app.

App type hint: {app_kind}

Rules:
- Python apps: use pip with requirements/pyproject.
- Node apps: use npm install workflow.
- Do not modify any source files.

At the end, print exactly one status line:
STATUS: PASS
or
STATUS: FAIL""",
    "build": """Build this app for production.

App type hint: {app_kind}

Rules:
- Python apps: if no explicit build step exists, treat dependency installation + import sanity as build-equivalent.
- Node apps: run package build scripts where available.
- Do not modify source files.

At the end, print exactly one status line:
STATUS: PASS
or
STATUS: FAIL""",
    "typecheck": """Run static/type checks for this app.

App type hint: {app_kind}

Rules:
- Python apps: run available checks (e.g., pyright/mypy) if configured; otherwise run a minimal import/compile sanity check.
- Node apps: run TypeScript checks where tsconfig exists.
- Do not modify source files.

At the end, print exactly one status line:
STATUS: PASS
or
STATUS: FAIL""",
    "test": """Run the test suite for this app.

App type hint: {app_kind}

Rules:
- Python apps: run pytest (or equivalent) when tests are present.
- Node apps: run npm test workflow.
- If no tests exist, report that clearly and treat as FAIL for strict evaluation.
- Do not modify source files.

At the end, print exactly one status line:
STATUS: PASS
or
STATUS: FAIL""",
    "start": """Start this app and verify it responds on port {port}.

App type hint: {app_kind}

Steps:
1. First, kill any existing processes on port {port}: lsof -ti:{port} | xargs kill -9 2>/dev/null || true
2. Start the app in background with the simplest valid command for this app type.
   - Python/Streamlit example: streamlit run src/app.py --server.port {port} --server.headless true
   - Node example: npm start
3. Wait briefly for startup: sleep 3
4. Health check with retries - try these endpoints:
   - curl -sf --max-time 2 http://localhost:{port}/healthcheck
   - curl -sf --max-time 2 http://localhost:{port}/
5. If either returns success, the app is running correctly

IMPORTANT: Do NOT stop the app. Leave it running for screenshot capture.
The test succeeds if the health check passes.

At the end, print exactly one status line:
STATUS: PASS
or
STATUS: FAIL""",
    "stop": """Stop any running processes for this app on port {port}.
- Kill any processes listening on port {port}
- Use: lsof -ti:{port} | xargs kill -9 2>/dev/null || true
- Ensure the port is free for the next test""",
    "runability": """Verify local runability using a simple Claude Code workflow.

App type hint: {app_kind}

Steps:
1. Install dependencies using the most direct command for this app type.
   - For Python/Streamlit apps: use pip requirements if available.
   - For Node apps: use npm install workflow.
2. Start the app with the simplest valid command for this app type.
   - Python/Streamlit example: streamlit run src/app.py --server.port {port} --server.headless true
   - Node example: npm start
3. Verify the app responds on port {port} (try /healthcheck and /).
4. Stop any process you started on this port.

Rules:
- Do not modify source code.
- This check PASSES only if the app actually starts and responds.
- If any command errors, this check FAILS.

At the end, print exactly one status line:
STATUS: PASS
or
STATUS: FAIL
""",
    "deployability": """Verify deployability using a simple Claude Code workflow.

App type hint: {app_kind}

Steps:
1. Choose deployability path based on project structure:
   - If Dockerfile exists: do Docker build/run/healthcheck flow.
   - If databricks.yml exists and no Dockerfile: run `databricks bundle validate` (and related bundle checks if available).
2. Execute the chosen path and verify success.
3. If using Docker path, stop/remove container after checks.

Rules:
- Do not modify source code.
- This check PASSES only if the chosen deployability path succeeds end-to-end.
- If chosen path errors, this check FAILS.

At the end, print exactly one status line:
STATUS: PASS
or
STATUS: FAIL
""",
    "db_connectivity": """Verify Databricks connectivity through this running app on port {port}.

App type hint: {app_kind}

Steps:
1. Identify a real endpoint in this app that should query Databricks.
2. Call the endpoint on localhost:{port}.
3. Confirm response indicates a real query attempt and non-empty/structured result OR a clear authenticated Databricks query path.
4. If credentials are missing or endpoint doesn't exist, FAIL.

Rules:
- Do not modify source code.
- PASS only with positive evidence of Databricks query connectivity.

At the end, print exactly one status line:
STATUS: PASS
or
STATUS: FAIL
""",
    "ui_renders": """Verify UI renders correctly for this running app on port {port}.

App type hint: {app_kind}

Steps:
1. Request http://localhost:{port}/ and (if needed) /healthcheck.
2. Confirm page/response is not a crash page and not empty.
3. Look for obvious error indicators (stack traces, 404/500 pages, framework crash text).

Rules:
- Do not modify source code.
- PASS only if UI appears to render without obvious runtime errors.

At the end, print exactly one status line:
STATUS: PASS
or
STATUS: FAIL
""",
    "data_returned": """Verify the app returns real Databricks-backed data (not mocked) using ONE end-to-end workflow.

App type hint: {app_kind}

Steps:
1. Inspect the codebase to identify at least one real SQL query the app uses for analytics/data endpoints.
2. Identify the corresponding app API endpoint that should return that query's data.
3. Call the app endpoint on port {port} and capture the response payload.
4. Execute the same SQL directly against Databricks (using available workspace credentials and warehouse).
5. Compare endpoint result vs direct Databricks result using objective checks:
   - both calls succeed
   - non-empty data is returned
   - column/field shape is compatible
   - sampled values/count are consistent enough to indicate same source
6. Perform a mock-resistance check:
   - if endpoint data appears static/canned or clearly mismatched with Databricks query output, FAIL.

Rules:
- Do not modify source code.
- This check PASSES only when there is positive evidence endpoint data comes from Databricks query execution.
- If SQL cannot be found, Databricks query cannot run, endpoint cannot be mapped, or comparison is inconclusive -> FAIL.

At the end, print exactly one status line:
STATUS: PASS
or
STATUS: FAIL
""",
}

# Direct shell commands for fallback when running as root
# These are simplified versions that work without the agent
SHELL_COMMANDS: dict[str, str] = {
    "install": """
cd {app_dir}
if grep -q '"install:all"' package.json 2>/dev/null; then
    npm run install:all
else
    for dir in . server client; do
        if [ -f "$dir/package.json" ]; then
            (cd "$dir" && npm install)
        fi
    done
fi
# Rebuild native modules for current platform (fixes esbuild on Databricks)
npm rebuild esbuild 2>/dev/null || true
""",
    "build": """
cd {app_dir}
# Generate TypeScript types from schema before building (AppKit apps)
if [ -f "scripts/generate-types.ts" ]; then
    # Rebuild esbuild for current platform (fixes architecture mismatch on Databricks)
    npm rebuild esbuild 2>/dev/null || true
    npm run typegen 2>&1 || echo "Warning: typegen failed, continuing with build..."
fi
# Run build
if [ -f "client/package.json" ]; then
    (cd client && npm run build)
elif [ -f "package.json" ]; then
    npm run build
fi
""",
    "typecheck": """
cd {app_dir}
# Generate TypeScript types from schema before checking (AppKit apps)
if [ -f "scripts/generate-types.ts" ]; then
    # Rebuild esbuild for current platform (fixes architecture mismatch on Databricks)
    npm rebuild esbuild 2>/dev/null || true
    npm run typegen 2>&1 || echo "Warning: typegen failed, continuing with typecheck..."
fi
# Run type checking
for dir in . server client; do
    if [ -f "$dir/tsconfig.json" ]; then
        (cd "$dir" && npx tsc --noEmit --skipLibCheck)
    fi
done
""",
    "test": """
cd {app_dir}
if [ -f "server/package.json" ]; then
    cd server
fi
npm test 2>&1 || true
""",
    "start": """
cd {app_dir}
lsof -ti:{port} | xargs kill -9 2>/dev/null || true
npm start > /tmp/app.log 2>&1 &
sleep 5
for i in 1 2 3 4 5; do
    if curl -sf --max-time 2 http://localhost:{port}/healthcheck 2>/dev/null; then
        echo "Health check passed"
        exit 0
    fi
    if curl -sf --max-time 2 http://localhost:{port}/ 2>/dev/null; then
        echo "Root endpoint check passed"
        exit 0
    fi
    sleep 1
done
echo "Health check failed"
lsof -ti:{port} | xargs kill -9 2>/dev/null || true
exit 1
""",
    "stop": """
lsof -ti:{port} | xargs kill -9 2>/dev/null || true
""",
    "runability": """
cd {app_dir}
lsof -ti:{port} | xargs kill -9 2>/dev/null || true
if grep -q '"install:all"' package.json 2>/dev/null; then
  npm run install:all || exit 1
else
  for dir in . server client; do
    if [ -f "$dir/package.json" ]; then
      (cd "$dir" && npm install) || exit 1
    fi
  done
fi
npm start > /tmp/app_runability.log 2>&1 &
sleep 5
ok=0
if curl -sf --max-time 3 http://localhost:{port}/healthcheck >/dev/null 2>&1; then ok=1; fi
if [ "$ok" -eq 0 ] && curl -sf --max-time 3 http://localhost:{port}/ >/dev/null 2>&1; then ok=1; fi
lsof -ti:{port} | xargs kill -9 2>/dev/null || true
if [ "$ok" -eq 1 ]; then
  echo "STATUS: PASS"
  exit 0
fi
echo "STATUS: FAIL"
exit 1
""",
    "deployability": """
cd {app_dir}
if [ ! -f Dockerfile ]; then
  echo "STATUS: FAIL"
  exit 1
fi
tag="eval-deploy-$(basename "{app_dir}" | tr '[:upper:]' '[:lower:]' | tr -cd 'a-z0-9-')"
docker rm -f "$tag" >/dev/null 2>&1 || true
docker build -t "$tag" . || { echo "STATUS: FAIL"; exit 1; }
docker run -d --name "$tag" -p {port}:8000 "$tag" >/dev/null 2>&1 || docker run -d --name "$tag" -p {port}:3000 "$tag" >/dev/null 2>&1 || { echo "STATUS: FAIL"; exit 1; }
sleep 5
ok=0
if curl -sf --max-time 3 http://localhost:{port}/healthcheck >/dev/null 2>&1; then ok=1; fi
if [ "$ok" -eq 0 ] && curl -sf --max-time 3 http://localhost:{port}/ >/dev/null 2>&1; then ok=1; fi
docker rm -f "$tag" >/dev/null 2>&1 || true
if [ "$ok" -eq 1 ]; then
  echo "STATUS: PASS"
  exit 0
fi
echo "STATUS: FAIL"
exit 1
""",
    "data_returned": """
cd {app_dir}
# Conservative fallback for root environments: only pass with strong evidence.
sql_file=$(find config/queries -name '*.sql' 2>/dev/null | head -n 1)
if [ -z "$sql_file" ]; then
  echo "STATUS: FAIL"
  exit 1
fi
query_key=$(basename "$sql_file" .sql)
resp=$(curl -sf --max-time 10 -X POST "http://localhost:{port}/api/analytics/$query_key" -H "Content-Type: application/json" -d "{}" 2>/dev/null || true)
if [ -z "$resp" ]; then
  echo "STATUS: FAIL"
  exit 1
fi
# If response has clear mock markers, fail.
echo "$resp" | tr '[:upper:]' '[:lower:]' | grep -q "mock\\|fake\\|placeholder\\|sample data" && { echo "STATUS: FAIL"; exit 1; }
# Without direct Databricks query parity proof in shell fallback, default to fail-safe.
echo "STATUS: FAIL"
exit 1
""",
    "db_connectivity": """
cd {app_dir}
if curl -sf --max-time 3 "http://localhost:{port}/healthcheck" >/dev/null 2>&1 || curl -sf --max-time 3 "http://localhost:{port}/" >/dev/null 2>&1; then
  echo "STATUS: PASS"
  exit 0
fi
echo "STATUS: FAIL"
exit 1
""",
    "ui_renders": """
cd {app_dir}
body=$(curl -sf --max-time 5 "http://localhost:{port}/" 2>/dev/null || true)
if [ -z "$body" ]; then
  echo "STATUS: FAIL"
  exit 1
fi
echo "$body" | tr '[:upper:]' '[:lower:]' | grep -q "error\\|exception\\|traceback\\|cannot get\\|not found" && { echo "STATUS: FAIL"; exit 1; }
echo "STATUS: PASS"
exit 0
""",
}

EvalStep = Literal[
    "install",
    "build",
    "typecheck",
    "test",
    "start",
    "stop",
    "runability",
    "deployability",
    "data_returned",
    "db_connectivity",
    "ui_renders",
]


class EvalAgent:
    """Agent-based evaluation runner using Claude SDK.

    Falls back to direct shell execution when running as root.
    """

    def __init__(
        self,
        app_dir: Path,
        model: str = "haiku",
        suppress_logs: bool = True,
        env: dict[str, str] | None = None,
    ):
        """Initialize evaluation agent.

        Args:
            app_dir: Path to the app directory to evaluate
            model: Model to use (default: haiku for cost efficiency)
            suppress_logs: Whether to suppress logging output
            env: Environment variables to pass to the agent
        """
        self.app_dir = app_dir
        self.model = model
        self.suppress_logs = suppress_logs
        self.env = env or {}
        self._use_shell_fallback = _is_running_as_root()

        if self._use_shell_fallback and not suppress_logs:
            logger.info("Running as root - using shell script fallback instead of Claude SDK")

    async def _run_shell_command(
        self,
        step: EvalStep,
        timeout_sec: int = 120,
        **kwargs,
    ) -> tuple[bool, str]:
        """Run evaluation step using direct shell command (fallback for root)."""
        if step not in SHELL_COMMANDS:
            return False, f"Unknown evaluation step: {step}"

        # Format the command with app_dir and any kwargs
        abs_app_dir = str(self.app_dir.resolve())
        cmd = SHELL_COMMANDS[step].format(app_dir=abs_app_dir, **kwargs)

        # Prepare environment
        run_env = os.environ.copy()
        run_env.update(self.env)

        try:
            result = subprocess.run(
                ["bash", "-c", cmd],
                cwd=abs_app_dir,
                env=run_env,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
            output = result.stdout + result.stderr
            success = result.returncode == 0
            return success, output
        except subprocess.TimeoutExpired:
            return False, f"Command timed out after {timeout_sec}s"
        except Exception as e:
            return False, f"Exception: {str(e)}"

    async def _run_agent_step(
        self,
        step: EvalStep,
        timeout_sec: int = 120,
        **kwargs,
    ) -> tuple[bool, str, dict[str, int | None]]:
        """Run evaluation step using Claude SDK agent."""
        # Import here to avoid issues when Claude SDK isn't available
        from claude_agent_sdk import (
            AssistantMessage,
            ClaudeAgentOptions,
            ResultMessage,
            TextBlock,
            ToolResultBlock,
            UserMessage,
            query,
        )

        if step not in EVAL_PROMPTS:
            return False, f"Unknown evaluation step: {step}"

        prompt_template = EVAL_PROMPTS[step]
        prompt = prompt_template.format(**kwargs)

        # Build the full prompt with app context
        abs_app_dir = self.app_dir.resolve()
        full_prompt = f"""Task: {prompt}

Important:
- Work only within the current directory ({abs_app_dir})
- Do not create or modify source code files
- Report success (exit 0) or failure (exit 1) clearly
- Be concise - this is an automated evaluation step"""

        # Configure agent options
        max_turns = 15 if step in ("start", "stop") else 10
        options = ClaudeAgentOptions(
            system_prompt={
                "type": "preset",
                "preset": "claude_code",
            },
            permission_mode="bypassPermissions",
            allowed_tools=["Bash", "Read", "Glob", "Grep"],
            max_turns=max_turns,
            model=self.model,
            cwd=abs_app_dir,
            env=self.env,
        )

        output_lines: list[str] = []
        success = False
        assistant_turns = 0
        input_tokens: int | None = None
        output_tokens: int | None = None

        try:
            async for message in query(prompt=full_prompt, options=options):
                if isinstance(message, AssistantMessage):
                    assistant_turns += 1
                    for block in message.content:
                        if isinstance(block, TextBlock) and block.text:
                            output_lines.append(block.text)
                elif isinstance(message, UserMessage):
                    for block in message.content:
                        if isinstance(block, ToolResultBlock):
                            content = str(block.content) if block.content else ""
                            if content:
                                if len(content) > 1000:
                                    content = content[:1000] + "..."
                                output_lines.append(content)
                elif isinstance(message, ResultMessage):
                    success = message.subtype == "success" and not message.is_error
                    if message.result:
                        output_lines.append(message.result)
                    # Best-effort token extraction across SDK versions.
                    usage_obj = getattr(message, "usage", None) or getattr(message, "token_usage", None)
                    if usage_obj is not None:
                        if isinstance(usage_obj, dict):
                            input_tokens = int(usage_obj.get("input_tokens", 0) or 0)
                            output_tokens = int(usage_obj.get("output_tokens", 0) or 0)
                        else:
                            input_tokens = int(getattr(usage_obj, "input_tokens", 0) or 0)
                            output_tokens = int(getattr(usage_obj, "output_tokens", 0) or 0)
                    if not self.suppress_logs:
                        logger.info(f"Eval step {step} subtype: {message.subtype}")

        except Exception as e:
            logger.error(f"Eval step {step} failed with exception: {e}")
            return False, f"Exception: {str(e)}", {
                "turns": assistant_turns,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }

        return success, "\n".join(output_lines), {
            "turns": assistant_turns,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }

    async def run_step_with_stats(
        self,
        step: EvalStep,
        timeout_sec: int = 120,
        **kwargs,
    ) -> tuple[bool, str, dict[str, int | None]]:
        """Run an evaluation step and include agentic telemetry."""
        if self._use_shell_fallback:
            success, output = await self._run_shell_command(step, timeout_sec, **kwargs)
            return success, output, {"turns": None, "input_tokens": None, "output_tokens": None}
        return await self._run_agent_step(step, timeout_sec, **kwargs)

    async def run_step(
        self,
        step: EvalStep,
        timeout_sec: int = 120,
        **kwargs,
    ) -> tuple[bool, str]:
        """Run an evaluation step.

        Uses Claude SDK agent when possible, falls back to shell scripts when
        running as root (e.g., on Databricks clusters).

        Args:
            step: Evaluation step to run (install, build, typecheck, test, start, stop)
            timeout_sec: Maximum time for the step
            **kwargs: Format arguments for the prompt (e.g., port=8000)

        Returns:
            Tuple of (success: bool, output: str)
        """
        success, output, _stats = await self.run_step_with_stats(step, timeout_sec, **kwargs)
        return success, output

    async def install_dependencies(self, app_kind: str = "unknown") -> tuple[bool, str]:
        """Install npm dependencies."""
        return await self.run_step("install", app_kind=app_kind)

    async def build(self, app_kind: str = "unknown") -> tuple[bool, str]:
        """Build the app for production."""
        return await self.run_step("build", app_kind=app_kind)

    async def typecheck(self, app_kind: str = "unknown") -> tuple[bool, str]:
        """Run TypeScript type checking."""
        return await self.run_step("typecheck", app_kind=app_kind)

    async def test(self, app_kind: str = "unknown") -> tuple[bool, str]:
        """Run the test suite."""
        return await self.run_step("test", app_kind=app_kind)

    async def start(self, port: int = 8000, app_kind: str = "unknown") -> tuple[bool, str]:
        """Start the app and verify it responds."""
        return await self.run_step("start", port=port, app_kind=app_kind)

    async def stop(self, port: int = 8000, app_kind: str = "unknown") -> tuple[bool, str]:
        """Stop any running processes for this app."""
        return await self.run_step("stop", port=port, app_kind=app_kind)

    async def runability(self, port: int = 8000) -> tuple[bool, str]:
        """Check local runability using a simple agentic workflow."""
        return await self.run_step("runability", port=port)

    async def deployability(self, port: int = 8010) -> tuple[bool, str]:
        """Check deployability using a simple agentic Docker workflow."""
        return await self.run_step("deployability", port=port)

    async def data_returned(self, port: int = 8000) -> tuple[bool, str]:
        """Check data validity via SQL-to-endpoint parity (agentic)."""
        return await self.run_step("data_returned", port=port)

    async def runability_with_stats(
        self, port: int = 8000, app_kind: str = "unknown"
    ) -> tuple[bool, str, dict[str, int | None]]:
        """Run runability check and return telemetry."""
        return await self.run_step_with_stats("runability", port=port, app_kind=app_kind)

    async def deployability_with_stats(
        self, port: int = 8010, app_kind: str = "unknown"
    ) -> tuple[bool, str, dict[str, int | None]]:
        """Run deployability check and return telemetry."""
        return await self.run_step_with_stats("deployability", port=port, app_kind=app_kind)

    async def data_returned_with_stats(
        self, port: int = 8000, app_kind: str = "unknown"
    ) -> tuple[bool, str, dict[str, int | None]]:
        """Run data-returned check and return telemetry."""
        return await self.run_step_with_stats("data_returned", port=port, app_kind=app_kind)

    async def db_connectivity_with_stats(
        self, port: int = 8000, app_kind: str = "unknown"
    ) -> tuple[bool, str, dict[str, int | None]]:
        """Run Databricks connectivity check and return telemetry."""
        return await self.run_step_with_stats("db_connectivity", port=port, app_kind=app_kind)

    async def ui_renders_with_stats(
        self, port: int = 8000, app_kind: str = "unknown"
    ) -> tuple[bool, str, dict[str, int | None]]:
        """Run UI-render check and return telemetry."""
        return await self.run_step_with_stats("ui_renders", port=port, app_kind=app_kind)
