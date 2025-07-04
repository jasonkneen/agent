import os
import pytest
import tempfile
import anyio
import contextlib

from fire import Fire
from api.agent_server.agent_client import AgentApiClient, MessageKind
from api.agent_server.agent_api_client import apply_patch, latest_unified_diff, DEFAULT_APP_REQUEST, DEFAULT_EDIT_REQUEST, spawn_local_server
from api.docker_utils import setup_docker_env, start_docker_compose, wait_for_healthy_containers, stop_docker_compose, get_container_logs
from log import get_logger
from tests.test_utils import requires_llm_provider, requires_llm_provider_reason

logger = get_logger(__name__)

pytestmark = pytest.mark.anyio


@contextlib.contextmanager
def empty_context():
    yield

@pytest.fixture
def anyio_backend():
    return 'asyncio'

def latest_app_name_and_commit_message(events):
    """Extract the most recent app_name and commit_message from events"""
    app_name = None
    commit_message = None

    for evt in reversed(events):
        try:
            if evt.message:
                # Update app_name if found and not yet set
                if app_name is None and evt.message.app_name is not None:
                    app_name = evt.message.app_name

                # Update commit_message if found and not yet set
                if commit_message is None and evt.message.commit_message is not None:
                    commit_message = evt.message.commit_message

                # If both are set, we can break
                if app_name is not None and commit_message is not None:
                    break
        except AttributeError:
            continue

    return app_name, commit_message

async def run_e2e(prompt: str, standalone: bool, with_edit=True, template_id=None):
    context = empty_context() if standalone else spawn_local_server()
    with context:
        async with AgentApiClient() as client:
            events, request = await client.send_message(prompt, template_id=template_id)
            assert events, "No response received from agent"
            while events[-1].message.kind == MessageKind.REFINEMENT_REQUEST:
                events, request = await client.continue_conversation(
                    previous_events=events,
                    previous_request=request,
                    message="just do it! no more questions, please",
                    template_id=template_id,
                )

            diff = latest_unified_diff(events)
            assert diff, "No diff was generated in the agent response"

            # Check that app_name and commit_message are present in the response
            app_name, commit_message = latest_app_name_and_commit_message(events)
            assert app_name is not None, "No app_name was generated in the agent response"
            assert commit_message is not None, "No commit_message was generated in the agent response"
            logger.info(f"Generated app_name: {app_name}")
            logger.info(f"Generated commit_message: {commit_message}")

            if with_edit:
                new_events, new_request = await client.continue_conversation(
                    previous_events=events,
                    previous_request=request,
                    message=DEFAULT_EDIT_REQUEST,
                    template_id=template_id,
                )
                updated_diff = latest_unified_diff(new_events)
                assert updated_diff, "No diff was generated in the agent response after edit"
                assert updated_diff != diff, "Edit did not produce a new diff"
            else:
                updated_diff = diff

            with tempfile.TemporaryDirectory() as temp_dir:
                # Determine template path based on template_id
                template_paths = {
                    "nicegui_agent": "nicegui_agent/template",
                    "trpc_agent": "trpc_agent/template",
                    None: "trpc_agent/template"  # default
                }

                for d in (diff, updated_diff):
                    success, message = apply_patch(d, temp_dir, template_paths[template_id])
                    assert success, f"Failed to apply patch: {message}"

                original_dir = os.getcwd()
                container_names = setup_docker_env()

                try:
                    os.chdir(temp_dir)

                    success, error_message = start_docker_compose(temp_dir, container_names["project_name"])
                    if not success:
                        # Get logs if possible for debugging
                        try:
                            logs = get_container_logs([
                                container_names["db_container_name"],
                                container_names["app_container_name"],
                            ])
                            for container, log in logs.items():
                                logger.error(f"Container {container} logs: {log}")
                        except Exception:
                            logger.error("Failed to get container logs")

                        logger.error(f"Error starting Docker containers: {error_message}")
                        raise RuntimeError(f"Failed to start Docker containers: {error_message}")

                    container_healthy = await wait_for_healthy_containers(
                        [
                            container_names["db_container_name"],
                            container_names["app_container_name"],
                        ],
                        ["db", "app"],
                        timeout=30,
                        interval=1
                    )

                    if not container_healthy:
                        raise RuntimeError("Containers did not become healthy within the timeout period")

                    if standalone:
                        input(f"App is running on http://localhost:80/, app dir is {temp_dir}; Press Enter to continue and tear down...")
                        print("🧹Tearing down containers... ")

                finally:
                    # Restore original directory
                    os.chdir(original_dir)

                    # Clean up Docker containers
                    stop_docker_compose(temp_dir, container_names["project_name"])

@pytest.mark.parametrize("template_id", [
    pytest.param("nicegui_agent", marks=pytest.mark.nicegui),
])
async def test_e2e_generation_nicegui(template_id):
    await run_e2e(standalone=False, prompt=DEFAULT_APP_REQUEST, template_id=template_id)

@pytest.mark.skipif(requires_llm_provider(), reason=requires_llm_provider_reason)
@pytest.mark.parametrize("template_id", [
    pytest.param("trpc_agent", marks=pytest.mark.trpc)
])
async def test_e2e_generation_trpc(template_id):
    await run_e2e(standalone=False, prompt=DEFAULT_APP_REQUEST, template_id=template_id)


def create_app(prompt):
    import coloredlogs

    coloredlogs.install(level="INFO")
    anyio.run(run_e2e, prompt, True)


if __name__ == "__main__":
    Fire(create_app)
