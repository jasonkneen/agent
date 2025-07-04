# Project Guidelines

This file contains the core guidelines for working with code in this repository.
AI assistants (Cursor, Claude, Devin, etc.) should refer to this file for context.

# Project Overview

This project implements an AI codegen agent system with various components for handling API interactions, core logic, and specific agent implementations. The system is built primarily in Python with some TypeScript components.

Refer to `agent/architecture.puml` for a visual representation of the system architecture.

# Project Structure

- `agent` - Contains the main codegen agent code
  - `api` - IO layer for the agent
    - `agent_server` - API of the agent server
      - `models.py` - Models for the agent server consistent with agent_api.tsp
      - `agent_api.tsp` - Server type specification for the agent server
      - `async_server.py` - Agent server implementation
      - `async_agent_session.py` - DEPRECATED: Use trpc_agent/agent_session.py instead
    - `cli` - CLI entrypoint (Note: Interacted via test clients, not a dedicated CLI app)
  - `core` - Core framework logic (base classes, state machine, etc.)
  - `trpc_agent` - Agent for fullstack code generation (new agents follow this pattern)
  - `llm` - LLM wrappers
  - `stash_bot` - DEPRECATED: Use trpc_agent implementation instead
  - `log.py` - Global logging and tracing

# Development Workflow

Commands should typically be run from the `./agent` directory:

- **Run all tests**: `uv run pytest -v .`
- **Lint code**: `uv run ruff check` 
- **Format code**: `uv run ruff format`
- **Run tests in isolated env**: `docker build --target test -t agent-test:latest . && docker run --rm agent-test:latest`
- **Build Docker image**: `docker build -t agent:latest .`
- **Run Docker container**: `docker run --rm -p 8000:8000 agent:latest`

# Lessons

## User Specified Lessons
- Using `anyio` instead of `asyncio` for async operations is preferred in this codebase
- Always activate the Python virtual environment before installing packages
- Never use mocks in tests unless explicitly required

## Cursor Learned

Track learnings and work progress here.

### SSE Message Handling Enhancement (Latest)

**Problem Solved**: Fixed JSON double-encoding issue in SSE events where content was being JSON-encoded twice, resulting in escaped JSON strings like `{\"status\":\"idle\",\"content\":\"[{\\\"role\\\"...}]\"...}`.

**Solution Implemented**:
1. **Model Changes**: Updated `AgentMessage` to use structured `messages` field (`List[ExternalContentBlock]`) instead of string `content` field
2. **Backward Compatibility**: Kept deprecated `content` field for legacy clients  
3. **Agent Session**: Modified `send_event()` to populate both fields - structured messages for new clients, JSON string for legacy
4. **Interactive Client**: Enhanced `print_event()` function to:
   - Use new structured `messages` field when available
   - Fall back to deprecated `content` field for backward compatibility
   - Display timestamps with messages in format `[HH:MM:SS] message content`
   - Handle multi-line content properly with indentation

**New Interactive Commands Added**:
- `/messages` - Show detailed message history with timestamps, event metadata, and structured formatting
- `/last` - Show latest messages from most recent event for quick review

**Technical Details**:
- `ExternalContentBlock` contains: `content: str`, `timestamp: datetime`, `role: Literal["assistant"]`  
- Messages are displayed with subtle timestamp formatting: `[90m[HH:MM:SS][0m content`
- Event status (idle/running) and kind (StageResult/ReviewResult/etc) shown with color coding
- Graceful handling of legacy JSON content for backward compatibility

This eliminates the double-JSON-encoding issue while providing a better user experience with structured, timestamped messages.

**Architectural Improvement**: Moved user-facing message formatting logic from `llm/common.py` to `api/agent_server/models.py` via `format_internal_message_for_display()` function, centralizing SSE-related user interface logic

**User-Friendly Tool Reporting**:
- Tool uses now display with emoji icons and descriptive names instead of technical format
- Context extraction shows relevant information (file paths, commands, descriptions) without JSON dumps
- Tool results show completion status with appropriate success/error indicators
- Before: `[Tool Use: start_fsm] Input: {"app_description": "..."}`
- After: `🚀 Starting application development | Building: A simple application that displays...`

This eliminates the double-JSON-encoding issue while providing a better user experience with structured, timestamped messages.

# Code Style Guidelines

## Python

### Key Libraries and Dependencies

- `anyio` - Preferred async library
- `pytest` - Testing framework
- `ruff` - Linting and formatting
- `uv` - Package management tool

### Code Organization

#### File Structure
- Keep modules focused on a single responsibility
- Group related functions/classes within a module
- Use `__init__.py` to expose public interfaces

### Formatting and Structure

### Imports
- **Order**: Standard library imports → Third-party library imports → Local/project imports
- **Example**:
  ```python
  import json
  import os
  from typing import Dict, List, Optional
  
  import anyio
  import pytest
  
  from agent.core import BaseClass
  from agent.log import get_logger
  ```

### Language Features

### Async Code
- **Preferred Library**: Use `anyio` rather than `asyncio`
- **Context Managers**: Use async context managers for resource management
- **Task Groups**: Group related async operations when possible
  ```python
  async with anyio.create_task_group() as tg:
      tg.start_soon(task_1)
      tg.start_soon(task_2)
  ```

### Type Annotations
- Annotate all function parameters and return values
- Use `Optional[Type]` or `Type | None` for parameters that can be None (prefer `Type | None`)
- Use union types with the pipe operator: `str | None`
- Use `TypedDict` for dictionary structures
- **Example**:
  ```python
  from typing import TypedDict, Any
  
  class Result(TypedDict):
      status: str
      data: Any
      
  def process_request(data: Dict[str, Any], timeout: float | None = None) -> Result:
      """
      Process the incoming request data.
      
      Args:
          data: The request data to process
          timeout: Optional timeout in seconds
          
      Returns:
          Result object containing processed data
      """
      # ... implementation ...
      return {"status": "success", "data": {}}
  ```

### Error Handling and Logging
- **Exception Types**: Use specific exception types rather than generic `Exception`. Avoid `except:` without specifying types.
- **Custom Exceptions**: Create custom exceptions for domain-specific errors.
- **Logging**: Always log exceptions with context using `logger.exception("message")` from `agent.log`.
- **Example**:
  ```python
  from agent.log import get_logger
  logger = get_logger(__name__)
  
  class InvalidDataError(ValueError):
      pass
  
  async def process_data(input_data):
      # ... processing ...
      if not valid:
          raise InvalidDataError("Input data failed validation")
      return processed_data
  
  async def handle_request(input_data):
      try:
          result = await process_data(input_data)
      except InvalidDataError as e:
          logger.exception("Invalid data format received")
          # Re-raise or handle specific error
          raise
      except Exception as e:
          logger.exception("An unexpected error occurred during processing")
          # Handle generic error
          raise
  ```

### Testing
- **Framework**: Use `pytest`
- **Structure**: 
    - Use descriptive test names starting with `test_`.
    - Group tests by functionality.
    - Use fixtures for common test setups.
- **Mocks**: Avoid mocks unless absolutely necessary.
- **Example**:
  ```python
  import pytest
  from your_module import process_request # Assuming process_request is defined elsewhere
  
  @pytest.fixture
  def sample_data():
      return {"key": "value"}
  
  def test_process_request_with_valid_data(sample_data):
      result = process_request(sample_data)
      assert result["status"] == "success"
  ```

### Common Pitfalls
- Avoid global state; use dependency injection.
- Be careful with default mutable arguments (e.g., `def func(arg=[]): ...`). Use `arg: list | None = None` and `if arg is None: arg = []` instead.
- Always close resources explicitly or use context managers.

## TypeScript

### Key Libraries and Dependencies

- `zod` - Schema validation
- `trpc` - API framework (Note: Primarily used for potential future frontend/backend interactions, not core agent logic)

### Code Organization

#### File Structure
- Use a modular approach with clear separation of concerns.
- Group related functionality in directories.
- Naming pattern: 
  - React components: `ComponentName.tsx` (If applicable)
  - Utility files: `utility-name.ts`
  - Type definitions: `types.ts` or within the module.

#### Import Order
1. React and framework imports (If applicable)
2. Third-party libraries (`zod`, `trpc`, etc.)
3. Local components and utilities
4. Types and interfaces
5. Styles and assets (If applicable)
- **Example**:
  ```typescript
  // import React, { useState, useEffect } from 'react'; // If using React
  
  import { z } from 'zod';
  // import { trpc } from 'trpc'; // If using tRPC client
  
  // import { Button } from '../components/Button'; // If using React components
  import { formatData } from '../utils/formatter';
  
  import type { User, UserPreferences } from '../types';
  
  // import styles from './styles.module.css'; // If using CSS Modules
  ```

### Types
- **Best Practices**: 
    - Use explicit interfaces over type aliases for objects.
    - Use Zod schemas for runtime validation, especially for API boundaries.
    - Export types from a central location or alongside their usage.
    - Use generics for reusable components and functions.
- **Example**:
  ```typescript
  import { z } from 'zod';
  
  // Define Zod schema for runtime validation
  export const userPreferencesSchema = z.object({
    theme: z.enum(['light', 'dark']),
    notifications: z.boolean(),
  });
  
  // Define interface using inferred type from Zod
  export type UserPreferences = z.infer<typeof userPreferencesSchema>;
  
  // Define main interface
  export interface User {
    id: string;
    name: string;
    email: string;
    preferences?: UserPreferences;
  }
  
  // Define Zod schema for User
  export const userSchema = z.object({
    id: z.string().uuid(),
    name: z.string().min(1),
    email: z.string().email(),
    preferences: userPreferencesSchema.optional(),
  });
  
  // Type for function arguments
  function updateUser(user: User, updates: Partial<User>): User {
    // Zod validation can be added here if needed
    // const validatedUpdates = userSchema.partial().parse(updates);
    return { ...user, ...updates }; 
  }
  ```

### Variables
- Prefer `const` over `let`.

### Naming
- Variables/Functions: `camelCase`
- Types/Interfaces: `PascalCase`

### Imports
- No renamed imports unless necessary for clarity or collision avoidance.

### API Integration (tRPC Example - If Applicable)
- **Patterns**: 
    - Define router schemas explicitly.
    - Use input validation for all procedures.
    - Centralize error handling.
- **Example**:
  ```typescript
  import { z } from 'zod';
  // Assume setup for router, procedure, TRPCError, and ctx (context) exists
  // import { router, procedure, TRPCError } from '../trpc'; 
  // import { db } from '../db'; // Example database context
  
  export const userRouter = router({
    getUser: procedure
      .input(z.object({ id: z.string() }))
      .query(async ({ input, ctx }) => {
        try {
          // Replace with actual data fetching logic
          // return await ctx.db.user.findUnique({ where: { id: input.id } });
          return { id: input.id, name: 'Dummy User' }; // Example return
        } catch (error) {
          console.error("Failed to fetch user:", error); // Use proper logging
          throw new TRPCError({
            code: 'INTERNAL_SERVER_ERROR',
            message: 'Failed to fetch user',
            // cause: error, // Optional: include original error
          });
        }
      }),
  });
  ```

### Component Structure (React Example - If Applicable)
- **Patterns**:
    - Use functional components with hooks.
    - Destructure props at the component level.
    - Define prop types using interfaces.
    - Use `React.FC` sparingly (prefer explicit return types like `JSX.Element`).
- **Example**:
  ```typescript
  import React from 'react'; // Assuming React is used
  // import styles from './styles.module.css'; // Example CSS Modules

  interface ButtonProps {
    label: string;
    onClick: () => void;
    disabled?: boolean;
  }
  
  function Button({ label, onClick, disabled = false }: ButtonProps): JSX.Element {
    return (
      <button 
        onClick={onClick}
        disabled={disabled}
        // className={styles.button} // Example styling
      >
        {label}
      </button>
    );
  }
  ```

### State Management (React Example - If Applicable)
- **Best Practices**: 
    - Use React hooks (`useState`, `useReducer`) for local state.
    - Consider context (`useContext`) for simple shared state.
    - Use dedicated state management libraries (like Zustand, Redux) for complex global state.
    - Keep state normalized.
- **Example using `useReducer`**:
  ```typescript
  // Assume reducer, initialState are defined
  // const [state, dispatch] = useReducer(reducer, initialState);
  
  // // Later in component
  // dispatch({ type: 'UPDATE_USER', payload: { id, data } });
  ```

### Error Handling
- **Patterns**: 
    - Use `try/catch` blocks for async operations.
    - Create custom error classes for domain-specific errors.
    - Use error boundaries for component errors (in React).
- **Example**:
  ```typescript
  class ApiError extends Error {
    constructor(message: string) {
      super(message);
      this.name = 'ApiError';
    }
  }
  
  async function fetchData() {
    // ... fetch logic that might throw ...
  }
  
  async function process() {
    let data;
    let error = null;
    try {
      data = await fetchData();
    } catch (err) {
      if (err instanceof ApiError) {
        error = `API Error: ${err.message}`;
      } else if (err instanceof Error) {
        error = `An unexpected error occurred: ${err.message}`;
        console.error(err); // Use proper logging
      } else {
        error = 'An unknown error occurred';
        console.error(err);
      }
    }
    // ... handle data or error ...
  }
  ```

### Testing (Example using Jest/React Testing Library - If Applicable)
- **Patterns**:
    - Use Jest and React Testing Library for UI components.
    - Test component behavior, not implementation details.
    - Mock API calls and external dependencies.
    - Use `data-testid` attributes for reliable test selectors.
- **Example**:
  ```typescript
  // import { render, screen, fireEvent } from '@testing-library/react';
  // import { UserProfile } from './UserProfile'; // Example component
  
  // const mockUser = { id: '1', name: 'Test User', email: 'test@example.com' };
  
  // test('displays user information correctly', () => {
  //   render(<UserProfile user={mockUser} />);
  
  //   expect(screen.getByText(mockUser.name)).toBeInTheDocument();
  //   expect(screen.getByText(mockUser.email)).toBeInTheDocument();
  
  //   // Example interaction
  //   // fireEvent.click(screen.getByRole('button', { name: /Edit Profile/i }));
  //   // expect(screen.getByTestId('edit-form')).toBeInTheDocument();
  // });
  ```

### Common Pitfalls
- Avoid `any` type; use `unknown` if type is truly unknown and perform checks.
- Be careful with nullish values; use optional chaining (`?.`) and nullish coalescing (`??`).
- Avoid type assertions (`as Type`) when possible; prefer type guards or narrowing.
- Don't use `==`; always use `===` for strict equality checks.
- Remember that empty objects (`{}`) are truthy.

# LLM Model Configuration (2025)

The agent system supports multiple LLM providers with category-based selection for optimal performance.

## Model Categories

The system uses three main categories:
- **FAST_TEXT**: Quick tasks (name generation, commit messages, simple text processing)
- **BEST_CODEGEN**: Complex code generation and reasoning tasks
- **VISION**: Image analysis and UI understanding

## Current Defaults (Updated Jan 2025)

```python
from llm.utils import get_fast_text_client, get_best_codegen_client, get_vision_client
from llm.models_config import ModelCategory

# Recommended usage - these use current defaults
fast_client = get_fast_text_client()        # Uses Ministral 3B
codegen_client = get_best_codegen_client()  # Uses Codestral 22B  
vision_client = get_vision_client()         # Uses LLaMA 3.2 Vision
```

## Latest Model Recommendations

Based on [recent performance benchmarks](https://medium.com/@rameshkannanyt0078/the-best-ollama-models-for-developers-and-programmers-1662dae514a7) and [Ollama model availability](https://ollama.com/), here are the current best options:

### For Local Development (Ollama)
- **Fast Tasks**: `ministral` (3B) - Fastest Mistral model
- **Code Generation**: `codestral` (22B) - Specialized for code
- **Vision Tasks**: `llama3.2-vision` - Latest Meta vision model
- **Advanced Coding**: `qwen3-coder` (7B) - Best coding performance
- **Large Tasks**: `qwen3` (30B) - Advanced reasoning with thinking mode

### For Cloud APIs
- **Fast Tasks**: `gemini-flash-lite` (Google)
- **Code Generation**: `sonnet` (Anthropic Claude)
- **Vision Tasks**: `gemini-flash` (Google)

## Environment Configuration

Create a `.env` file to override defaults:

```bash
# Recommended: Latest non-Chinese models (2025)
LLM_FAST_MODEL=ministral           # Mistral 3B
LLM_CODEGEN_MODEL=codestral        # Mistral 22B 
LLM_VISION_MODEL=llama3.2-vision   # Meta Vision

# Alternative: All-American models
# LLM_FAST_MODEL=phi4              # Microsoft 14B
# LLM_CODEGEN_MODEL=granite-code   # IBM 20B
# LLM_VISION_MODEL=llama3.2-vision # Meta Vision

# For highest performance (includes Chinese models)
# LLM_CODEGEN_MODEL=qwen3-coder    # Best for coding
# LLM_VISION_MODEL=qwen3           # Advanced vision + thinking

# API Keys (if using cloud models)
ANTHROPIC_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here
OLLAMA_HOST=http://localhost:11434
```

## Migration from Old Function Names

The system now provides both legacy and new function names:

```python
# Legacy (still works)
from llm.utils import get_fast_llm_client, get_codegen_llm_client, get_vision_llm_client

# New recommended names (same functionality)
from llm.utils import get_fast_text_client, get_best_codegen_client, get_vision_client

# Or use the category-based approach
from llm.utils import get_llm_client_by_category
from llm.models_config import ModelCategory

client = get_llm_client_by_category(ModelCategory.BEST_CODEGEN)
```

## Setting Up Ollama

To use local models, install and start Ollama:

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull recommended models
ollama pull ministral      # Fast tasks
ollama pull codestral:22b  # Code generation
ollama pull llama3.2-vision # Vision tasks

# Optional: Advanced models
ollama pull qwen3-coder:7b  # Excellent coding
ollama pull qwen3:30b       # Advanced reasoning
```

Then set `OLLAMA_HOST=http://localhost:11434` in your `.env` file.

## Model Performance Notes

- **Ministral (3B)**: Fastest for simple tasks, low memory usage
- **Codestral (22B)**: Best balance of performance and speed for coding
- **Qwen3-Coder (7B)**: Highest coding performance per parameter
- **LLaMA 3.2 Vision**: Most reliable for image analysis
- **Qwen3 (30B)**: Advanced reasoning with "thinking mode" capability

Choose models based on your hardware capabilities and performance requirements.

## Test LLM Provider Flexibility (Jan 2025)

**Problem Solved**: Tests were hardcoded to only run when `GEMINI_API_KEY` was set, making it impossible to run tests with Ollama or other LLM providers.

**Solution Implemented**:
1. **Centralized Provider Detection**: Created `tests/test_utils.py` with `is_llm_provider_available()` function that checks for any configured LLM provider:
   - Cloud providers: `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, `AWS_SECRET_ACCESS_KEY`/`PREFER_BEDROCK`
   - Local providers: `PREFER_OLLAMA` 
   - Explicit model overrides: `LLM_FAST_MODEL`, `LLM_CODEGEN_MODEL`, `LLM_VISION_MODEL`

2. **Updated Test Conditions**: Replaced hardcoded `@pytest.mark.skipif(os.getenv("GEMINI_API_KEY") is None)` with flexible `@pytest.mark.skipif(requires_llm_provider())` in:
   - `tests/test_e2e.py::test_e2e_generation` 
   - `tests/test_cached_llm.py` (4 tests)

3. **Provider-Agnostic Test Names**: Renamed tests from `test_gemini*` to `test_llm_*` to reflect their provider independence

**Usage Examples**:
```bash
# Run tests with Ollama
PREFER_OLLAMA=1 uv run pytest tests/test_e2e.py

# Run tests with Gemini  
GEMINI_API_KEY=your_key uv run pytest tests/test_e2e.py

# Run tests with Anthropic
ANTHROPIC_API_KEY=your_key uv run pytest tests/test_e2e.py

# Tests are skipped when no provider is configured
uv run pytest tests/test_e2e.py  # SKIPPED
```

This enables the test suite to work with any configured LLM provider, supporting both cloud and local development workflows.

**Critical Fix**: Updated `tests/test_utils.py` to load `.env` files at import time, ensuring environment variables are available when `@pytest.mark.skipif` conditions are evaluated during test collection phase.

