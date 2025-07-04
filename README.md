<div align="center">
  <img src="logo.png" alt="app.build logo" width="150">
</div>

# app.build (agent)

**app.build** is an open-source AI agent for generating production-ready full-stack applications from a single prompt.

## What it builds

- **Full-stack web apps** with Bun, React, Vite, Fastify and Drizzle
- **Neon Postgres database** provisioned instantly via API
- **Applications tested** ahead of generation with smoke tests using Playwright
- **GitHub repository** with complete source code
- **CI/CD and deployment** via the [app.build platform](https://github.com/appdotbuild/platform)
- **Automatic validation** with ESLint, TypeScript, and runtime verification

## Try it

```bash
npx @app.build/cli
```

## Architecture

This agent doesn't generate entire applications at once. Instead, it breaks down app creation into small, well-scoped tasks that run in isolated sandboxes:

1. **Database schema generation** - Creates typed database models
2. **API handler logic** - Builds validated Fastify routes
3. **Frontend components** - Generates React UI with proper typing

Each task is validated independently using ESLint, TypeScript compilation, test execution, and runtime logs before being accepted.

More details on the architecture can be found in the [blog on our design decisions](https://www.app.build/blog/design-decisions).

## Repository structure

This is the **agent** repository containing the core code generation engine and runtime environment. The CLI and platform code are available in the [platform repository](https://github.com/appdotbuild/platform).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and contribution guidelines.

## Running locally

Local development instructions are available in [CONTRIBUTING.md](CONTRIBUTING.md).

---

Built to showcase agent-native infrastructure patterns. Fork it, remix it, use it as a reference for your own projects.
