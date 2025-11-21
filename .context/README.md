# Codebase Context Directory

This directory follows the [Codebase Context Specification (CCS) v1.1](https://github.com/nascousa/ccs).

## Structure

```
.context/
├── index.md              # Primary context with project overview & architecture
├── docs.md              # Detailed technical documentation
├── diagrams/            # Visual documentation in Mermaid format
│   ├── architecture.mmd # High-level system architecture
│   ├── data-flow.mmd    # Training pipeline data flow
│   └── unet-model.mmd   # U-Net model architecture details
└── README.md            # This file
```

## Purpose

The `.context` directory provides comprehensive documentation for:
- **Human Developers**: Quick onboarding and architectural understanding
- **AI Tools**: Structured context for GitHub Copilot and other AI assistants
- **Maintainers**: Living documentation that evolves with the codebase

## How to Use

### For Developers
Start with `index.md` for project overview, then reference `docs.md` for detailed implementation guidance.

### For AI Tools
AI development tools should:
1. Read `.context/index.md` first for architectural overview
2. Parse YAML front matter for structured project metadata
3. Reference `docs.md` for detailed implementation patterns
4. Visualize `diagrams/*.mmd` files for system understanding

### Viewing Diagrams
Mermaid diagrams can be viewed using:
- GitHub's built-in Mermaid renderer
- VS Code with Mermaid extension
- Online: https://mermaid.live/

## Maintenance

This documentation should be:
- ✅ Updated alongside code changes
- ✅ Reviewed during code review
- ✅ Kept consistent with implementation
- ✅ Version controlled with the code

## Quick Links

- **Project Proposal**: `../docs/proposal.md`
- **Course Guidelines**: `../docs/guideline.md`
- **GitHub Copilot Instructions**: `../.github/copilot-instructions.md`
- **Example Implementation**: `../example/TrainingForClassification/`
