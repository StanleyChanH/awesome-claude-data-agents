# Creating Custom Data Science Agents

This guide explains how to create your own specialized data science agents for the Awesome Data Science Agents collection.

## Agent Structure

Every data science agent follows a standardized structure:

```yaml
---
name: agent-name
description: Expert [specialization] who [does what]. [Additional expertise]. Examples: <example>Context: [when to use]. user: "[example user request]" assistant: "I'll use agent-name [action]" <commentary>[why this agent is best]</commentary></example>
---

# Agent Name
[Brief description of agent's role and expertise]

## Core Expertise

[3-5 bullet points covering main areas of expertise]

## Analytical Framework

### 1. [Framework Component 1]
[Detailed explanation of methodology]

### 2. [Framework Component 2]
[Detailed explanation of methodology]

## Implementation Patterns

### [Pattern Name]
```python
[Code example showing implementation approach]
```

## Quality Standards

[Checklist or standards that agent follows]

## Deliverables

[What the agent typically produces]

Remember: [Key principle or reminder]
```

## Creating Your First Agent

### Step 1: Identify the Specialization

Choose a specific area of data science expertise that's not already covered by existing agents. Examples:

- Market basket analysis specialist
- Geospatial data analyst
- Experimental design expert
- Causal inference specialist
- Text mining expert
- Recommendation system specialist

### Step 2: Define the Agent Metadata

Create the YAML frontmatter with essential information:

```yaml
---
name: market-basket-analyst
description: Expert market basket analysis specialist who discovers product associations and purchasing patterns through advanced association rule mining and pattern recognition techniques. Examples: <example>Context: Retail analysis needs. user: "What products are frequently bought together?" assistant: "I'll use the market-basket-analyst to discover product associations" <commentary>Market-basket-analyst specializes in association rule mining</commentary></example>
---
```

### Step 3: Write the Agent System Prompt

Develop comprehensive system instructions covering:

**Core Expertise**: What specific analytical techniques and methods does this agent master?

**Analytical Framework**: How does the agent approach problems systematically?

**Implementation Patterns**: What code patterns and methodologies does the agent use?

**Quality Standards**: What standards and best practices does the agent follow?

### Step 4: Add Examples and Context

Include practical examples that show when and how to use the agent:

```yaml
Examples:
- <example>
  Context: Retail store wants to optimize product placement
  user: "Help us understand which products should be placed together"
  assistant: "I'll use the market-basket-analyst to analyze product associations and recommend optimal placement strategies"
  <commentary>Market-basket-analyst excels at discovering product relationships for business optimization</commentary>
</example>
```

## Agent Categories and Guidelines

### Analysis Agents
- Focus on data understanding and statistical analysis
- Provide rigorous statistical methodology
- Include hypothesis testing and validation
- Emphasize data quality and preprocessing

### Machine Learning Agents
- Cover end-to-end ML lifecycle
- Include model evaluation and validation
- Address production considerations
- Focus on reproducibility and best practices

### Visualization Agents
- Emphasize clear communication of insights
- Include both static and interactive techniques
- Address accessibility and design principles
- Focus on storytelling with data

### Core/Infrastructure Agents
- Focus on code quality and reproducibility
- Include testing and validation
- Address deployment and monitoring
- Emphasize documentation and maintainability

## Best Practices for Agent Design

### 1. Be Specific and Focused
- Each agent should have a clear, specific area of expertise
- Avoid creating overly general "do everything" agents
- Focus on depth rather than breadth of knowledge

### 2. Include Practical Examples
- Provide 2-3 concrete examples of when to use the agent
- Show the user request and agent response pattern
- Include commentary explaining why this agent is optimal

### 3. Emphasize Methodology
- Explain the analytical framework clearly
- Include code patterns and implementation approaches
- Define quality standards and deliverables

### 4. Ensure Reproducibility
- Include considerations for reproducible analysis
- Address data quality and validation
- Emphasize documentation and best practices

### 5. Consider Orchestration
- Design agents to work well with others
- Include handoff information for follow-up tasks
- Consider how the agent fits into larger workflows

## Testing Your Agent

### 1. Unit Testing
Create test scenarios to validate your agent:

```python
def test_market_basket_analyst():
    # Test with sample transaction data
    # Verify association rule mining works
    # Check interpretation quality
    # Validate output format
```

### 2. Integration Testing
Test how your agent works with others:

```python
def test_integration_with_data_explorer():
    # Verify handoff from data-explorer works
    # Test information passing
    # Validate workflow integration
```

### 3. User Acceptance Testing
Test with realistic user scenarios:

```python
def test_user_scenarios():
    # Test with real user requests
    # Verify response quality
    # Check for appropriate agent selection
```

## Submitting Your Agent

### 1. File Organization
- Place agents in appropriate category directory
- Use kebab-case for file names
- Include comprehensive documentation

### 2. Documentation
- Create README for your agent
- Include usage examples
- Document dependencies and requirements

### 3. Review Process
- Ensure code quality standards
- Validate agent specialization
- Test orchestration compatibility

## Agent Template

Use this template to create new agents:

```yaml
---
name: your-agent-name
description: Brief one-sentence description. Additional expertise area. Examples: <example>Context: When this agent is most useful. user: "Example user request" assistant: "I'll use your-agent-name to address this" <commentary>Why this agent is the best choice</commentary></example>
---

# Your Agent Name

Brief description of the agent's role and primary value proposition.

## Core Expertise

- [Key expertise area 1 with specific focus]
- [Key expertise area 2 with specific focus]
- [Key expertise area 3 with specific focus]
- [Key expertise area 4 with specific focus]

## Analytical Framework

### 1. [Framework Component Name]
[Detailed explanation of how the agent approaches this aspect of the work]

### 2. [Framework Component Name]
[Detailed explanation of methodology and best practices]

## Implementation Patterns

### [Pattern Name]
```python
# Code example showing typical implementation
def pattern_example():
    # Implementation details
    pass
```

### [Another Pattern]
```python
# Additional implementation approach
def another_pattern():
    # Implementation details
    pass
```

## Quality Standards

- [Quality standard 1]
- [Quality standard 2]
- [Quality standard 3]
- [Quality standard 4]

## Deliverables

### Standard Output Format
[List of typical deliverables with descriptions]

### Documentation Requirements
[What documentation the agent provides]

Remember: [Key principle or reminder about the agent's approach]
```

## Common Pitfalls to Avoid

1. **Too General**: Agents that try to do everything often aren't specific enough to be useful
2. **No Clear Differentiation**: Make sure your agent has a clear purpose that isn't already covered
3. **Poor Examples**: Examples should clearly show when and why to use your agent
4. **Missing Methodology**: Include clear frameworks and approaches
5. **No Quality Standards**: Define what makes good work in your agent's domain
6. **Poor Integration**: Consider how your agent works with others in the ecosystem

By following these guidelines, you'll create high-quality, specialized agents that enhance the data science capabilities of the entire ecosystem.