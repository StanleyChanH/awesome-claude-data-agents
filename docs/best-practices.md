# Data Science Agents Best Practices

This guide outlines best practices for using the Data Science Agents team effectively and getting the most value from your analytical workflows.

## Orchestration Best Practices

### 1. Always Start with the Right Orchestrator

For complex analytical projects, always begin with the appropriate orchestrator:

```bash
# For multi-step analytical projects
claude "use @data-science-orchestrator to plan this analysis"

# For initial project setup
claude "use @data-team-configurator to analyze my data environment"

# For focused statistical analysis
claude "use @data-analyst to analyze this dataset"
```

### 2. Follow the Agent Routing Map Exactly

When the data-science-orchestrator provides a routing map, follow it precisely:

- ✅ **DO**: Use only the agents explicitly recommended
- ✅ **DO**: Follow the suggested execution order
- ✅ **DO**: Pass results between agents as recommended
- ❌ **DON'T**: Substitute agents based on your own judgment
- ❌ **DON'T**: Skip recommended steps
- ❌ **DON'T**: Reorder agents without good reason

### 3. Use Human-in-the-Loop Approval

Always review the orchestrator's plan before execution:

```bash
# After getting the routing plan
"Based on the data-science-orchestrator's analysis, I'll:
1. Use data-explorer to understand the dataset
2. Use data-cleaner to address quality issues
3. Use ml-engineer to build the model

Should I proceed with this plan, or would you like to modify anything?"
```

## Data Management Best Practices

### 1. Data Quality First

Always assess and address data quality early:

```bash
# Start with data quality assessment
claude "use @data-cleaner to assess data quality and recommend improvements"

# Then proceed with analysis
claude "use @data-explorer to analyze the cleaned dataset"
```

### 2. Reproducible Data Handling

Ensure your Python data workflow is reproducible:

- Set random seeds for all stochastic operations (numpy.random.seed(), random.seed())
- Document data sources and versions
- Use Python environment specifications (requirements.txt or environment.yml)
- Save intermediate results with joblib or pickle where appropriate
- Use Jupyter notebooks for exploratory analysis with clear cell execution order

### 3. Appropriate Data Splits

Always maintain proper train/validation/test splits:

```bash
# Let ml-engineer handle proper data splitting
claude "use @ml-engineer to ensure proper train/test splitting for this model"
```

## Analytical Workflow Best Practices

### 1. Understand Before Analyzing

Always start with exploration:

```bash
# For new datasets
claude "use @data-explorer to understand this dataset's characteristics"

# Then proceed with specific analysis
claude "use @statistical-analyst to analyze relationships in the data"
```

### 2. Validate Assumptions

Continuously validate analytical assumptions:

```bash
# Use model-validator for rigorous validation
claude "use @model-validator to validate the model's assumptions and performance"
```

### 3. Statistical Rigor

Ensure statistical validity:

```bash
# For statistical testing
claude "use @statistical-analyst to ensure proper statistical methodology"

# For model validation
claude "use @model-validator to perform comprehensive validation"
```

## Machine Learning Best Practices

### 1. Start Simple

Begin with simple models before complex ones:

```bash
# Let ml-engineer select appropriate model complexity
claude "use @ml-engineer to develop and compare models of appropriate complexity"
```

### 2. Feature Engineering Matters

Invest time in feature engineering:

```bash
# For better model performance
claude "use @feature-engineer to create predictive features"
```

### 3. Cross-Validation Essential

Always use proper cross-validation:

```bash
# Ensure robust model evaluation
claude "use @model-validator to implement proper cross-validation"
```

### 4. Consider Production Early

Design for deployment from the start:

```bash
# Include deployment considerations
claude "use @ml-engineer to design a production-ready ML pipeline"
```

## Visualization Best Practices

### 1. Choose the Right Visualization

Select appropriate chart types for your data:

```bash
# Let data-visualizer choose optimal visualizations
claude "use @data-visualizer to create appropriate visualizations for this analysis"
```

### 2. Focus on Clarity

Prioritize clear communication of insights:

- Use appropriate color schemes
- Ensure accessibility
- Label axes and provide context
- Highlight key insights

### 3. Interactive When Valuable

Use interactivity to enhance understanding:

```bash
# For complex datasets
claude "use @interactive-dashboard-creator to build an exploratory dashboard"
```

## Code Quality Best Practices

### 1. Review Early and Often

Incorporate code review throughout development:

```bash
# Review analytical code for quality
claude "use @data-science-code-reviewer to review this analytical code"
```

### 2. Document Thoroughly

Ensure your analysis is well-documented:

```bash
# Create comprehensive documentation
claude "use @analytics-documentation-specialist to document this analysis"
```

### 3. Follow Data Science Standards

Adhere to data science coding standards:

- Use clear variable names
- Include type hints where appropriate
- Add comments for complex logic
- Structure code logically

## Communication Best Practices

### 1. Know Your Audience

Tailor communication to the audience:

```bash
# For business stakeholders
claude "use @report-designer to create a business-friendly report"

# For technical teams
claude "use @analytics-documentation-specialist to create technical documentation"
```

### 2. Provide Context

Always provide sufficient context:

- Explain methodology choices
- State assumptions clearly
- Discuss limitations
- Provide recommendations

### 3. Visualize Key Insights

Use visualizations to enhance understanding:

```bash
# Create compelling visualizations
claude "use @data-visualizer to create visualizations that highlight key insights"
```

## Common Python Workflow Patterns

### 1. Exploratory Data Analysis (pandas-based)
```bash
# Standard EDA workflow with Python/pandas
1. data-explorer: Understand dataset (df.info(), df.describe(), df.isnull().sum())
2. data-cleaner: Address quality issues (df.dropna(), df.fillna(), outlier detection)
3. statistical-analyst: Analyze relationships (correlation analysis, hypothesis testing)
4. data-visualizer: Create visualizations (matplotlib, seaborn, plotly)
```

### 2. Predictive Modeling (scikit-learn pipeline)
```bash
# Complete ML workflow with Python/scikit-learn
1. data-science-orchestrator: Plan analysis
2. data-explorer: Initial exploration (pandas profiling)
3. data-cleaner: Data preparation (preprocessing, imputation)
4. feature-engineer: Feature creation (ColumnTransformer, FeatureUnion)
5. ml-engineer: Model development (Pipeline, GridSearchCV)
6. model-validator: Validation (cross_val_score, classification_report)
7. data-visualizer: Results visualization (feature importance, confusion matrix)
8. data-science-code-reviewer: Quality review (PEP8, type hints, docstrings)
```

### 3. Statistical Analysis
```bash
# Statistical investigation
1. data-analyst: Statistical analysis
2. statistical-analyst: Rigorous testing
3. data-visualizer: Statistical plots
4. report-designer: Findings report
```

### 4. Time Series Analysis
```bash
# Temporal data workflow
1. time-series-analyst: Temporal analysis
2. statistical-analyst: Statistical validation
3. data-visualizer: Time series plots
4. report-designer: Forecast report
```

## Pitfalls to Avoid

### 1. Skipping Data Exploration
❌ Don't jump directly to modeling without understanding the data

### 2. Ignoring Data Quality
❌ Don't proceed with analysis without addressing data quality issues

### 3. Overfitting
❌ Don't create overly complex models that don't generalize

### 4. Ignoring Statistical Assumptions
❌ Don't use statistical methods without validating assumptions

### 5. Poor Communication
❌ Don't present results without proper context and explanation

### 6. No Validation
❌ Don't present results without proper validation

### 7. Not Documenting
❌ Don't leave analysis without proper documentation

## Optimization Tips

### 1. Use Parallel Processing
When the orchestrator suggests parallel tasks, use them:

```bash
# If orchestrator suggests parallel execution
"I'll run data-explorer and data-cleaner in parallel to speed up the analysis"
```

### 2. Cache Intermediate Results
Save intermediate results for reuse:

```bash
# Cache expensive computations
"Please save the cleaned dataset for reuse in subsequent analysis steps"
```

### 3. Iterative Improvement
Use feedback loops to improve analysis:

```bash
# Refine based on results
"Based on the data-explorer findings, let's refine our approach and focus on [specific aspect]"
```

## Measuring Success

### 1. Analytical Quality
- Statistical validity
- Reproducibility
- Documentation completeness
- Code quality

### 2. Business Impact
- Actionable insights
- Clear recommendations
- Stakeholder understanding
- Implementation feasibility

### 3. Technical Excellence
- Performance efficiency
- Scalability considerations
- Maintainability
- Production readiness

By following these best practices, you'll get the most value from the Data Science Agents team and ensure high-quality, impactful analytical outcomes.