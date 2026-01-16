# Claude Code Instructions

**Important**: Read [docs/AI-CODING.md](docs/AI-CODING.md) for complete project guidelines.

---

## Claude Code-specific Settings

### Planning Mode
- Use plan mode (`/plan`) for complex architectural decisions
- Exit plan mode with clear implementation steps

### File References
- Reference files with tab-completion for accuracy
- Use markdown links: `[filename](path/to/file)` for clickable references

### Virtual Environment
Ensure the `venv` virtual environment is activated:
```bash
source venv/bin/activate
```

---

## Quick Reference

### Common Commands
```bash
# Activate environment
source venv/bin/activate

# Start Jupyter Lab
jupyter lab
```

### Key Documentation
- **Project Guidelines**: [docs/AI-CODING.md](docs/AI-CODING.md)
- **Mind Monitor CSV Spec**: [docs/MIND_MONITOR_CSV_SPECIFICATION.md](docs/MIND_MONITOR_CSV_SPECIFICATION.md)

---

## Report Table Formatting Standards

### Standard Table Format
All report tables use a standardized 3-column format:
- **Metric**: Indicator name (English)
- **Value**: Numerical value
- **Unit**: Unit of measurement (English)

### Implementation

**1. Data Formatting (Python side)**
```python
from lib.templates.formatters import format_respiratory_stats

# Convert data object to DataFrame
respiratory_stats = format_respiratory_stats(respiration_result)

# Pass to template
context = {'ecg': {'respiratory_stats': respiratory_stats}}
```

**2. Template Rendering (Jinja2 side)**
```jinja2
{{ ecg.respiratory_stats|df_to_markdown(floatfmt='.1f', index=False, standardize_columns=True) }}
```

### Adding New Tables

1. Create formatter function in `lib/templates/formatters.py`:
   ```python
   def format_your_stats(data_object) -> pd.DataFrame:
       """Convert data to Metric/Value/Unit DataFrame"""
       stats = []
       stats.append({'Metric': 'Your Metric', 'Value': data_object.value, 'Unit': 'unit'})
       return pd.DataFrame(stats)
   ```

2. Use in script:
   ```python
   from lib.templates.formatters import format_your_stats
   context['your_stats'] = format_your_stats(your_data)
   ```

3. Render in template:
   ```jinja2
   {{ your_stats|df_to_markdown(index=False, standardize_columns=True) }}
   ```

### Language Standard
- **All table content must be in English** (Metric names, Units)
- Japanese is allowed only in section headers and explanatory text
