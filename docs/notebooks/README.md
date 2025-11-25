# Story Notebooks - Viewing Instructions

## VS Code Setup

1. **Install Extensions** (you mentioned you already have these):
   - Jupyter extension (by Microsoft)
   - Python extension (by Microsoft)

2. **Open a Notebook**:
   - Open any `.ipynb` file in VS Code
   - VS Code will automatically detect it as a Jupyter notebook

3. **Select Kernel**:
   - Click the kernel selector in the top-right of the notebook
   - Choose your Python environment (should have `plotly`, `pandas`, `numpy`, `torch`, `ipywidgets` installed)

4. **Run Cells**:
   - Click "Run All" in the toolbar, or
   - Run cells individually with `Shift+Enter`

## Viewing Plotly Graphs

The notebooks use **Plotly** for interactive visualizations. In VS Code:

- **Interactive plots** should appear inline below each cell that calls `.show()`
- If graphs don't appear, try:
  1. Make sure you've run the import cell first (Cell 2 in each notebook)
  2. Check that Plotly is installed: `pip install plotly`
  3. Restart VS Code if graphs still don't render

## Troubleshooting

**If graphs don't show:**
```python
# Add this to a cell and run it to check your renderer:
import plotly.io as pio
print("Available renderers:", pio.renderers.list())
print("Current default:", pio.renderers.default)
```

**To force a specific renderer**, add this after imports:
```python
import plotly.io as pio
pio.renderers.default = "notebook"  # or "vscode" if available
```

**If you see errors about missing modules:**
```bash
pip install plotly pandas numpy torch ipywidgets
```

## Notebook Structure

- **`master_story.ipynb`**: High-level overview, geometry, comparisons
- **`pinn_story.ipynb`**: Deep dive into PINN training and validation
- **`mmc_story.ipynb`**: MMC optimization walkthrough

All notebooks share utilities from `story_helpers.py` in the same directory.

