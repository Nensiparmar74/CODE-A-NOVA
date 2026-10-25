from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parents[1]

nb1=nbf.v4.new_notebook()
nb1.cells=[
 nbf.v4.new_markdown_cell('# CropYield Predictor - Data Validation\n\nValidation of schema, missing values, ranges, categories, duplicates, and leakage checks on the raw agricultural dataset.'),
 nbf.v4.new_code_cell("""import pandas as pd\nfrom src.data_validation import validate_full_dataset, check_duplicates\ndf = pd.read_csv('data/raw/yield_df.csv')\nreport = validate_full_dataset(df)\nreport"""),
 nbf.v4.new_code_cell("""print('Raw dataset shape:', df.shape)\nprint('Duplicate rows:', check_duplicates(df))\nprint('Missing values:', int(df.isna().sum().sum()))\nprint('Validation passed:', report['is_valid'])""")
]

nb2=nbf.v4.new_notebook()
nb2.cells=[
 nbf.v4.new_markdown_cell('# CropYield Predictor - Exploratory Data Analysis\n\nThis notebook contains 21 EDA visualizations. Every chart has a title, axis labels, and a written interpretation.'),
 nbf.v4.new_code_cell("""import json\nfrom pathlib import Path\nfrom IPython.display import display, Image\nimport scripts_generate_eda\nfigures = json.load(open('models/eda_figures.json'))\nprint(f'Visualizations generated: {len(figures)}')"""),
 nbf.v4.new_code_cell("""for fig in figures:\n    print(f\"Figure {fig['id']}: {fig['title']}\")\n    display(Image(filename=str(Path(fig['file'])), width=720))\n    print('Interpretation:', fig['interpretation'])\n    print('-' * 80)""")
]

nb3=nbf.v4.new_notebook()
nb3.cells=[
 nbf.v4.new_markdown_cell('# CropYield Predictor - Model Training and Evaluation\n\nSummary of the production model, model-comparison artifacts, held-out test metrics, tuning evidence, and serialized inference pipeline.'),
 nbf.v4.new_code_cell("""import json, joblib, pandas as pd\nmetrics = json.load(open('models/evaluation_metrics.json'))\ncomparison = pd.DataFrame(json.load(open('models/model_comparison.json')))\nprint('Model comparison:')\ndisplay(comparison)\nprint('Final evaluation:')\nprint(json.dumps(metrics, indent=2))"""),
 nbf.v4.new_code_cell("""pipeline = joblib.load('models/crop_yield_model.joblib')\nsample = {'Area':'India','Item':'Rice, paddy','Year':2008,'average_rain_fall_mm_per_year':1200.0,'pesticides_tonnes':40000.0,'avg_temp':26.5}\nprint('Serialized pipeline type:', type(pipeline).__name__)\nprint('Sample inference:', pipeline.predict(sample))""")
]

for name,nb in [('01_data_validation.ipynb',nb1),('02_eda.ipynb',nb2),('03_modeling.ipynb',nb3)]:
    path=ROOT/'notebooks'/name
    nbf.write(nb,path)
    client=NotebookClient(nb,timeout=300,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}})
    client.execute()
    nbf.write(nb,path)
    print('executed',path)
