# CODE-A-NOVA Month 3 — Customer Segmentation

## Project overview
This project uses K-Means clustering to group mall customers by annual income and spending score, then visualizes the resulting customer segments.

## Internship task
This is the **Level 1(b): Customer Segmentation** option from the CODE-A-NOVA Month 3 Data Science task brief. The task asks for data preprocessing, K-Means clustering, visualization of customer segments, and interpretation for marketing strategies.

## Dataset
Download `Mall_Customers.csv` from:
https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python

Place the CSV at `data/Mall_Customers.csv`. The dataset is not included in this ZIP; check its license and attribution terms before redistribution.

Expected columns:
- `CustomerID`
- `Gender`
- `Age`
- `Annual Income (k$)`
- `Spending Score (1-100)`

## Requirements
Python 3.9 or later. Install packages with:

```bash
python -m pip install -r requirements.txt
```

## Run on Windows
1. Extract the ZIP.
2. Download the CSV and put it in the `data` folder.
3. Open PowerShell in the project folder.
4. Run:

```powershell
python -m pip install -r requirements.txt
python customer_segmentation.py
```

## Project structure
```text
CODE-A-NOVA-Customer-Segmentation/
├── customer_segmentation.py
├── README.md
├── requirements.txt
├── .gitignore
├── GITHUB_UPLOAD.txt
├── data/
│   └── README.txt
└── outputs/
    └── README.txt
```

## Methodology
1. Inspect the dataset and check missing values and duplicates.
2. Explore customer age, income, and spending.
3. Standardize annual income and spending score.
4. Use the Elbow Method to compare cluster counts.
5. Apply K-Means and visualize the assigned clusters.
6. Summarize customer profiles and interpret the results.

## Outputs
The script creates these files inside `outputs/`:
- `age_distribution.png`
- `income_spending.png`
- `elbow_method.png`
- `customer_segments.png`
- `segmented_customers.csv`
- `segment_profiles.csv`

The script starts with five clusters as a candidate. Review the Elbow Method plot and justify or adjust K based on your analysis. Cluster IDs are arbitrary labels.

## Interpretation
Use the generated profile table and charts to describe the actual customer groups. Possible business actions include loyalty rewards for high-spending customers or targeted offers for customers with lower spending, but conclusions should be supported by your results.

## Limitations
The dataset is relatively small and uses a mall-assigned spending score rather than detailed purchase histories. Results should not automatically be generalized to all customers.

## Dataset attribution
Dataset source: https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python
