# SalaryLens — MIE1624 Bonus Interactive Dashboard

A fully static and interactive website for the **MIE1624 Assignment 2 Bonus (2 marks)**. Designed for **GitHub Pages**: no backend, hosting subscription, Python, model training or uploaded survey CSV is required to serve the site.

## Features

- Ask only the **13 original survey questions** used in the final model, with original multiple-choice option strings.
- Show the complete predicted probability distribution over **15 ordered annual salary buckets (USD)**.
- Display ranked technologies using the same Q5.5 **change in expected bucket index** metric.
- Distinguish **"None used"** from **"Not answered"** for multi-select questions.
- Perform inference entirely **inside the visitor's browser**; the site's assets contain trained coefficients and training-fitted preprocessing parameters, but **no individual salary data or survey rows**.

## Publish on GitHub Pages (no command line required)

1. Sign in to [GitHub](https://github.com/) and select **New repository**.
2. Choose a repository name, e.g. `salarylens-dashboard`, and set it to **Public** (GitHub Free Pages hosting).
3. Open the repository and select **Add file → Upload files**.
4. **Unzip the download first. Upload the files inside it to the repository root** (do not upload the ZIP alone):
   - `index.html`
   - `style.css`
   - `app.js`
   - `model.json`
   - `.nojekyll` (hidden file; recommended)
   - `README.md` (optional for the site itself)
5. Select **Commit changes**.
6. Go to **Settings → Pages**.
7. Under **Build and deployment**, set **Source → Deploy from a branch**; select branch `main` and folder `/(root)`, then **Save**.
8. The published site will be available at `https://YOUR_USERNAME.github.io/salarylens-dashboard/` (replace the account and repository names).
9. Use **Settings → Pages → Visit site** to open the actual deployed link.

> GitHub Pages hosts static HTML/CSS/JavaScript only. This project performs calculations in JavaScript using a model trained beforehand in Python, so it works without a Python server.

## Preview locally

From the folder containing `index.html`, run:

```bash
python -m http.server 8000
```

Then open `http://localhost:8000/`. Double-clicking `index.html` may not work because browsers can block local `fetch('./model.json')` requests.

## File descriptions

| File | Purpose |
|---|---|
| `index.html` | Site structure and inputs/results sections |
| `style.css` | Responsive layout and styling |
| `app.js` | Original survey form, preprocessing, cumulative-threshold logistic inference, charts and recommendation ranking |
| `model.json` | Model feature order, 14 fitted binary logistic regressions, training medians/scaling, valid survey options and candidate technologies |
| `.nojekyll` | Avoid GitHub Pages/Jekyll interfering with static assets |

All deployed files must be in the **same directory**. The original CSV and notebook are **not** required on the website and should **not** be uploaded as part of this deployment.

## Model consistency and verification

Source: `MIE1624_Assignment2_With_Bonus.ipynb` (Q1–Q5), using only the provided `processed_data_assignment2_2026.csv` data for model fitting.

- Stratified 80/20 split (`random_state=42`)
- Training-only preprocessing and feature selection (MI and Extra Trees)
- Final model: **cumulative-threshold ordinal logistic**, 14 `sklearn.linear_model.LogisticRegression` binary classifiers
- Final tuning: **L1 regularization, C = 1**, fitted on the complete training partition
- Browser inference exactly reproduces notebook logic: sigmoid of the 14 fitted models; cumulative maximum of thresholds; adjacent differences for valid 15-bucket probabilities
- Recommendations consider one unreported selected technology at a time and sort by `E[bucket | new technology] - E[bucket | existing answers]`, skipping non-answered technology questions

Verification: three held-out test respondent profiles were compared between Python and JavaScript. Largest absolute probability difference across 15 buckets was below `3e-15` using the same exported model coefficients. Browser UI was also tested for completion, probability chart generation and ranked technology output.

### Important limitations

This is an educational demonstration. Survey respondents with reported salaries do not represent every developer; salary categories have unequal dollar widths; associations are not causal. Predictions are **not personal financial or career advice**, and modeled skill additions are not evidence of a real pay increase.

## Suggested short report paragraph (Bonus)

> An interactive dashboard was implemented as a static website and deployed through GitHub Pages. The interface asks only the 13 original survey questions required by the final model, including single-select, numeric, and multiple-select responses. Browser-side JavaScript recreates the training-fitted transformations and evaluates the 14 fitted cumulative logistic classifiers without retraining. It displays the probability distribution across all 15 ordered salary buckets and ranks candidate technologies using the same expected-bucket improvement measure defined in Q5.5. Missing responses are distinguished from explicitly reporting no tools. The dashboard is an explanatory tool rather than a causal salary estimator. [Insert your deployed GitHub Pages URL here.]

## Maintaining the model

If you later retrain the notebook or change the selected features, the website must be updated with a newly exported `model.json` containing the matching selected feature order, coefficients, intercepts, numeric scaling and survey choices. Do not silently combine new model parameters with old preprocessing.
