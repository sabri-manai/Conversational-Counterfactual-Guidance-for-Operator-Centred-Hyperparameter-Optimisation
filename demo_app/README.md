# CF-XHPO conversational-interface demo (Streamlit)

Source for the deployed Section 3.6 conversational interface. A live instance is hosted
separately at https://huggingface.co/spaces/smana/cf-xai-hpo.

## Run locally

```bash
python -m venv venv && source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
streamlit run app.py
```

The app expects, relative to its working directory:

- `surrogate_ready_dataset/patchcore_surrogate_dataset_xgb.csv` (produced by
  `../proditec/02_preprocessing.ipynb`)
- `surrogate_pkl_cfs_metadata/surrogate_catboost.cbm` (a CatBoost model at this **unprefixed**
  path — copy or symlink `../proditec/surrogate_pkl_cfs_metadata/proditec_surrogate_catboost.cbm`
  here under the name `surrogate_catboost.cbm` if running against the Proditec pipeline's
  output; the demo intentionally uses generic, dataset-neutral filenames since it is deployed as
  a standalone artifact separate from the analysis notebooks).

Loads `Qwen/Qwen2.5-3B-Instruct` by default (selectable in the sidebar; `Phi-3-mini` and
`Llama-3.2-3B` are also listed, matching the paper's Section 3.6 alternatives). A GPU is
strongly recommended; the app falls back to CPU (float32) automatically if none is available.
