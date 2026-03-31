# Vectora Python SDK

Vectora helps you monitor sklearn models in production without adding latency to your prediction path.

The SDK captures:
- prediction metrics like accuracy, F1, precision, and recall
- feature distribution summaries for drift detection
- SHAP-based feature importance when `shap` is available
- trace IDs that connect predictions back to the Vectora dashboard

## Install

```bash
pip install vectora
```

## Quickstart

```python
from vectora import VectoraClient
from vectora.models import SklearnConnector

client = VectoraClient(api_key="vct_live_xxx")
connector = SklearnConnector(
    client=client,
    model=your_sklearn_model,
    model_id="11111111-1111-1111-1111-111111111111",
)

predictions = connector.predict(X_test, y_true=y_test)
```

`predict()` returns the model's predictions immediately. Vectora sends the monitoring payload in a background thread so your production inference path stays fast.

## What gets sent

Each prediction call sends a payload to `/api/ingest/metrics` with:
- `trace_id`
- `model_id`
- `metrics`
- `feature_distributions`
- `shap_values`
- `sample_count`
- `timestamp`

If SHAP is not installed, the SDK logs a warning once and continues without SHAP values.

If the network call fails, the SDK logs the error to stderr and never raises it back to your prediction path.

## API

### `VectoraClient`

```python
client = VectoraClient(
    api_key="vct_live_xxx",
    base_url="https://vectora.ai",
    timeout=5.0,
    max_retries=2,
)
```

### `SklearnConnector`

```python
connector = SklearnConnector(client, model_id="11111111-1111-1111-1111-111111111111", model=trained_model)
predictions = connector.predict(X, y_true=y_true)
```

## Coming Soon

`vectora.llm` and `vectora.agent` are reserved for future releases and raise `ComingSoonError` when accessed.

## Release

Tagging the repository with `v*` triggers the GitHub Actions publish workflow in [`.github/workflows/publish.yml`](.github/workflows/publish.yml).

```bash
git tag v0.1.0
git push origin v0.1.0
```

The workflow:
- runs the SDK test suite
- builds the source and wheel distributions
- publishes to PyPI using GitHub Actions trusted publishing
