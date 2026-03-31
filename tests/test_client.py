from __future__ import annotations

import re
from uuid import uuid4

import numpy as np
import pytest

from vectora import VectoraClient
from vectora.compliance.trace import generate_trace_id
from vectora.exceptions import VectoraConfigError
from vectora.models import SklearnConnector


class FakeModel:
    def predict(self, X):
        array = np.asarray(X)
        return np.where(array[:, 0] > 0.5, 1, 0)


def test_client_requires_vectora_key_prefix():
    with pytest.raises(VectoraConfigError):
        VectoraClient(api_key="invalid_key")


def test_generate_trace_id_matches_expected_format():
    trace_id = generate_trace_id()
    assert re.fullmatch(r"vct_\d{8}_[a-z0-9]{4}", trace_id)


def test_sklearn_connector_requires_uuid_model_id():
    with pytest.raises(VectoraConfigError):
        SklearnConnector(client=VectoraClient(api_key="vct_live_12345678901234567890"), model_id="fraud-detector-v1")


def test_predict_returns_underlying_model_predictions(monkeypatch):
    client = VectoraClient(api_key="vct_live_12345678901234567890")
    connector = SklearnConnector(client=client, model=FakeModel(), model_id=str(uuid4()))

    monkeypatch.setattr(SklearnConnector, "_send_payload", lambda self, payload: None)

    X = np.array([[0.1, 0.2], [0.8, 0.4], [0.7, 0.9]])
    expected = FakeModel().predict(X)

    predictions = connector.predict(X)

    assert np.array_equal(predictions, expected)
