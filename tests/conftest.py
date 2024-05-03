"""Generic tests environment configuration. This file implement all generic
fixtures to simplify model and api specific testing.

Modify this file only if you need to add new fixtures or modify the existing
related to the environment and generic tests.
"""

# pylint: disable=redefined-outer-name
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from imgclas import api


@pytest.fixture(scope="module")
def metadata():
    """Fixture to return get_metadata to assert properties."""
    return api.get_metadata()


@pytest.fixture(scope="module")
def predictions(predict_args, predict_kwds):
    """Fixture to return predictions to assert properties."""
    predictions = np.random.dirichlet(np.ones(10), size=[20])
    with patch("imgclas.api.model") as model:
        model.predict = MagicMock(return_value=predictions)
        return api.predict(*predict_args, **predict_kwds)

