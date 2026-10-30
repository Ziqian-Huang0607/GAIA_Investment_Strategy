from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gaia_ai.backtest import expanding_window_backtest
from gaia_ai.config import load_config
from gaia_ai.data import load_panel, prepare_panel
from gaia_ai.model import latest_predictions


@pytest.fixture(scope="session")
def project_root():
    return ROOT


@pytest.fixture(scope="session")
def config():
    return load_config(ROOT / "config" / "default.json")


@pytest.fixture(scope="session")
def prepared(config):
    panel = load_panel(ROOT / "data" / "gaia_quarterly_panel.csv")
    return prepare_panel(panel, "2026-10-30", config.model)


@pytest.fixture(scope="session")
def backtest_result(prepared, config):
    return expanding_window_backtest(prepared, config.model)


@pytest.fixture(scope="session")
def latest_result(prepared, config):
    return latest_predictions(prepared, "2026-10-30", config.model)
