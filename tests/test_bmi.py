import importlib.util
import math
import pathlib


def load_module(path):
    spec = importlib.util.spec_from_file_location("bmi_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_bmi_basic():
    mod = load_module(pathlib.Path("generated") / "bmi.py")
    # 70kg, 175cm -> 22.86
    assert math.isclose(mod.bmi(70, 175), 22.86, rel_tol=0, abs_tol=1e-9)


def test_bmi_rounding():
    mod = load_module(pathlib.Path("generated") / "bmi.py")
    val = mod.bmi(81.2, 173.4)
    # rounded to 2 decimals
    assert isinstance(val, float)
    assert round(val, 2) == val
