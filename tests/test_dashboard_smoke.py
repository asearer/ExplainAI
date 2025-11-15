def test_dashboard_import():
    import importlib
    module = importlib.import_module("dashboard.app")
    assert module is not None
