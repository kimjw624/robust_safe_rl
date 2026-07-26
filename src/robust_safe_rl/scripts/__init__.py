"""Runnable entry points.

Each module exposes a ``main()`` and is invoked with ``python -m``:

- ``run_tracking_demo``     : closed-loop geometric tracking sanity check + plots.
- ``train_autoencoder``     : train one force-disturbance OOD detector.
- ``test_autoencoder``      : re-test a checkpoint on fresh ID/OOD data.
- ``evaluate_ood_detection``: confusion-matrix report for force OOD.
- ``evaluate_mass_moi_ood`` : cross-disturbance (mass/MOI) OOD evaluation.
- ``sweep_autoencoder``     : hyperparameter + regularization grid sweep.
"""
