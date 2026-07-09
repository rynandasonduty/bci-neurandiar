"""
backend/src/experiments_p4_p7

Supplementary single-variable experiments P4-P7, each testing one variable
against the P3 champion configuration (E5_Data_Augmentation / S3 / Barlow).

Fully isolated from backend/src/models and backend/src/preprocessing: no file
owned by P1/P2/P3 is modified here. New behavior comes from subclassing
(signal_processors_ext.py, dataset_builders_ext.py) or from calling existing,
unmodified functions with different parameters. P4, P5, P6, and P7 do not
depend on one another's code or outputs.
"""
