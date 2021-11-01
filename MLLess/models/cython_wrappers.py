def get_lr_model(**kwargs):
    from lithops.libs.lr_model import LogisticRegressionModel
    return LogisticRegressionModel(**kwargs)


def get_sparse_lr_model(**kwargs):
    from lithops.libs.sparse_lr_model import SparseLogisticRegressionModel
    return SparseLogisticRegressionModel(**kwargs)


def get_mf_model(**kwargs):
    from lithops.libs.mf_model import MatrixFactorisationModel
    return MatrixFactorisationModel(**kwargs)
