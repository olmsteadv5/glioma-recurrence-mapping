# explainer: one module per outside dataset, picked by the patient number's leading digit (see paths.COLLECTIONS)

from datasets import predict_gbm, rhuh

ADAPTERS = {5: predict_gbm, 6: predict_gbm, 7: predict_gbm, 8: rhuh, 9: rhuh}


def adapter(number):
    return ADAPTERS[number // 10000]
