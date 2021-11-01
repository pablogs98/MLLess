from MLLess.utils.storage_backends import CosBackend


def worker(ibm_cos, bucket, model, epoch, step, key):
    storage = CosBackend(ibm_cos, bucket)

    # get model and minibatch from COS
    minibatch = storage.get(key)
    model_instance = storage.get(model)

    # update model
    model_instance.step(epoch, step, minibatch)

    return model_instance
