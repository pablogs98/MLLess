from benchmarks.pywren.matrix_factorisation.model import MatrixFactorisation


def reducer(results):
    minibatch_size_sum = 0
    loss_sum = 0.0

    n_workers = len(results)
    result = None

    for model in results:
        loss, minibatch_size = model.loss
        loss_sum += loss
        minibatch_size_sum += minibatch_size

        if isinstance(model, MatrixFactorisation):
            if result is None:
                result = {'L': model.L, 'R': model.R}
            else:
                result['L'] += model.L
                result['R'] += model.R
        else:
            if result is None:
                result = model.weights
            else:
                result += model.weights

    for model in results:
        if isinstance(model, MatrixFactorisation):
            model.L = result['L'] / n_workers
            model.R = result['R'] / n_workers
        else:
            model.weights = result / n_workers


    return {'models': results, 'loss': loss_sum / minibatch_size_sum}
