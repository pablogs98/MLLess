import csv
import time

import lithops
import lithops as pywren

from MLLess.utils.storage_backends import CosBackend
from benchmarks.pywren.key_generator import KeyGenerator
from benchmarks.pywren.reducer import reducer
from benchmarks.pywren.worker import worker


def run_pywren(models, bucket, dataset, n_minibatches, epochs, results_file, local=False):
    t0 = time.time()
    print(f"\nStarting execution: \n\t{len(models)} workers. \n\tDataset: {dataset}. \n\t{epochs} epochs.")
    with open(results_file, 'w') as file:
        results_writer = csv.writer(file)
        results_writer.writerow(['step', 'loss', 'time'])
        n_workers = len(models)
        steps_per_epoch = int(n_minibatches / n_workers)
        if n_minibatches % n_workers != 0:
            steps_per_epoch += 1
        key_generator = KeyGenerator(dataset, n_workers, n_minibatches)
        storage = CosBackend(lithops.Storage().get_client(), bucket)

        if local:
            pw = pywren.LocalhostExecutor()
        else:
            pw = pywren.FunctionExecutor(runtime_memory=2048)
        try:
            for epoch in range(epochs):
                for step in range(steps_per_epoch):
                    minibatch_keys = key_generator.get_keys(n_workers)
                    payload = []
                    for k, key in enumerate(minibatch_keys):
                        model_key = f'w_{models[k].worker_id}_model'
                        storage.put(model_key, models[k])
                        args = {'bucket': bucket, 'model': model_key, 'epoch': epoch, 'step': step, 'key': key}
                        payload.append(args)
                    f = pw.map_reduce(worker, payload, reducer)
                    result = pw.get_result(f)
                    models = result['models']
                    loss = result['loss']
                    print(f"Step {step} done. Loss={loss}")
                    results_writer.writerow([step, loss, time.time() - t0])
        except KeyboardInterrupt:
            t1 = time.time()
            print(f"\nFinished execution. Elapsed time: {t1 - t0} s\n. Cost ($): {0.000017 * 2 * n_workers * (t1 - t0)}")
    t1 = time.time()
    print(f"\nFinished execution. Elapsed time: {t1 - t0} s\n. Cost ($): {0.000017 * 2 * n_workers * (t1 - t0)}")
