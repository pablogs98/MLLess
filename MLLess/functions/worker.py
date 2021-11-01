import time

import numpy as np

from MLLess.utils.communication import RabbitMQCommunicator, BackendRabbitMQCommunicator
from MLLess.utils.storage import StorageIterator


def worker(ibm_cos, args):
    """
    Generic worker cloud function.
    :param ibm_cos: Lithops Storage instance. Provides the function access to the selected Cloud Storage instance
                    in the configuration

    :param args:
    """
    t0 = time.time()
    np.seterr(all='raise')

    # basic parameters
    bucket = args['bucket']
    dataset = args['dataset']
    worker_id = args['worker_id']
    n_minibatches = args['n_minibatches']
    n_workers = args['n_workers']
    seed = args['seed']
    local = args['local']
    executor_id = args['executor_id']
    dataset_path = args['dataset_path']
    rabbitmq_params = args['rabbitmq_params']
    backend = args['backend']
    slack = args['slack']
    n_redis = args['n_redis']
    redis_hosts = args['redis_hosts']

    iter_per_epoch = int(n_minibatches / n_workers)
    if n_minibatches % n_workers != 0:
        iter_per_epoch += 1

    # RabbitMQ & storage iterator
    if backend is not None:
        storage = backend(storage=ibm_cos, bucket=bucket, n_redis=n_redis, redis_hosts=redis_hosts)
        communicator = BackendRabbitMQCommunicator(executor_id, worker_id, slack, storage, rabbitmq_params)
    else:
        communicator = RabbitMQCommunicator(executor_id, worker_id, slack, rabbitmq_params)

    iterator = StorageIterator(ibm_cos, bucket, dataset, worker_id, n_minibatches, seed, n_workers, dataset_path, local)
    print(f'Running worker with id={worker_id}, execution={executor_id}')

    get_model = args.pop('model', None)
    args['communicator'] = communicator
    model = get_model(**args)

    step = 0
    epoch = 0

    local_times = []

    for minibatch in iterator:
        step_total_time = time.time()

        # Step
        ti0 = time.time()
        loss = model.step(epoch, step, minibatch)
        t_process = time.time() - ti0

        ts0 = time.time()
        if n_workers > 1:
            significant_updates = model.get_significant_updates(step)
        else:
            significant_updates = None
        t_generate_updates = time.time() - ts0

        t_fetch_minibatch = iterator.cos_time

        # Communicate significant updates
        t_up_0 = time.time()
        update_available = significant_updates is not None

        if update_available:
            communicator.send_updates(step, significant_updates)

        t_write_update = time.time() - t_up_0

        # Listen to updates or supervisor msgs
        t_r_u0 = time.time()
        communicator.send_step_end(step, loss, update_available)
        iterator.n_workers = communicator.listen(n_workers, step, update_available, model)
        communicator.aggregate_updates(storage, model)

        t_read_updates = time.time() - t_r_u0

        t_p_u0 = time.time()

        # Send model if killed
        if communicator.killed:
            communicator.send_model_on_death(model.get_weights())
        elif len(communicator.received_models) != 0:
            for m in communicator.received_models:
                model.aggregate_model(m)
            communicator.received_models = []
            iter_per_epoch = int(n_minibatches / n_workers)
            if n_minibatches % n_workers != 0:
                iter_per_epoch += 1
        t_process_updates = time.time() - t_p_u0

        # Generate time stats
        step_total_time = time.time() - step_total_time
        t_read_redis_updates = communicator.fetch_update_time
        communicator.fetch_update_time = 0.0
        t_read_rabbit_updates = t_read_updates - t_read_redis_updates

        local_times.append(
            (t_fetch_minibatch, t_process, t_generate_updates,
             t_write_update, t_read_updates, t_read_redis_updates, t_read_rabbit_updates,
             t_process_updates, step_total_time))

        if step % iter_per_epoch == 0 and step != 0:
            epoch += 1

        # Finish execution if end is reached (killed, training end, etc.)
        if communicator.end:
            communicator.send_model_on_finish((model.get_weights(), local_times))
            break
        step += 1

    total_time = time.time() - t0
    print(f'Finished execution of worker with id={worker_id}. Iterations = {step}. Total time: {total_time}.\n')
    communicator.send_times(total_time)
