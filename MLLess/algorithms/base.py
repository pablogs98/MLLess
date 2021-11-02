import pickle
import time
import json
import csv

import pika
import lithops
import redis
from redis import StrictRedis

from MLLess.functions.supervisor import supervisor
from MLLess.functions.worker import worker
from MLLess.utils.storage_backends import RedisBackend, CosBackend


class Base:
    def __init__(self, n_workers, significance_threshold=0.0, end_threshold=0.0, remove_threshold=None,
                 remove_interval=None,
                 learning_rate=0.8, runtime_memory=2048, seed=8, slack=0, local=False, max_epochs=0, max_time=None,
                 min_workers=None, dataset_path=None, storage_backend='redis', n_redis=1):
        # Function executor
        if local:
            self.fexec = lithops.LocalhostExecutor(workers=n_workers + 1, log_level='debug')
        else:
            self.fexec = lithops.FunctionExecutor(runtime_memory=runtime_memory,
                                                  runtime='pablogs98/mlless-ibmcf-python-v38:21.1')

        # Basic data
        self.n_workers = n_workers
        self.executor_id = str(self.fexec.executor_id)
        self.learning_rate = learning_rate
        self.end_threshold = end_threshold
        self.asp_threshold = significance_threshold
        self.seed = seed  # minibatch pseudo-random selection
        self.min_workers = min_workers  # dynamically adjust number of workers
        self.max_epochs = max_epochs
        self.max_time = max_time
        self.n_minibatches = 0
        self.slack = slack
        self.remove_threshold = remove_threshold
        self.remove_interval = remove_interval

        # Local execution
        self.local = local  # execute in a local machine
        self.dataset_path = dataset_path

        # After-training data and statistics
        self.model = None
        self.n_epochs = 0
        self.loss_history = None
        self.iter_times_decomposition = None
        self.iter_time_history = []
        self.accuracy = 0.0
        self.super_time = None
        self.acc_functions_time = 0.0
        self.total_train_time = 0.0
        self.estimated_cost = 0.0

        # Storage backend
        self.n_redis = n_redis
        if storage_backend is not None:
            if storage_backend == 'redis':
                self.storage_backend = RedisBackend
            elif storage_backend == 'cos' and not local:
                self.storage_backend = CosBackend
            else:
                self.storage_backend = None
        else:
            self.storage_backend = None

        # RabbitMQ url, connection, channel
        lithops_config = self.fexec.config
        host = lithops_config['rabbit_mq']['rabbitmq_host']
        port = lithops_config['rabbit_mq']['rabbitmq_port']
        user = lithops_config['rabbit_mq']['rabbitmq_username']
        password = lithops_config['rabbit_mq']['rabbitmq_password']
        credentials = pika.PlainCredentials(user, password)
        self.rabbitmq_params = pika.ConnectionParameters(host=host, port=port, credentials=credentials,
                                                         blocked_connection_timeout=9999, heartbeat=600)
        self.connection = None
        self.channel = None

    def fit(self, trainset, n_minibatches):
        while True:
            try:
                self.connection = pika.BlockingConnection(self.rabbitmq_params)
                break
            except pika.exceptions.ConnectionClosed as exception:
                print(f"There was an error initialising RabbitMQ - {exception}. Trying again in 5 seconds...")
                time.sleep(5)

        self.channel = self.connection.channel()
        self.channel.exchange_declare(exchange=self.executor_id, exchange_type='fanout')
        self.channel.queue_declare('{}_supervisor'.format(self.executor_id))
        self.channel.queue_declare('{}_results'.format(self.executor_id))
        self.channel.queue_declare('{}_client'.format(self.executor_id))

        for w in range(self.n_workers):
            queue = '{}_w{}'.format(self.executor_id, w)
            self.channel.queue_declare(queue=queue)
            self.channel.queue_bind(queue=queue, exchange=self.executor_id)

        payload, sup_payload = self._initialise(trainset, n_minibatches)

        print(f'\nStarting Training - Running {self.n_workers} workers.')
        init_time = time.time()
        self.fexec.map(worker, payload)
        self.fexec.call_async(supervisor, data=sup_payload)
        self._iterate()
        self._clean()
        end_time = time.time()
        self.total_train_time = end_time - init_time
        self.functions_cost = (self.acc_functions_time + self.super_time) * 0.000017 * 2
        self.redis_cost = (0.17 / 3600) * self.total_train_time
        self.rabbitmq_cost = (0.15 / 3600) * self.total_train_time
        self.estimated_cost = self.redis_cost + self.rabbitmq_cost + self.functions_cost

        print("\nElapsed training time: %f s\n" % (end_time - init_time))

    def test(self, testset, verbose=False):
        pass

    def generate_stats(self, file_prefix, get_loss_iter_time_history=True, get_times_json=True,
                       get_iter_time_decomposition=True):
        if get_times_json:
            with open(f"{file_prefix}.json", 'w') as file:
                results_dict = {'total_train_time': self.total_train_time,
                                'acc_functions_time': self.acc_functions_time,
                                'super_time': self.super_time,
                                'functions_cost': self.functions_cost,
                                'redis_cost': self.redis_cost,
                                'rabbitmq_cost': self.rabbitmq_cost,
                                'total_estimated_cost': self.estimated_cost}
                json.dump(results_dict, file, indent=4)
            print(f"Generated results file: {file_prefix}.json.")

        if get_loss_iter_time_history:
            with open(f"{file_prefix}_loss-time-history.csv", 'w') as file:
                writer = csv.writer(file)
                header = ['step', 'loss', 'time']
                writer.writerow(header)
                data = [[step, loss] for step, loss in enumerate(self.loss_history)]
                acc_time = 0.0
                for t, time_ in enumerate(self.iter_time_history):
                    data[t].append(acc_time)
                    acc_time += time_
                writer.writerows(data)
            print(f"Generated results file: {file_prefix}.csv.")

        if get_iter_time_decomposition:
            with open(f"{file_prefix}_step-time-decomposition.csv", 'w') as file:
                writer = csv.writer(file)
                header = ['step', 't_fetch_minibatch', 't_process',
                          't_generate_updates', 't_write_update',
                          't_read_updates', 't_redis_read_updates',
                          't_rabbit_read_updates', 't_process_updates',
                          'step_total_time']
                writer.writerow(header)
                data = [[step, *times] for step, times in enumerate(self.iter_times_decomposition)]
                writer.writerows(data)

    def _initialise(self, trainset, n_minibatches):
        redis_hosts = self.fexec.config['redis_hosts']

        params = {'dataset': trainset, 'executor_id': self.executor_id, 'asp_threshold': self.asp_threshold,
                  'n_minibatches': n_minibatches,
                  'bucket': self.fexec.config['buckets']['datasets'], 'end_threshold': self.end_threshold,
                  'n_workers': self.n_workers, 'seed': self.seed,
                  'local': self.local, 'dataset_path': self.dataset_path,
                  'backend': self.storage_backend, 'slack': self.slack,
                  'rabbitmq_params': self.rabbitmq_params, 'n_redis': self.n_redis,
                  'redis_hosts': redis_hosts}

        sup_params = {'executor_id': self.executor_id, 'n_workers': self.n_workers,
                      'threshold': self.end_threshold, 'amqp_params': self.rabbitmq_params,
                      'n_minibatches': n_minibatches, 'epochs': self.max_epochs,
                      'backend': self.storage_backend, 'max_time': self.max_time,
                      'bucket': self.fexec.config['buckets']['datasets'],
                      'remove_threshold': self.remove_threshold, 'remove_interval': self.remove_interval,
                      'slack': self.slack, 'n_redis': self.n_redis, 'redis_hosts': redis_hosts}

        r = StrictRedis(host=redis_hosts[0])

        while True:
            try:
                r.flushall()
                break
            except redis.exceptions.ConnectionError as exception:
                print(f"There was an error initialising Redis - {exception}. Trying again in 5 seconds...")

        return params, sup_params

    def _iterate(self):
        times_received = 0
        result_received = False
        time_received = False
        time_diff = []

        def on_message(ch, _method, _properties, body):
            nonlocal times_received, result_received, time_received, time_diff
            m, w_id, msg = pickle.loads(body)

            if m == 'R':
                self.model, self.loss_history, self.iter_time_history, self.iter_times_decomposition, self.super_time, self.n_epochs = msg
                result_received = True
            elif m == 'T':
                self.acc_functions_time += msg
                times_received += 1
                if times_received == self.n_workers:
                    time_received = True
            elif m == 'E':
                print(f"Exception from w{w_id}, step={msg[1]}")
                raise msg[0]
            elif m == 'TD':
                # print(f"Time diff = {msg}")
                time_diff.append(msg)
            else:
                if w_id != -1:
                    print(f'Message from worker {w_id} -> {msg}')
                else:
                    print(msg)
            if result_received and time_received:
                ch.stop_consuming()

        self.channel.basic_consume(consumer_callback=on_message,
                                   queue="{}_client".format(self.executor_id),
                                   no_ack=False)
        self.channel.start_consuming()

    def _clean(self):
        self.fexec.clean()
        self.channel.exchange_delete(exchange=self.executor_id)
        self.channel.queue_delete(queue='{}_client'.format(self.executor_id))
        self.connection.close()
