import time

import pika
import pickle

from MLLess.utils.predictor.scheduler import ReclaimAction, WorkerScheduler


class CommonCommunicator:
    def __init__(self, executor_id, worker_id, slack, rabbitmq_params):
        super().__init__()

        self.connection = pika.BlockingConnection(rabbitmq_params)
        self.channel = self.connection.channel()
        self.executor_id = executor_id
        self.worker_id = worker_id
        self.slack = slack

        self.exchange = self.executor_id

    def __del__(self):
        self.connection.close()

    def send_debug(self, msg):
        self.channel.basic_publish(exchange='',
                                   routing_key='{}_client'.format(self.executor_id),
                                   body=pickle.dumps(('D', self.worker_id, msg)))

    def send_exception(self, exception, step):
        self.channel.basic_publish(exchange='',
                                   routing_key='{}_client'.format(self.executor_id),
                                   body=pickle.dumps(
                                       ("E", self.worker_id, (exception, step))))

    def consume(self, callback, queue):
        self.channel.basic_consume(consumer_callback=callback, queue=queue, no_ack=False)
        self.channel.start_consuming()


class SupervisorCommunicator(CommonCommunicator):
    def __init__(self, executor_id, slack, rabbitmq_params):
        super().__init__(executor_id, -1, slack, rabbitmq_params)

    class WorkerState:
        def __init__(self, i):
            self.w_id = i
            self.alive = True
            self.step = 0
            self.loss = 0.0
            self.blocked = False

        def kill(self):
            self.alive = False

        def progress(self, loss):
            self.step += 1
            self.loss = loss

        def block(self):
            self.blocked = True

        def unblock(self):
            self.blocked = False

    def send_finish(self):
        self.channel.basic_publish(exchange=self.exchange,
                                   routing_key='',
                                   body=pickle.dumps(("F", -1, None)))

    def send_continue(self, expected_updates, step):
        self.channel.basic_publish(exchange=self.exchange,
                                   routing_key='',
                                   body=pickle.dumps(("C", -1, (expected_updates, step))))

    def send_kill(self, worker_states, expected_updates, n_minibatches, n_workers, scheduler,
                  remove_interval, step, time_window, worker_id):
        killed = []
        if (time.time() - time_window) >= remove_interval:
            context = {'current_step': step}
            action = scheduler.schedule(context, verbose=True)

            if type(action) == ReclaimAction:
                scheduler.scale_down(verbose=True)
                worker_states[worker_id].kill()
                msg = f'[supervisor]: Scheduler removed worker with id={worker_id}'
                print(msg)
                self.send_debug(msg)
                self.channel.basic_publish(exchange='',
                                           routing_key='{}_w{}'.format(self.executor_id, worker_id),
                                           body=pickle.dumps(("K", -1, expected_updates)))
                killed.append(worker_id)
                n_workers -= 1
            time_window = time.time()

        steps_per_epoch = int(n_minibatches / n_workers)
        if n_minibatches % n_workers != 0:
            steps_per_epoch += 1

        self.send_continue(expected_updates, step)

        for k in killed:
            try:
                expected_updates.pop(k)
            except KeyError:
                pass

        return n_workers, steps_per_epoch, time_window

    def send_slowest_step(self, step):
        self.channel.basic_publish(exchange=self.exchange,
                                   routing_key='',
                                   body=pickle.dumps(("S", -1, step)))

    def send_results(self, results, step_loss_history, step_time_history, result_times, super_time, epoch):
        self.channel.basic_publish(exchange='',
                                   routing_key='{}_client'.format(self.executor_id),
                                   body=pickle.dumps(
                                       ('R', -1,
                                        (results, step_loss_history, step_time_history, result_times, super_time,
                                         epoch,))))

    # todo: delete
    def send_time_diff(self, msg):
        self.channel.basic_publish(exchange='',
                                   routing_key='{}_client'.format(self.executor_id),
                                   body=pickle.dumps(('TD', -1, msg)))

    def listen_results(self, n_workers, storage):
        received = 0
        result = None
        result_times = None

        # noinspection PyUnresolvedReferences
        def receive_results(ch, _method, _properties, body):
            nonlocal received, n_workers, result, storage, result_times

            received += 1

            if storage is not None:
                model = storage.get(body.decode())
            else:
                model = pickle.loads(body)

            times = model[1]
            model = model[0]

            if result_times is None:
                result_times = times
            else:
                if len(result_times) < len(times):
                    result_times = times
                # if len(result_times) < len(times):
                #     tmp_times = result_times
                #     result_times = times
                #     times = tmp_times
                # for t, time_ in enumerate(times):
                #     result_times[t] = tuple(map(add, result_times[t], time_))

            if len(model) == 1:
                if result is None:
                    result = model[0]
                else:
                    result = result + model[0]
            else:
                if result is None:
                    result = [model[0], model[1]]
                else:
                    result[0] = result[0] + model[0]
                    result[1] = result[1] + model[1]

            if received == n_workers:
                if len(model) == 1:
                    result = result / n_workers
                else:
                    result[0] = result[0] / n_workers
                    result[1] = result[1] / n_workers
                ch.stop_consuming()
            else:
                self.send_finish()

        self.consume(callback=receive_results, queue="{}_results".format(self.executor_id))
        return result, result_times

    def listen_workers(self, n_workers, step_loss_history, threshold, epochs, n_minibatches, remove_threshold,
                       remove_interval, init_time, max_time):

        # Worker scheduler
        scheduler = WorkerScheduler(n_workers, tol=remove_threshold)

        # control variables
        steps_per_epoch = int(n_minibatches / n_workers)
        if n_minibatches % n_workers != 0:
            steps_per_epoch += 1
        under_threshold = 0
        step = 0
        current_epoch_step = 0
        epoch = 0
        auto_scale = remove_threshold is not None and remove_interval is not None

        # stats
        worker_states = [self.WorkerState(i) for i in range(n_workers)]

        time_window = time.time()

        prev_time = 0
        slowest_step = 0

        def on_message(ch, _method, _properties, body):
            nonlocal step, epoch, epochs, under_threshold, n_workers, steps_per_epoch, current_epoch_step, \
                time_window, init_time, prev_time, slowest_step

            msg_type, w_id, content = pickle.loads(body)
            if msg_type == 'I':
                w_step, loss, update_available = content

                if worker_states[w_id].alive:
                    count = step_loss_history.update_loss_history(w_step, loss[0], loss[1])

                    if count == 1:
                        prev_time = time.time()
                    elif count == n_workers:
                        self.send_time_diff(time.time() - prev_time)

                    step_loss_history.set_update_available(w_id, w_step, update_available)

                    # self.send_debug(f"Received 'I' from {w_id} @ step {step}")

                if step_loss_history.check_step_end(step):
                    # self.send_debug(f"STEP {step} END!")
                    if step % 15 == 0:
                        self.send_debug(f"[supervisor] step {step} -> Loss = {step_loss_history.get_loss(step)}")

                    if max_time is None or (time.time() - init_time) < max_time:
                        if auto_scale:
                            scheduler.update(step, step_loss_history.get_loss(step), step_loss_history.get_time(step))

                        if step_loss_history.get_loss(step) <= threshold:
                            under_threshold += 1
                            if under_threshold == 10:
                                self.send_finish()
                                ch.stop_consuming()
                            elif auto_scale and scheduler.is_in_stationary_regime(verbose=True):
                                n_workers, steps_per_epoch, time_window = self.send_kill(worker_states,
                                                                                         step_loss_history.get_update_available(
                                                                                             None),
                                                                                         n_minibatches, n_workers,
                                                                                         scheduler,
                                                                                         remove_interval,
                                                                                         step, time_window, w_id)
                            else:
                                self.send_continue(step_loss_history.get_update_available(step), step)
                        else:
                            under_threshold = 0
                            if current_epoch_step % steps_per_epoch == 0 and step != 0:
                                current_epoch_step = 0
                                self.send_debug(f"[supervisor]: EPOCH {epoch} is over!")
                                epoch += 1
                                if epoch == epochs:
                                    self.send_finish()
                                    ch.stop_consuming()
                                elif auto_scale and scheduler.is_in_stationary_regime(verbose=True):
                                    n_workers, steps_per_epoch, time_window = self.send_kill(worker_states,
                                                                                             step_loss_history.get_update_available(
                                                                                                 step),
                                                                                             n_minibatches,
                                                                                             n_workers,
                                                                                             scheduler,
                                                                                             remove_interval,
                                                                                             step, time_window, w_id)
                                else:
                                    self.send_continue(step_loss_history.get_update_available(step), step)

                            elif auto_scale and scheduler.is_in_stationary_regime(verbose=True):
                                n_workers, steps_per_epoch, time_window = self.send_kill(worker_states,
                                                                                         step_loss_history.get_update_available(
                                                                                             step),
                                                                                         n_minibatches, n_workers,
                                                                                         scheduler,
                                                                                         remove_interval,
                                                                                         step, time_window, w_id)
                            else:
                                self.send_continue(step_loss_history.get_update_available(step), step)
                        step_loss_history.clear_update_available(step)
                        step += 1
                        current_epoch_step += 1
                    else:
                        self.send_finish()
                        ch.stop_consuming()
                    step_loss_history.n_workers = n_workers
            elif msg_type == 'P':
                w_step = content
                # self.send_debug(f"Received 'P' from {w_id} @ step {w_step}")
                step_loss_history.increment_processed_updates(w_step)

                if step_loss_history.get_processed_updates(slowest_step) == n_workers:
                    step_loss_history.zero_processed_updates(slowest_step)
                    slowest_step += 1
                    self.send_slowest_step(slowest_step)

        self.consume(callback=on_message, queue='{}_supervisor'.format(self.executor_id))
        return epoch


class RabbitMQCommunicator(CommonCommunicator):
    def __init__(self, executor_id, worker_id, slack, rabbitmq_params):
        super().__init__(executor_id, worker_id, slack, rabbitmq_params)
        self.fetch_update_time = 0.0
        self.queue = '{}_w{}'.format(self.executor_id, self.worker_id)

        self.total_t_au = 0
        self.total_t_barrier = 0

        self.updates_queue = []

        self.received = 0

        self.received_models = []

        self.killed = False
        self.end = False

        self.slowest_step = 0
        self.previous_slowest_step = 0

        if self.slack == 0:
            self.received_updates = {}
        else:
            self.received_updates = [{}] * (self.slack + 1)

        if self.slack == 0:
            self.expected_updates = None
        else:
            self.expected_updates = [None] * (self.slack + 1)

    def __del__(self):
        self.channel.queue_delete(queue=self.queue)
        super().__del__()

    def send_times(self, times):
        self.channel.basic_publish(exchange='',
                                   routing_key='{}_client'.format(self.executor_id),
                                   body=pickle.dumps(('T', self.worker_id, times)))

    def send_updates(self, step, updates):
        pi = pickle.dumps(("U", self.worker_id, (updates, step)))
        self.channel.basic_publish(exchange=self.exchange,
                                   routing_key='',
                                   body=pi)

    def send_step_end(self, step, loss, update_available):
        self.channel.basic_publish(exchange='',
                                   routing_key='{}_supervisor'.format(self.executor_id),
                                   body=pickle.dumps(('I', self.worker_id, (step, loss, update_available))))

    def send_updates_processed(self, step):
        self.channel.basic_publish(exchange='',
                                   routing_key='{}_supervisor'.format(self.executor_id),
                                   body=pickle.dumps(('P', self.worker_id, step)))

    def aggregate_updates(self, storage, model, _=False):
        for update_key in self.updates_queue:
            update = storage.get(update_key)
            model.aggregate_updates(update)
        self.updates_queue = []

    def listen(self, n_workers, step, update_available, _):
        t_au_0 = time.time()
        t_au_1 = t_au_0

        if update_available:
            if self.slack == 0:
                self.received_updates = {self.worker_id: update_available}
            else:
                self.received_updates[step % (self.slack + 1)] = {self.worker_id: update_available}

        if self.slack == 0:
            self.expected_updates = None

        def on_message(ch, _method, _properties, body):
            nonlocal n_workers
            nonlocal t_au_1
            nonlocal step

            msg_type, w_id, content = pickle.loads(body)

            if w_id != self.worker_id:
                if msg_type == 'C':
                    # 'C'ontinue message sent by the supervisor when the step is over
                    if self.slack == 0:
                        self.expected_updates = content[0]
                    else:
                        # self.send_debug(f"Received 'C'. Expected is now {content[0]}, content[1] is {content[1]}")
                        self.expected_updates[content[1] % (self.slack + 1)] = content[0]
                if msg_type == 'S':
                    # 'S'lowest step message sent by the supervisor (slack != 0)
                    self.slowest_step = content
                elif msg_type == 'U':
                    # 'U'pdate message sent by other workers containing updates
                    tg0 = time.time()
                    self.updates_queue.append(content[0])
                    tg1 = time.time()
                    if self.slack == 0:
                        self.received_updates[w_id] = True
                    else:
                        self.received_updates[content[1] % (self.slack + 1)][w_id] = True
                    t_au_1 = time.time()
                    self.fetch_update_time += tg1 - tg0
                elif msg_type == 'K':
                    # 'K'ill message sent by the supervisor when a worker is no longer needed
                    self.send_debug("Received 'K' - Finishing execution...")
                    self.end = True
                    self.killed = True
                elif msg_type == 'D':
                    # 'D'ead message sent by another worker when it was killed by the supervisor
                    key, self.expected_updates = content
                    received = self._get_object_from_msg(key)
                    self.received_models.append(received)
                    n_workers -= 1
                elif msg_type == 'F':
                    # 'F'inish msg sent by the supervisor when the convergence threshold has been reached
                    self.end = True
                    ch.stop_consuming()
                elif msg_type == 'EXCEPTION':
                    self.end = True
                    ch.stop_consuming()

            if self.slack == 0:
                if self.expected_updates == self.received_updates:
                    ch.stop_consuming()
                    self.expected_updates = None
                    self.received_updates = {}
            else:
                if self.expected_updates[self.slowest_step % (self.slack + 1)] == \
                        self.received_updates[self.slowest_step % (self.slack + 1)]:
                    self.send_updates_processed(self.slowest_step)
                    self.expected_updates[self.slowest_step % (self.slack + 1)] = None
                    self.received_updates[self.slowest_step % (self.slack + 1)] = {}
                else:
                    pass
                if (step - self.slowest_step) < self.slack:
                    ch.stop_consuming()

        self.consume(callback=on_message, queue=self.queue)

        self.total_t_barrier += time.time() - t_au_1
        self.total_t_au += t_au_1 - t_au_0

        return n_workers

    def _get_object_from_msg(self, msg):
        return msg

    def _send_model_on_death(self, weights):
        self.channel.basic_publish(exchange=self.exchange,
                                   routing_key='',
                                   body=pickle.dumps(('D', self.worker_id, weights)))

    def send_model_on_finish(self, weights):
        self.channel.basic_publish(exchange='',
                                   routing_key='{}_results'.format(self.executor_id),
                                   body=pickle.dumps(weights))


class BackendRabbitMQCommunicator(RabbitMQCommunicator):
    def __init__(self, executor_id, worker_id, slack, storage, rabbitmq_params):
        super().__init__(executor_id, worker_id, slack, rabbitmq_params)
        self.storage = storage

    def send_updates(self, step, updates):
        # self.send_debug(f"Sent updates! Step: {step}")
        key = '{}_up_w{}_i{}'.format(self.executor_id, self.worker_id, step)
        self.storage.put(key, updates)
        self.channel.basic_publish(exchange=self.exchange,
                                   routing_key='',
                                   body=pickle.dumps(("U", self.worker_id, (key, step))))

    def _get_object_from_msg(self, msg):
        return self.storage.get(msg)

    def send_model_on_death(self, weights):
        key = '{}_model_w{}'.format(self.executor_id, self.worker_id)
        self.storage.put(key, weights)
        self.channel.basic_publish(exchange=self.exchange,
                                   routing_key='',
                                   body=pickle.dumps(('D', self.worker_id, (key, self.expected_updates))))

    def send_model_on_finish(self, weights):
        key = '{}_model_r_w{}'.format(self.executor_id, self.worker_id)
        self.storage.put(key, weights)
        self.channel.basic_publish(exchange='',
                                   routing_key='{}_results'.format(self.executor_id),
                                   body=key)
