import threading
from queue import Queue
import time

PRODUCER_KILL = -1
PRODUCER_CONTINUE = 0
PRODUCER_RETRY = 1
WAITING_TIME = 2


class ProducerConsumerQueue(object):
    def __init__(self, create_produce_object_func, upper_bound=5):
        self.queue = Queue(upper_bound)
        self.lock = threading.Lock()
        self.create_produce_object_func = create_produce_object_func
        self.upper_bound = upper_bound
        self.producers = 1
        self._create_producer()

    def produce(self, obj):
        with self.lock:
            if self.queue.full():
                if self.producers > 1:
                    self.producers -= 1
                    return PRODUCER_KILL
                else:
                    return PRODUCER_RETRY
            else:
                self.queue.put(obj)
                return PRODUCER_CONTINUE

    def consume(self):
        self.lock.acquire()
        queue_obj = None
        create_producer_flag = False
        try:
            if self.queue.empty():
                self._create_producer()
                self.producers += 1
                create_producer_flag = True
                self.lock.release()
            queue_obj = self.queue.get()
        finally:
            if not create_producer_flag:
                self.lock.release()
            return queue_obj.get()

    def _create_producer(self):
        producer = Producer(self, self.create_produce_object_func())
        producer.start()


class Producer(threading.Thread):
    def __init__(self, queue, produce_obj):
        threading.Thread.__init__(self)
        self.queue = queue
        self.produce_obj = produce_obj

    def run(self):
        return_val = PRODUCER_CONTINUE
        obj = None
        while return_val != PRODUCER_KILL:
            if return_val == PRODUCER_CONTINUE:
                obj = self.produce_obj()
            if return_val == PRODUCER_RETRY:
                time.sleep(WAITING_TIME)
            return_val = self.queue.produce(obj)


class QueueObj(object):
    def __init__(self, object_list):
        self.obj = object_list

    def get(self):
        return self.obj[0]
