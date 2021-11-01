import pickle
import zlib
import re

from redis import StrictRedis


class StorageBackend:
    def __init__(self, compression=True, **kwargs):
        self.compression = compression

    def put(self, key, object_):
        data = pickle.dumps(object_)
        if self.compression:
            data = zlib.compress(data)
        self._put(key, data)

    def _put(self, key, object_):
        raise NotImplementedError("Available in subclasses: RedisBackend, CosBackend")

    def get(self, key):
        data = self._get(key)
        if self.compression:
            data = zlib.decompress(data)
        data = pickle.loads(data)
        return data

    def _get(self, key):
        raise NotImplementedError("Available in subclasses: RedisBackend, CosBackend")

    def delete(self, keys):
        raise NotImplementedError("Available in subclasses: RedisBackend, CosBackend")


class RedisBackend(StorageBackend):
    def __init__(self, compression=True, n_redis=1, **kwargs):
        super().__init__(compression)
        kwargs.pop('storage')
        kwargs.pop('bucket')
        redis_hosts = kwargs['redis_hosts']

        self.n_redis = n_redis
        self.storage = []
        for i in range(n_redis):
            self.storage.append(StrictRedis(host=redis_hosts[i]))

    def _put(self, key, object_):
        redis_host = self._extract_host(key)
        self.storage[redis_host].set(key, object_)

    def _get(self, key):
        redis_host = self._extract_host(key)
        object_ = self.storage[redis_host].get(key)
        return object_

    def delete(self, keys):
        for redis in self.storage:
            redis.delete(*keys)

    def _extract_host(self, key):
        w_id = re.findall(r'_w(\d*)', key)[0]
        w_id = int(w_id)
        redis_host = w_id % self.n_redis
        return redis_host


class CosBackend(StorageBackend):
    def __init__(self, storage, bucket, compression=True, **kwargs):
        super().__init__(compression, **kwargs)
        self.storage = storage
        self.bucket = bucket

    def _put(self, key, object_):
        self.storage.put_object(Bucket=self.bucket, Key='{}'.format(key), Body=object_)

    def _get(self, key):
        data = self.storage.get_object(Bucket=self.bucket, Key=key)['Body'].read()
        return data

    def delete(self, keys):
        keys = [{'Key': key} for key in keys]
        delete_dict = {'Objects': keys}
        self.storage.delete_objects(Bucket=self.bucket, Delete=delete_dict)
