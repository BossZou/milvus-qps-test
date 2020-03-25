import datetime
import numpy
import os
import sys
import time
import concurrent.futures
import milvus
import numpy

from milvus_observer.client import MilvusClient
from milvus_observer.dataset import get_dataset
from milvus_observer.utils import generate_combinations
from milvus_observer.utils import FastReadCounter

INSERT_INTERVAL = 50000


def build_all(connect, X_train, collection_scheme, build_param):
    if connect.exists_collection():
        connect.delete()
        time.sleep(2)
    connect.create_collection(
        collection_scheme["collection_name"], collection_scheme["dim"],
        collection_scheme["index_size"], collection_scheme["metric_type"])
    loops = len(X_train) // INSERT_INTERVAL + 1
    for i in range(loops):
        start = i*INSERT_INTERVAL
        end = min((i+1)*INSERT_INTERVAL, len(X_train))
        tmp_vectors = X_train[start:end]
        if start < end:
            connect.insert(
                tmp_vectors, ids=[i for i in range(start, end)])
        connect.flush()
    if connect.count() != len(X_train):
        print("Table row count is not equal to insert vectors")
        return
    connect.create_index(
        collection_scheme["index_type"], index_param=build_param)
    connect.preload_collection()


def run(definition, connection_num, run_count, batch, searchonly):
    print("  run_count:%d, batch:%r, clients:%d, searchonly:%r" %
          (run_count, batch, connection_num, searchonly))
    collection_scheme = definition["collection_scheme"]

    X_train, X_test = get_dataset(definition)
    build_params = generate_combinations(definition["build_args"])
    search_params = generate_combinations(definition["search_args"])
    if not searchonly:
        for pos, build_param in enumerate(build_params, 1):
            print("Running train argument group %d of %d..." %
                  (pos, len(build_params)))
            print("build_params:", build_param)
            client = MilvusClient(
                collection_name=collection_scheme["collection_name"])
            build_all(client, X_train, collection_scheme, build_param)
            run_paralle(
                search_params, collection_scheme["collection_name"], connection_num, X_test, run_count, batch)
    else:
        run_paralle_process(
            search_params, collection_scheme["collection_name"], connection_num, X_test, run_count, batch)


def run_paralle(search_params, collection_name, connection_num, X_test, run_count, batch):
    pool = [MilvusClient(collection_name=collection_name)
            for n in range(connection_num)]
    for pos, search_param in enumerate(search_params, 1):
        # batch_size = int(search_param["testsize"]/connection_num)
        # if batch_size <= 0:
        #     print("Error: testsize < clients, skip")
        #     return

        print("Running search argument group %d of %d..." %
              (pos, len(search_params)))
        print("collection: %s, search_params: %s" %
              (collection_name, search_param))
        if search_param["testsize"] == 1:
            query_vector = [X_test[0]]
        else:
            query_vector = X_test[0:search_param["testsize"]]

        min_total_time = float('inf')
        for _ in range(run_count):
            counter = FastReadCounter()
            total_time = float('-inf')
            with concurrent.futures.ThreadPoolExecutor(max_workers=connection_num) as executor:
                future_results = {executor.submit(
                    run_single_query, pool[pos], counter, [query_vector[0]], search_param): pos for pos in range(connection_num)}
                for future in concurrent.futures.as_completed(future_results):
                    data = future.result()
                    total_time = total_time if total_time > data["total_time"] else data["total_time"]
                min_total_time = min_total_time if min_total_time < total_time else total_time
        average_search_time = min_total_time / search_param["testsize"]
        print("QPS: %d\n" % (1.0 / average_search_time))


def run_single_query(connect, counter, query, search_param):
    start = time.time()
    while counter.value < search_param["testsize"]:
        counter.increment()
        connect.query(query, search_param["topk"], search_param=search_param)
    total_time = time.time() - start
    attrs = {"total_time": total_time}
    return attrs


def run_paralle_process(search_params, collection_name, connection_num, X_test, run_count, batch):
    for pos, search_param in enumerate(search_params, 1):
        print("Running search argument group %d of %d..." %
              (pos, len(search_params)))
        print("collection: %s, search_params: %s" %
              (collection_name, search_param))
        if search_param["testsize"] == 1:
            query_vector = [X_test[0]]
        else:
            query_vector = X_test[0:search_param["testsize"]]

        test_time = 20
        max_runs = int('-inf')
        for _ in range(run_count):
            start = time.time() + 3
            end = start + test_time
            runs = 0
            with concurrent.futures.ProcessPoolExecutor(max_workers=connection_num) as executor:
                future_results = {executor.submit(run_single_query_process, collection_name, [
                                                  query_vector[0]], search_param, start, end)}
                for future in concurrent.futures.as_completed(future_results):
                    data = future.result()
                    runs += data["runs"]
                max_runs = max_runs if max_runs > runs else runs
        average_search_time = float(test_time / max_runs)
        print("QPS: %d\n" % (1.0 / average_search_time))


def run_single_query_process(collection_name, query, search_param, starttime, endtime):
    connect = MilvusClient(collection_name=collection_name)
    counter = 0

    while time.time() < starttime:
        pass

    while time.time() < endtime:
        connect.query(query, search_param["topk"], search_param=search_param)
        counter += 1
    attrs = {"runs": counter}
    return attrs
