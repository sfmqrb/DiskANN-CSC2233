import os
import sys
from typing import Dict, Tuple, Callable, List
import itertools
import pprint
import time
import traceback
import datetime
import subprocess
import re
import json
import glob

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from lib.util import create_build, run_and_parse_output, download_sift, revise_sift1B


class Timed:
    def __init__(self, print_: Callable = None):
        self.print_ = print_

    def __enter__(self):
        self.start_ = time.perf_counter()
        self.end_ = None
        self.time_elapsed_ = None
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.end_ = time.perf_counter()
        self.time_elapsed_ = self.end_ - self.start_
        if self.print_ is not None:
            self.print_(f"elapsed {self.time_elapsed_:.5f} seconds")
        if exc_type is not None:
            print("".join(traceback.format_exception(exc_type, exc_value, exc_tb)), file=sys.stderr)

    def time_elapsed(self):
        return self.time_elapsed_


def binded_print(*args):
    def _print(*p_args):
        print(*args, *p_args)

    return _print


def generate_key(params, prefix="", excluded_keys=()):
    results = []
    for k, v in params.items():
        if k in excluded_keys: continue
        if isinstance(v, list):
            v = "-".join([str(vi) for vi in v])
        results.append(f"{k}{v}")
    results.sort()
    if prefix != "":
        results.insert(0, prefix)
    return "_".join(results)


def generate_keys(params_list, **kwargs):
    return [generate_key(params, **kwargs) for params in params_list]


def get_as_list(dict_, key, default):
    val = dict_.get(key, default)
    if not isinstance(val, list):
        val = [val]
    return val


def group_benchmark_params(all_params):
    """
    Group benchmark parameters to runs that can be executed together
    """
    # benchmarks[build_key][index_key] = (build_param, index_param, query_param)
    benchmarks: Dict[str, Dict[str, Tuple[Dict, Dict, list]]] = {}
    for param in all_params:
        build_isolate_alpha = get_as_list(param, "build_isolate_alpha", ["Off"])
        build_knn_index = get_as_list(param, "build_knn_index", [0])
        build_tracking = get_as_list(param, "build_tracking", [False])
        for build_isolate_alpha_, build_knn_index_, build_tracking_ in itertools.product(build_isolate_alpha, build_knn_index, build_tracking):
            build_param = {
                "isolate_alpha": build_isolate_alpha_,
                "knn_index": build_knn_index_,
                "tracking": build_tracking_,
            }
            build_key = generate_key(build_param)
            if build_key not in benchmarks:
                benchmarks[build_key] = {}
            index_data = get_as_list(param, "index_data", ["sift_learn.fbin"])
            index_l = get_as_list(param, "index_l", [64])
            index_r = get_as_list(param, "index_r", [64])
            index_alpha = get_as_list(param, "index_alpha", [1.2])
            index_saturate_graph = get_as_list(param, "index_saturate_graph", [False])
            query_data = get_as_list(param, "query_data", ["sift_query.fbin"])
            query_k = get_as_list(param, "query_k", [10])
            query_l = get_as_list(param, "query_l", [64])
            for index_data_, index_l_, index_r_, index_alpha_, index_saturate_graph_ in itertools.product(
                    index_data, index_l, index_r, index_alpha, index_saturate_graph):
                index_param = {
                    "data": index_data_,
                    "l": index_l_,
                    "r": index_r_,
                    "alpha": index_alpha_,
                    "saturate_graph": index_saturate_graph_
                }
                index_key = generate_key(index_param)
                if index_key not in benchmarks[build_key]:
                    benchmarks[build_key][index_key] = (build_param, index_param, [])
                for query_data_, query_k_ in itertools.product(query_data, query_k):
                    query_param = {
                        "num_runs": param.get("query_num_runs", 1),
                        "data": query_data_,
                        "k": query_k_,
                        "l": query_l,  # L support passing as list
                    }
                    benchmarks[build_key][index_key][2].append(query_param)
    return [l for ll in benchmarks.values() for l in ll.values()]


DEFAULT_TRACKING_PORT = 5555


def format_cmd(cmd: List[str]):
    grouped = []
    i = 0
    last_group_start = i
    while i < len(cmd):
        if cmd[i].startswith("-"):
            grouped.append(" ".join(cmd[last_group_start: i]))
            last_group_start = i
        i += 1
    grouped.append(" ".join(cmd[last_group_start:]))
    return "    ".join(grouped) + "\n"


def generate_random_dataset(rand_data_gen, output_dir, n_name, N, D=128):
    os.makedirs(output_dir, exist_ok=True)
    random_vector_file = os.path.join(output_dir, f"rand_{D}_{n_name}.fbin")
    if os.path.exists(random_vector_file):
        print(f"Dataset rand_{D}_{n_name} already exists")
        return random_vector_file  # no need to regenerate
    cmd = [
        rand_data_gen,
        "--data_type", "float",
        "--output_file", random_vector_file,
        "--ndims", str(D),
        "--npts", str(N),
    ]
    run_serialized_command(cmd, None)
    print(f"Dataset rand_{D}_{n_name} is ready")
    return random_vector_file


def run_build_step(state, temp_state, trans_state, isolate_alpha, knn_index, tracking):
    project_root = temp_state["project_root"]
    build_key = trans_state["build_key"]
    build_path = os.path.join(project_root, "build", build_key)
    data_path = os.path.join(project_root, "build", "data")
    if build_path not in temp_state["recent_builds"]:
        build_path = create_build(project_root, build_subdir=build_key,
                                  tracking=tracking, type="Release", ISOLATE_ALPHA=isolate_alpha, KNN_INDEX=knn_index)
        temp_state["recent_builds"].add(build_path)
    sift_data_path = os.path.join(data_path, "sift")
    rand_data_path = os.path.join(data_path, "rand")
    bigann_data_path = os.path.join(data_path, "bigann")
    if "bigann" in temp_state["required_datasets"] and "bigann" not in temp_state["recent_datasets"]:
        revise_sift1B(data_path, os.path.join(build_path, "apps"))
        temp_state["recent_datasets"].add("bigann") 
    if "sift" in temp_state["required_datasets"] and "sift" not in temp_state["recent_datasets"]:
        download_sift(data_path, os.path.join(build_path, "apps"))
        temp_state["recent_datasets"].add("sift")
    if "rand" in temp_state["required_datasets"] and "rand" not in temp_state["recent_datasets"]:
        rand_data_gen = os.path.join(build_path, "apps", "utils", "rand_data_gen")
        generate_random_dataset(rand_data_gen, rand_data_path, "1k", N=1_000, D=128)
        generate_random_dataset(rand_data_gen, rand_data_path, "10k", N=10_000, D=128)
        generate_random_dataset(rand_data_gen, rand_data_path, "1m", N=1_000_000, D=128)
        generate_random_dataset(rand_data_gen, rand_data_path, "10m", N=10_000_000, D=128)
        temp_state["recent_datasets"].add("rand")
    trans_state.update({
        "build_path": build_path,
        "sift_data_path": sift_data_path,
        "rand_data_path": rand_data_path,
        "bigann_data_path": bigann_data_path,
        "build_memory_index": os.path.join(build_path, "apps", "build_memory_index"),
        "search_memory_index": os.path.join(build_path, "apps", "search_memory_index"),
        "compute_groundtruth": os.path.join(build_path, "apps", "utils", "compute_groundtruth"),
    })


def run_serialized_command(cmd, output_dir, wait_one_sec=False):
    # flush a CMD file for easier debug
    cmd_dump = format_cmd(cmd)
    if output_dir is not None:
        cmd_file = os.path.join(output_dir, "CMD")
        with open(cmd_file, "w") as f:
            f.write(cmd_dump)

    # run command
    if output_dir is not None:
        stdout_file = os.path.join(output_dir, "STDOUT")
        stderr_file = os.path.join(output_dir, "STDERR")
        stdout_f = open(stdout_file, "w")
        stderr_f = open(stderr_file, "w")
    else:
        stdout_file = None
        stderr_file = None
        stdout_f = subprocess.DEVNULL
        stderr_f = subprocess.DEVNULL

    process = subprocess.Popen(cmd, stdout=stdout_f, stderr=stderr_f, text=True)
    process.wait()
    if wait_one_sec:
        time.sleep(1)

    if output_dir is not None:
        stdout_f.close()
        stderr_f.close()
    # if process.returncode == 0:
    #     print("stdout running command:")
    #     print(cmd_dump)
    #     print("Return code:", process.returncode)
    #     with open(stdout_file, "r") as f:
    #         std_output = f.read()
    #     print("STDOUT:")
    #     print(std_output)
     
    if process.returncode != 0:
        print("Error running command:")
        print(cmd_dump)
        print("Return code:", process.returncode)
        if output_dir is not None:
            # Read and display STDOUT and STDERR content for a detailed log
            with open(stdout_file, "r") as f:
                std_output = f.read()
            with open(stderr_file, "r") as f:
                err_output = f.read()
            print("STDOUT:")
            print(std_output)
            print("STDERR:")
            print(err_output)
        # Optionally, you could raise an exception with these details:
        # raise Exception(f"Command failed with return code {process.returncode}.\nSTDOUT: {std_output}\nSTDERR: {err_output}")

    return stdout_file, stderr_file

def extract_index_step_data(write_output, diskann_index_path):
    indexing_time = None
    index_build_info = {}

    with open(write_output, 'r') as file:
        for line in file:
            # Extract indexing time
            if indexing_time is None:
                match_time = re.search(r"Indexing time:\s*([\d.]+)", line)
                if match_time:
                    indexing_time = float(match_time.group(1))

            # Extract index build degree data
            match_build = re.search(
                r"Index built with degree:\s*max:(\d+)\s+avg:([\d.]+)\s+min:(\d+)\s+count\(deg<2\):(\d+)",
                line
            )
            if match_build:
                index_build_info['max_degree'] = int(match_build.group(1))
                index_build_info['avg_degree'] = float(match_build.group(2))
                index_build_info['min_degree'] = int(match_build.group(3))
                index_build_info['count_deg_lt_2'] = int(match_build.group(4))

    index_size_byte = 0
    index_data_size_byte = 0
    if indexing_time is not None:
        index_size_byte = os.path.getsize(diskann_index_path)
        index_data_size_byte = os.path.getsize(diskann_index_path + ".data")
    else:
        indexing_time = -1.0

    return {
        "indexing_time_seconds": indexing_time,
        "index_size_byte": index_size_byte,
        "index_data_size_byte": index_data_size_byte,
        **index_build_info
    }


def index_benchmark_exist(state, build_key, index_key):
    if "index_raw_data" not in state or f"{build_key}_{index_key}" not in state["index_raw_data"]:
        return False
    return True


def run_index_step(state, temp_state, trans_state, data, l, r, alpha, saturate_graph, use_existing_index=True, data_type='float'):
    project_root: str = temp_state["project_root"]
    build_key = trans_state["build_key"]
    index_key = trans_state["index_key"]
    index_base_path: str = os.path.join(project_root, "build", build_key, index_key)
    diskann_index_path: str = os.path.join(index_base_path, "index_output")
    diskann_index_data_path: str = os.path.join(index_base_path, "index_output.data")
    if not use_existing_index or not os.path.exists(diskann_index_path):
        os.makedirs(index_base_path, exist_ok=True)

        # prepare CMD
        build_memory_index = trans_state["build_memory_index"]
        data_path_prefix = trans_state["sift_data_path"] if data.startswith("sift") else trans_state["rand_data_path"]
        if (data.startswith("bigann")):
            data_path_prefix = trans_state["bigann_data_path"]
        data_path = os.path.join(data_path_prefix, data)
        cmd = [
            build_memory_index,
            "--data_type", data_type,
            "--dist_fn", "l2",
            "--data_path", data_path,
            "--index_path_prefix", diskann_index_path,
            "-R", str(r),
            "-L", str(l),
            "--alpha", str(alpha),
            "--num_threads", "1",
            "--tracking_addr", f"tcp://localhost:{DEFAULT_TRACKING_PORT}",
            "--saturate_graph" if saturate_graph else "",
        ]

        print(cmd)
        # build index and process output
        stdout_file, _ = run_serialized_command(cmd, index_base_path)
    else:
        stdout_file = os.path.join(index_base_path, "STDOUT")

    expr_data = extract_index_step_data(stdout_file, diskann_index_path)
    expr_data_key = f"{build_key}_{index_key}"
    if "index_raw_data" not in state:
        state["index_raw_data"] = {}
    state["index_raw_data"][expr_data_key] = expr_data

    trans_state.update({
        "diskann_index_path": diskann_index_path,
        "diskann_index_data_path": diskann_index_data_path,
        "indexed_data_file": data,
    })


def query_benchmark_exist(state, build_key, index_key, *query_keys):
    if "query_raw_data" not in state or any(
            f"{build_key}_{index_key}_{query_key}" not in state["query_raw_data"] for query_key in query_keys):
        return False
    return True


def extract_query_step_data(out_files):
    data = []
    pattern = re.compile(r"\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)")

    for out_file in out_files:
        with open(out_file, 'r') as file:
            for line in file:
                match = pattern.match(line)
                if match:
                    l, qps, avg_dist_cmps, mean_lat, p99_lat, recall = (float(match.group(i)) for i in
                                                                        range(1, 7))
                    data.append({
                        "l": l,
                        "QPS": qps,
                        "average_distance_compare": avg_dist_cmps,
                        "mean_latency": mean_lat,
                        "p99_latency": p99_lat,
                        "recall": recall
                    })
    return data


def get_ground_truth(state, temp_state, trans_state, data_file: str, query_data_file: str, k=100, data_type='float'):
    data_path_prefix = trans_state["sift_data_path"] if data_file.startswith("sift") else trans_state["rand_data_path"]
    if (data_file.startswith("bigann")):
        data_path_prefix = trans_state["bigann_data_path"]
    ground_truth_file = os.path.join(data_path_prefix, "d{}_q{}.gt_{}".format(data_file.replace(".fbin", ""),
                                                                              query_data_file.replace(".fbin", ""),
                                                                              k))
    if not os.path.exists(ground_truth_file):
        compute_groundtruth = trans_state["compute_groundtruth"]
        cmd = [
            compute_groundtruth,
            "--data_type", data_type,
            "--dist_fn", "l2",
            "--base_file", os.path.join(data_path_prefix, data_file),
            "--query_file", os.path.join(data_path_prefix, query_data_file),
            "--gt_file", ground_truth_file,
            "--K", str(k)
        ]
        run_serialized_command(cmd, None)
    return ground_truth_file


def run_query_step(state, temp_state, trans_state, query_key, num_runs, data, k, l, data_type='float'):
    project_root: str = temp_state["project_root"]
    build_key = trans_state["build_key"]
    index_key = trans_state["index_key"]
    query_base_path: str = os.path.join(project_root, "build", build_key, index_key, query_key)
    os.makedirs(query_base_path, exist_ok=True)

    search_memory_index = trans_state["search_memory_index"]
    diskann_index_path = trans_state["diskann_index_path"]
    indexed_data_file = trans_state["indexed_data_file"]
    ground_truth_file = get_ground_truth(state, temp_state, trans_state, indexed_data_file, data, data_type=data_type)
    data_path_prefix = trans_state["sift_data_path"] if data.startswith("sift") else trans_state["rand_data_path"]
    if (data.startswith("bigann")):
        data_path_prefix = trans_state["bigann_data_path"]
    data_path = os.path.join(data_path_prefix, data)

    run_result_bin_files = []
    run_stdouts = []
    for run_i in range(num_runs):
        query_run_path = os.path.join(query_base_path, f"run_{run_i}")
        result_path = os.path.join(query_run_path, "result")
        os.makedirs(query_run_path, exist_ok=True)
        cmd = [
            search_memory_index,
            "--data_type", data_type,
            "--dist_fn", "l2",
            "--index_path_prefix", diskann_index_path,
            "--query_file", data_path,
            "--gt_file", ground_truth_file,
            "-K", str(k),
            "-L", *[str(l_) for l_ in l],
            "--result_path", result_path,
            "--num_threads", "1"
        ]
        stdout, _ = run_serialized_command(cmd, query_run_path)
        run_stdouts.append(stdout)
        bin_file_pattern = os.path.join(query_run_path, 'result*.bin')
        run_result_bin_files.extend(glob.glob(bin_file_pattern))

    expr_data = extract_query_step_data(run_stdouts)
    expr_data_key = f"{build_key}_{index_key}_{query_key}"
    if "query_raw_data" not in state:
        state["query_raw_data"] = {}
    state["query_raw_data"][expr_data_key] = expr_data

    trans_state.update({
        "query_run_result_bin_files": run_result_bin_files,
    })


def run_serialized_benchmarks(project_root, params, last_state=None, dry_run=False, forced_run_all=False,
                              use_existing_index=True, delete_index_after_query=True):
    # relays on stateful system:
    # - persist_state: persisted states
    # - temp_state: temporary states, exist across different benchmarks
    # - trans_state: transient states, only exist across benchmark steps (in one benchmark)

    def benchmark_main(state):
        # collect required dataset
        referenced_datasets = set()
        for _, index_param, query_params in params:
            referenced_datasets.add(index_param["data"])
            for query_param in query_params:
                referenced_datasets.add(query_param["data"])
        required_datasets = []
        if any(ds.startswith("sift") for ds in referenced_datasets):
            required_datasets.append("sift")
        if any(ds.startswith("rand") for ds in referenced_datasets):
            required_datasets.append("rand")
        if any(ds.startswith("bigann") for ds in referenced_datasets):
            required_datasets.append("bigann")
        # temporary states, exist across different benchmarks (not persisted)
        
        temp_state = {"project_root": project_root,
                      "recent_builds": set(),
                      "recent_datasets": set(),
                      "required_datasets": required_datasets}

        num_benchmarks = len(params)
        num_sub_benchmarks = sum(qp["num_runs"] for _, _, query_params in params for qp in query_params)
        print(f"Collected {num_benchmarks} benchmarks, including {num_sub_benchmarks} sub benchmarks.")
        accum_elapsed_time = 0.
        accum_num_benchmarks = 0
        for i, (build_param, index_param, query_params) in enumerate(params):
            if dry_run:
                print("- build_param:",
                      pprint.pformat(build_param, sort_dicts=True, compact=True, underscore_numbers=True))
                print("- index_param:",
                      pprint.pformat(index_param, sort_dicts=True, compact=True, underscore_numbers=True))
                print("- query_params:")
                for query_i, query_param in enumerate(query_params):
                    print(f"  - [{query_i}]",
                          pprint.pformat(query_param, sort_dicts=True, compact=True, underscore_numbers=True))
                continue

            trans_state = {}

            # check can skip
            build_key = generate_key(build_param, prefix="build")
            index_key = generate_key(index_param, prefix="index")
            query_keys = generate_keys(query_params, prefix="query", excluded_keys={"num_runs", "l"})
            trans_state["build_key"] = build_key
            trans_state["index_key"] = index_key
            if (not forced_run_all
                    and index_benchmark_exist(state, build_key, index_key)
                    and query_benchmark_exist(state, build_key, index_key, *query_keys)):
                print("Skipped {}/{}, data already exists".format(i + 1, num_benchmarks))
                continue

            total_elapsed_time = 0.

            # build step
            with Timed(binded_print(f" {i + 1:>3}/{num_benchmarks} [build]")) as t:
                run_build_step(state, temp_state, trans_state, **build_param)
            total_elapsed_time += t.time_elapsed()

            # index step
            # need to run index, regardless of it exists of not, because query depends on it
            with Timed(binded_print(f" {i + 1:>3}/{num_benchmarks} [index]")) as t:
                run_index_step(state, temp_state, trans_state, use_existing_index=use_existing_index, **index_param, data_type='uint8')
            total_elapsed_time += t.time_elapsed()

            # query step
            for query_i, (query_param, query_key) in enumerate(zip(query_params, query_keys)):
                if not query_benchmark_exist(state, build_key, index_key, query_key):
                    with Timed(binded_print(f" {i + 1:>3}/{num_benchmarks} [query_{query_i}]")) as t:
                        run_query_step(state, temp_state, trans_state, query_key, **query_param, data_type='uint8')
                    total_elapsed_time += t.time_elapsed()

            # remove the index after query to conserve disk space
            if delete_index_after_query:
                try:
                    os.remove(trans_state["diskann_index_path"])
                    os.remove(trans_state["diskann_index_data_path"])
                    for bin_file in trans_state["query_run_result_bin_files"]:
                        os.remove(bin_file)
                except:
                    print("Failed to cleanup index files.")

            # estimate remaining time
            accum_elapsed_time += total_elapsed_time
            accum_num_benchmarks += 1
            estimated_time_remain = (accum_elapsed_time / accum_num_benchmarks) * (num_benchmarks - i - 1)
            print("Completed {}/{}, estimated remaining {}".format(i + 1, num_benchmarks, datetime.timedelta(
                seconds=estimated_time_remain)))

    persist_state = last_state if last_state is not None else {}
    try:
        with Timed(binded_print("Benchmark main loop")):
            benchmark_main(persist_state)
    except Exception as e:
        print("run_serialized_benchmarks stopped early due to exception")
        print(e)
    finally:
        return persist_state


def save_state(state_dict, out_dir):
    timestamp = datetime.datetime.now()
    json_object = {
        "timestamp": timestamp.strftime('%Y_%m_%d %H:%M:%S'),
        "state": state_dict
    }
    file_name = "state_{}.json".format(timestamp.strftime("%Y%m%d_%H%M%S"))
    result_file_name = os.path.join(out_dir, file_name)
    with open(result_file_name, "w") as f:
        json.dump(json_object, f, indent=4)
    print(f"State saved to {file_name}")
    return result_file_name


def load_state(state_file):
    with open(state_file, "r") as file:
        json_object = json.load(file)
    print(f"State loaded from \"{os.path.basename(state_file)}\" (timestamp = {json_object['timestamp']})")
    return json_object["state"]


def combine_states_to_df(*states, params, params_to_key, flatten_param, subkeys=(), subkeys_as=()):
    # convert all params to key
    key_to_param = {}
    for build_param, index_param, query_params in params:
        for query_param in query_params:
            param_key = params_to_key(build_param, index_param, query_param)
            key_to_param[param_key] = flatten_param(build_param, index_param, query_param)

    # group the rows together
    grouped_rows: Dict[str, list[Dict]] = {}

    def add_to_group(group_key, param_dict, row_dict):
        if group_key not in grouped_rows:
            grouped_rows[group_key] = []
        # replace subkeys in row_dict
        row_dict_replaced = dict(row_dict)
        for sk, sk_as in zip(subkeys, subkeys_as):
            sk_val = row_dict_replaced[sk]
            del row_dict_replaced[sk]
            row_dict_replaced[sk_as] = sk_val
        grouped_rows[group_key].append({**param_dict, **row_dict_replaced})

    for state in states:
        for k, v in state.items():
            matched_param = key_to_param[k]
            if isinstance(v, dict):
                grouping_key = "_".join([k, *[f"{sk_as}{v[sk]}" for sk, sk_as in zip(subkeys, subkeys_as)]])
                add_to_group(grouping_key, matched_param, v)
            else:
                for vv in v:
                    grouping_key = "_".join([k, *[f"{sk_as}{vv[sk]}" for sk, sk_as in zip(subkeys, subkeys_as)]])
                    add_to_group(grouping_key, matched_param, vv)

    rows = [row for group in grouped_rows.values() for row in group]
    return pd.DataFrame(rows)


def index_param_to_key(build_param, index_param, _):
    return "{}_{}".format(generate_key(build_param, "build"), generate_key(index_param, "index"))


def flatten_index_param(build_param, index_param, _):
    return {
        **{f"build_{k}": v for k, v in build_param.items()},
        **{f"index_{k}": v for k, v in index_param.items()},
    }


def query_param_to_key(build_param, index_param, query_param):
    return "{}_{}_{}".format(generate_key(build_param, "build"), generate_key(index_param, "index"),
                             generate_key(query_param, "query", excluded_keys={"num_runs", "l"}))


def flatten_query_param(build_param, index_param, query_param):
    return {
        **{f"build_{k}": v for k, v in build_param.items()},
        **{f"index_{k}": v for k, v in index_param.items()},
        **{f"query_{k}": v for k, v in query_param.items() if k not in ["l", "num_runs"]},
    }


def collect_alpha_data_points(df, all_alpha, column_keys):
    data = np.array([df[df["index_alpha"] == alpha][column_keys] for alpha in all_alpha])
    data = data.reshape(len(all_alpha), len(column_keys), -1)
    mean_data = data.mean(axis=2).transpose()
    min_data = data.min(axis=2).transpose()
    max_data = data.max(axis=2).transpose()
    return mean_data, min_data, max_data


PLOT_COLORS = ["lightcoral", "red", "lightgreen", "green", "lightblue", "blue", "plum", "purple"]
PLOT_COLORS_2 = ["red", "green", "blue", "purple", "darkorange", "saddlebrown"]


def plot_construction_benchmark(df: pd.DataFrame, columns, name_prefix):
    all_index_l = sorted(df["index_l"].unique())
    all_alpha = sorted(df["index_alpha"].unique())
    all_index_r = sorted(df["index_r"].unique())
    all_index_dataset = sorted(df["index_data"].unique())
    all_isolate_alpha = sorted(df["build_isolate_alpha"].unique())
    all_knn_index = sorted(df["build_knn_index"].unique())
    n_row = len(all_index_r) * len(all_index_dataset)
    n_col = len(columns)
    fig, axs = plt.subplots(n_row, n_col, figsize=(n_col * 5, n_row * 5), squeeze=False)
    for plot_row, (index_r, index_dataset) in enumerate(itertools.product(all_index_r, all_index_dataset)):
        for plot_col, column_name in enumerate(columns):
            for line_i, (index_l, isolate_alpha, knn_index) in enumerate(itertools.product(all_index_l, all_isolate_alpha, all_knn_index)):
                line_name = f"L_index={index_l}, isolate_alpha={isolate_alpha}, knn_index={knn_index}"
                line_color = PLOT_COLORS[line_i % len(PLOT_COLORS)]
                line_mean, line_min, line_max = collect_alpha_data_points(
                    df[(df["index_l"] == index_l) &
                       (df["index_r"] == index_r) &
                       (df["index_data"] == index_dataset) &
                       (df["build_isolate_alpha"] == isolate_alpha) &
                       (df["build_knn_index"] == knn_index)],
                    all_alpha, [column_name]
                )
                # only one dimension
                line_mean = line_mean.squeeze()
                line_min = line_min.squeeze()
                line_max = line_max.squeeze()
                axs[plot_row][plot_col].fill_between(all_alpha, line_min, line_max, color=line_color, alpha=0.1)
                axs[plot_row][plot_col].plot(all_alpha, line_mean, "^-", label=line_name, color=line_color)
            axs[plot_row][plot_col].set_xlabel("alpha")
            axs[plot_row][plot_col].set_ylabel(column_name)
            axs[plot_row][plot_col].set_title(f"R_index={index_r}, dataset={index_dataset}")
            axs[plot_row][plot_col].grid(True, linestyle='--', alpha=0.7)
    # fig.suptitle(fr"$\alpha$-sensitivity benchmark: index construction")
    handles, labels = axs[0][0].get_legend_handles_labels()
    legend_width_frac = 0.07
    legend_ax = fig.add_axes((0.0, 0.0, legend_width_frac, 1))
    legend_ax.axis('off')
    legend_ax.legend(handles, labels, loc='right', title="Legend")
    fig.tight_layout(rect=(legend_width_frac, 0, 1, 1))
    plt.savefig("{}_index_benchmark.png".format(name_prefix.replace(" ", "_").lower()), bbox_inches="tight")


def plot_query_benchmark(df: pd.DataFrame, columns, name_prefix):
    all_index_l = sorted(df["index_l"].unique())
    all_alpha = sorted(df["index_alpha"].unique())
    all_index_r = sorted(df["index_r"].unique())
    all_index_dataset = sorted(df["index_data"].unique())
    all_isolate_alpha = sorted(df["build_isolate_alpha"].unique())
    all_knn_index = sorted(df["build_knn_index"].unique())

    all_query_l = sorted(df["query_l"].unique())
    all_query_dataset = sorted(df["query_data"].unique())
    n_row = len(all_index_r) * len(all_index_dataset) * len(all_query_l) * len(all_query_dataset)
    n_col = len(columns)
    fig, axs = plt.subplots(n_row, n_col, figsize=(n_col * 5, n_row * 5), squeeze=False)
    for plot_row, (index_r, index_dataset, query_l, query_dataset) in enumerate(itertools.product(
            all_index_r, all_index_dataset, all_query_l, all_query_dataset)):
        for plot_col, column_name in enumerate(columns):
            for line_i, (index_l, isolate_alpha, knn_index) in enumerate(itertools.product(all_index_l, all_isolate_alpha, all_knn_index)):
                line_name = f"L_index={index_l}, isolate_alpha={isolate_alpha}, knn_index={knn_index}"
                line_color = PLOT_COLORS[line_i % len(PLOT_COLORS)]
                line_mean, line_min, line_max = collect_alpha_data_points(
                    df[(df["index_l"] == index_l) &
                       (df["index_r"] == index_r) &
                       (df["index_data"] == index_dataset) &
                       (df["query_l"] == query_l) &
                       (df["query_data"] == query_dataset) &
                        (df["build_knn_index"] == knn_index) &
                       (df["build_isolate_alpha"] == isolate_alpha)],
                    all_alpha, [column_name]
                )
                # only one dimension
                line_mean = line_mean.squeeze()
                line_min = line_min.squeeze()
                line_max = line_max.squeeze()
                axs[plot_row][plot_col].fill_between(all_alpha, line_min, line_max, color=line_color, alpha=0.1)
                axs[plot_row][plot_col].plot(all_alpha, line_mean, "^-", label=line_name, color=line_color)
            axs[plot_row][plot_col].set_xlabel("alpha")
            axs[plot_row][plot_col].set_ylabel(column_name)
            axs[plot_row][plot_col].set_title(
                f"dataset={index_dataset}, {query_dataset}\nR_index={index_r}, L_query={query_l}")
            axs[plot_row][plot_col].grid(True, linestyle='--', alpha=0.7)
    # fig.suptitle(fr"$\alpha$-sensitivity benchmark: query")
    handles, labels = axs[0][0].get_legend_handles_labels()
    legend_width_frac = 0.07
    legend_ax = fig.add_axes((0.0, 0.0, legend_width_frac, 1))
    legend_ax.axis('off')
    legend_ax.legend(handles, labels, loc='right', title="Legend")
    fig.tight_layout(rect=(legend_width_frac, 0, 1, 1))
    plt.savefig("{}_query_benchmark.png".format(name_prefix.replace(" ", "_").lower()), bbox_inches="tight")


def dominates(a, b, objectives):
    """
    Returns True if point a dominates point b according to the objectives.
    Each element in 'objectives' should be either 'min' or 'max'.
    """
    strictly_better = False
    for i, obj in enumerate(objectives):
        if obj == 'min':
            # For "min", lower is better.
            if a[i] > b[i]:
                return False
            elif a[i] < b[i]:
                strictly_better = True
        elif obj == 'max':
            # For "max", higher is better.
            if a[i] < b[i]:
                return False
            elif a[i] > b[i]:
                strictly_better = True
        else:
            raise ValueError("Objective must be either 'min' or 'max'.")
    return strictly_better


def compute_pareto(points, objectives):
    """
    Computes the Pareto frontier from a list of points.
    - points: list of tuples, where each tuple contains metric values.
    - objectives: list of strings ('min' or 'max') corresponding to each metric.
    Returns a list of points that are on the Pareto frontier.
    """
    pareto_points_idx = []
    for i, (point) in enumerate(points):
        if not any(dominates(other, point, objectives) for other in points if other != point):
            pareto_points_idx.append(i)
    return pareto_points_idx


def plot_pareto_frontier(index_df: pd.DataFrame, query_df: pd.DataFrame,
                         compare_columns: list[Tuple[Tuple[str, str, str], Tuple[str, str, str], bool]],
                         name_prefix):
    # multiple set of graph, identified by (index_data, query_data, query_l)
    # multiple set of points on one graph, identified by (isolate_alpha, index_r, index_l)
    # each point is identified by (index_alpha)
    all_index_l = sorted(query_df["index_l"].unique())
    all_alpha = sorted(query_df["index_alpha"].unique())
    all_index_r = sorted(query_df["index_r"].unique())
    all_index_dataset = sorted(query_df["index_data"].unique())
    all_isolate_alpha = sorted(query_df["build_isolate_alpha"].unique())
    all_knn_index = sorted(query_df["build_knn_index"].unique())
    all_query_l = sorted(query_df["query_l"].unique())
    all_query_dataset = sorted(query_df["query_data"].unique())
    n_row = len(all_index_dataset) * len(all_query_dataset) * len(all_query_l)
    n_col = len(compare_columns)
    fig, axs = plt.subplots(n_row, n_col, figsize=(n_col * 7, n_row * 7), squeeze=False)
    for plot_row, (index_dataset, query_dataset, query_l) in enumerate(
            itertools.product(all_index_dataset, all_query_dataset, all_query_l)):
        for plot_col, ((col1_name, col1_objective, col1_src),
                       (col2_name, col2_objective, col2_src),
                       plot_pareto) in enumerate(compare_columns):
            for line_i, (isolate_alpha, index_l, index_r, knn_index) in enumerate(
                itertools.product(all_isolate_alpha, all_index_l, all_index_r, all_knn_index)):
                line_name = f"isolate_alpha={isolate_alpha}, L_index={index_l}, R_index={index_r}, knn_index={knn_index}"
                line_color = PLOT_COLORS_2[line_i % len(PLOT_COLORS_2)]
                val_data, configs = [], []
                for index_alpha, in itertools.product(all_alpha):
                    vals = []
                    for col_name, col_src in [(col1_name, col1_src), (col2_name, col2_src)]:
                        if col_src == "index":
                            matched_col_data = index_df[(index_df["index_l"] == index_l) &
                                                        (index_df["index_r"] == index_r) &
                                                        (index_df["index_alpha"] == index_alpha) &
                                                        (index_df["index_data"] == index_dataset) &
                                                        (index_df["build_knn_index"] == knn_index) &
                                                        (index_df["build_isolate_alpha"] == isolate_alpha)][col_name]
                        else:
                            matched_col_data = query_df[(query_df["index_l"] == index_l) &
                                                        (query_df["index_r"] == index_r) &
                                                        (query_df["index_alpha"] == index_alpha) &
                                                        (query_df["index_data"] == index_dataset) &
                                                        (query_df["query_l"] == query_l) &
                                                        (query_df["query_data"] == query_dataset) &
                                                        (query_df["build_knn_index"] == knn_index) &
                                                        (query_df["build_isolate_alpha"] == isolate_alpha)][col_name]
                        val = matched_col_data.mean()
                        vals.append(val)
                    val_data.append(vals)
                    configs.append((f"{index_alpha:.2f}",))
                # compute the pareto frontier
                pareto_points_idx = compute_pareto(val_data, objectives=(col1_objective, col2_objective))
                if plot_pareto:
                    val_data = [(*val_data[idx], configs[idx]) for idx in pareto_points_idx]
                else:
                    val_data = [(*vals, config) for vals, config in zip(val_data, configs)]
                val_sorted = sorted(val_data, key=lambda v: (v[0], v[1]))
                x_data = [val[0] for val in val_sorted]
                y_data = [val[1] for val in val_sorted]
                axs[plot_row][plot_col].plot(x_data, y_data, "^-", label=line_name, color=line_color)
                for x, y, (label,) in val_sorted:
                    axs[plot_row][plot_col].annotate(label, (x, y), color=line_color, textcoords="offset points",
                                                     xytext=(0, 0), fontsize=10)
            axs[plot_row][plot_col].set_xlabel(col1_name)
            axs[plot_row][plot_col].set_ylabel(col2_name)
            axs[plot_row][plot_col].set_title(f"L_query={query_l}")
            axs[plot_row][plot_col].grid(True, linestyle='--', alpha=0.7)

    # fig.suptitle(fr"$\alpha$-sensitivity benchmark")
    handles, labels = axs[0][0].get_legend_handles_labels()
    legend_width_frac = 0.07
    legend_ax = fig.add_axes((0.0, 0.0, legend_width_frac, 1))
    legend_ax.axis('off')
    legend_ax.legend(handles, labels, loc='right', title="Legend")
    fig.tight_layout(rect=(legend_width_frac, 0, 1, 1))
    plt.savefig("{}_pareto_frontier_benchmark.png".format(name_prefix.replace(" ", "_").lower()), bbox_inches="tight")


def run_benchmarks(param, **kwargs):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    state = run_serialized_benchmarks(project_root, param, **kwargs)
    return save_state(state, project_root)


def consolidate_data(param, *state_files):
    states = [load_state(state_file) for state_file in state_files]
    if len(states) == 0:
        return None, None
    index_df = combine_states_to_df(*[state["index_raw_data"] for state in states],
                                    params=param, params_to_key=index_param_to_key,
                                    flatten_param=flatten_index_param)
    print(f"Loaded {index_df.shape[0]} data points for index.")
    query_df = combine_states_to_df(*[state["query_raw_data"] for state in states],
                                    params=param, params_to_key=query_param_to_key,
                                    flatten_param=flatten_query_param, subkeys=["l"], subkeys_as=["query_l"])
    print(f"Loaded {query_df.shape[0]} data points for query.")
    return index_df, query_df


def plot_benchmarks(index_df, query_df, name_prefix):
    # index construction benchmarks
    index_plot_columns = ["indexing_time_seconds", "index_size_byte", "index_data_size_byte", "max_degree",
                          "avg_degree", "min_degree", "count_deg_lt_2"]
    plot_construction_benchmark(index_df, index_plot_columns, name_prefix=name_prefix)

    # query benchmarks
    query_plot_columns = ["QPS", "average_distance_compare", "mean_latency", "p99_latency", "recall"]
    plot_query_benchmark(query_df, query_plot_columns, name_prefix=name_prefix)

    # column comparisons, plotted on pareto frontier
    # column_name, column_objective (min/max), column_sources (query/index)
    recall_col = ("recall", "max", "query")
    QPS_col = ("QPS", "max", "query")
    p99_latency_col = ("p99_latency", "min", "query")
    mean_latency_col = ("mean_latency", "min", "query")
    average_distance_compare_col = ("average_distance_compare", "min", "query")
    indexing_time_col = ("indexing_time_seconds", "min", "index")
    index_size_byte_col = ("index_size_byte", "min", "index")
    plot_pareto_frontier(index_df, query_df, compare_columns=[
        # column_x, column_y, plot_pareto
        (recall_col, QPS_col, True),
        (recall_col, average_distance_compare_col, True),
        # (recall_col, mean_latency_col, True),
        (QPS_col, index_size_byte_col, False),
    ], name_prefix=name_prefix)
    print("Plotted", name_prefix)


alpha_one_to_one_point_six = [1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6]
alpha_one_to_two = [1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, 1.55, 1.6, 1.65, 1.7,
                    1.75, 1.8, 1.85, 1.9, 1.95, 2.0]
alpha_1_2 = [1.2]

if __name__ == '__main__':
    dataset_bench = "bigann_50M.bbin"
    grouped_param = group_benchmark_params([
        # ============ Sift benchmarks ============
        # {  # building parameters
        #     "build_isolate_alpha": ["On", "Off"],
        #     "build_tracking": False,
        #     # indexing parameters
        #     "index_data": "sift_base.fbin",
        #     "index_l": [25, 50, 75],
        #     "index_r": [64],
        #     "index_alpha": alpha_one_to_two,
        #     # query parameters
        #     "query_num_runs": 5,
        #     "query_data": "sift_query.fbin",
        #     "query_l": [10, 50],
        #     "query_k": [10], },
        {  # building parameters
            "build_isolate_alpha": ["Off"],
            "build_knn_index": [0],
            "build_tracking": False,
            # indexing parameters
            "index_data": dataset_bench,
            "index_l": [50],
            "index_r": [64],
            "index_alpha": alpha_1_2,
            # query parameters
            "query_num_runs": 5,
            "query_data": "bigann_query.bbin",
            "query_l": [50],
            "query_k": [10], },
        # ============ Random benchmarks ============
        # {  # building parameters
        #     "build_isolate_alpha": ["On", "Off"],
        #     "build_tracking": False,
        #     # indexing parameters
        #     "index_data": "rand_128_1m.fbin",
        #     "index_l": [25, 50, 75],
        #     "index_r": [64],
        #     "index_alpha": alpha_one_to_two,
        #     # query parameters
        #     "query_num_runs": 5,
        #     "query_data": "rand_128_10k.fbin",
        #     "query_l": [10, 50],
        #     "query_k": 10, },
        # {  # building parameters
        #     "build_isolate_alpha": ["On", "Off"],
        #     "build_tracking": False,
        #     # indexing parameters
        #     "index_data": "rand_128_1m.fbin",
        #     "index_l": [25, 50, 75],
        #     "index_r": [64],
        #     "index_alpha": alpha_one_to_two,
        #     # query parameters
        #     "query_num_runs": 5,
        #     "query_data": "rand_128_10k.fbin",
        #     "query_l": [50, 100],
        #     "query_k": 50, },
    ])
    run_benchmarks(
        grouped_param,
        dry_run=False,
        last_state=None,
        use_existing_index=False,
        delete_index_after_query=False
    )
    if False:
        index_df, query_df = consolidate_data(
            grouped_param,
            "state_20250401_152636.json",
        )
        print(query_df)
        if index_df is not None and query_df is not None:
            plot_benchmarks(
                index_df[index_df["index_data"] == dataset_bench],
                query_df[(query_df["index_data"] == dataset_bench) &
                         (query_df["query_k"] == 10)],
                name_prefix="sift1m_k10"
            )
            # plot_benchmarks(
            #     index_df[index_df["index_data"] == "rand_128_1m.fbin"],
            #     query_df[(query_df["index_data"] == "rand_128_1m.fbin") &
            #              (query_df["query_k"] == 10)],
            #     name_prefix="rand1m_k10"
            # )
            # plot_benchmarks(
            #     index_df[index_df["index_data"] == "rand_128_1m.fbin"],
            #     query_df[(query_df["index_data"] == "rand_128_1m.fbin") &
            #              (query_df["query_k"] == 50)],
            #     name_prefix="rand1m_k50"
            # )
        else:
            print("None plotted.")
