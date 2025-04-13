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
import matplotlib

from lib.util import create_build, run_and_parse_output, download_sift

MULTI_THREADED_READ_BENCHMARKS = False
MULTI_THREADED_WRITE_BENCHMARKS = False


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
        build_isolate_alpha_direct = get_as_list(param, "build_isolate_alpha_direct", ["Off"])
        build_two_pass_indexing = get_as_list(param, "build_two_pass_indexing", ["Off"])
        build_two_pass_index_scaled_l_first_pass = get_as_list(param, "build_two_pass_index_scaled_l_first_pass",
                                                               ["Off"])
        build_two_pass_index_scaled_l_last_pass = get_as_list(param, "build_two_pass_index_scaled_l_last_pass", ["Off"])
        build_two_pass_index_sampled_visit_order = get_as_list(param, "build_two_pass_index_sampled_visit_order",
                                                               ["Off"])
        build_sorted_visit_order = get_as_list(param, "build_sorted_visit_order", ["Off"])
        build_tracking = get_as_list(param, "build_tracking", [False])
        for (build_isolate_alpha_,
             build_isolate_alpha_direct_,
             build_two_pass_indexing_,
             build_two_pass_index_scaled_l_first_pass_,
             build_two_pass_index_scaled_l_last_pass_,
             build_two_pass_index_sampled_visit_order_,
             build_sorted_visit_order_,
             build_tracking_) in itertools.product(
            build_isolate_alpha,
            build_isolate_alpha_direct,
            build_two_pass_indexing,
            build_two_pass_index_scaled_l_first_pass,
            build_two_pass_index_scaled_l_last_pass,
            build_two_pass_index_sampled_visit_order,
            build_sorted_visit_order,
            build_tracking):
            build_param = {
                "isolate_alpha": build_isolate_alpha_,
                "isolate_alpha_direct": build_isolate_alpha_direct_,
                "two_pass_indexing": build_two_pass_indexing_,
                "two_pass_index_scaled_l_first_pass": build_two_pass_index_scaled_l_first_pass_,
                "two_pass_index_scaled_l_last_pass": build_two_pass_index_scaled_l_last_pass_,
                "two_pass_index_sampled_visit_order": build_two_pass_index_sampled_visit_order_,
                "sorted_visit_order": build_sorted_visit_order_,
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


def run_build_step(state, temp_state, trans_state,
                   isolate_alpha,
                   isolate_alpha_direct,
                   two_pass_indexing,
                   build_two_pass_index_scaled_l_first_pass_,
                   build_two_pass_index_scaled_l_last_pass_,
                   two_pass_index_sampled_visit_order,
                   sorted_visit_order,
                   tracking):
    project_root = temp_state["project_root"]
    build_key = trans_state["build_key"]
    build_path = os.path.join(project_root, "build", build_key)
    data_path = os.path.join(project_root, "build", "data")
    if build_path not in temp_state["recent_builds"]:
        build_path = create_build(project_root, build_subdir=build_key,
                                  tracking=tracking, type="Release",
                                  ISOLATE_ALPHA=isolate_alpha,
                                  ISOLATE_ALPHA_DIRECT=isolate_alpha_direct,
                                  TWO_PASS_INDEXING=two_pass_indexing,
                                  TWO_PASS_INDEX_SCALED_L_FIRST_PASS=build_two_pass_index_scaled_l_first_pass_,
                                  TWO_PASS_INDEX_SCALED_L_LAST_PASS=build_two_pass_index_scaled_l_last_pass_,
                                  TWO_PASS_INDEX_SAMPLED_VISIT_ORDER=two_pass_index_sampled_visit_order,
                                  SORTED_VISIT_ORDER=sorted_visit_order)
        temp_state["recent_builds"].add(build_path)
    sift_data_path = os.path.join(data_path, "sift")
    rand_data_path = os.path.join(data_path, "rand")
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

    if process.returncode != 0:
        print("Error running command:\n", cmd_dump)
        print("Log at: ", stderr_file)
        raise Exception("failed benchmark, run_serialized_command")

    return stdout_file, stderr_file


def extract_index_step_data(log_file_path, diskann_index_path):
    # Default values for metrics
    sort_visit_order_time = 0.0
    first_pass_time = None
    second_pass_time = None
    total_link_time = 0.0
    total_indexing_time = None
    total_save_time = None
    max_degree = None
    avg_degree = None
    min_degree = None
    count_deg_lt_2 = None

    # Open and parse the log file line by line.
    with open(log_file_path, "r") as file:
        for line in file:
            # Extract Sort Visit Order time (if present)
            match_sort = re.search(r"Sort Visit Order time:\s*([\d.]+)s", line)
            if match_sort:
                sort_visit_order_time = float(match_sort.group(1))

            # Extract Pass 0 time
            match_pass0 = re.search(r"Pass 0\s*:\s*([\d.]+)s", line)
            if match_pass0 and first_pass_time is None:
                first_pass_time = float(match_pass0.group(1))

            # Extract Pass 1 time
            match_pass1 = re.search(r"Pass 1\s*:\s*([\d.]+)s", line)
            if match_pass1 and second_pass_time is None:
                second_pass_time = float(match_pass1.group(1))

            # Extract Link time from "Link time:" line
            match_link = re.search(r"Link time:\s*([\d.]+)s", line)
            if match_link:
                total_link_time = float(match_link.group(1))

            # Extract Indexing time
            if total_indexing_time is None:
                match_indexing = re.search(r"Indexing time:\s*([\d.]+)", line)
                if match_indexing:
                    total_indexing_time = float(match_indexing.group(1))

            # Extract Save time
            if total_save_time is None:
                match_save = re.search(r"Time taken for save:\s*([\d.]+)s", line)
                if match_save:
                    total_save_time = float(match_save.group(1))

            # Extract index built degree metrics
            match_build = re.search(
                r"Index built with degree:\s*max:(\d+)\s+avg:([\d.]+)\s+min:(\d+)\s+count\(deg<2\):(\d+)",
                line
            )
            if match_build:
                max_degree = int(match_build.group(1))
                avg_degree = float(match_build.group(2))
                min_degree = int(match_build.group(3))
                count_deg_lt_2 = int(match_build.group(4))

    # Set default values when metrics are missing
    if total_indexing_time is None:
        total_indexing_time = -1.0
    if total_save_time is None:
        total_save_time = 0.0
    if first_pass_time is None:
        first_pass_time = 0.0
    if second_pass_time is None:
        last_pass_time = first_pass_time
        first_pass_time = 0.0
    else:
        last_pass_time = second_pass_time

    # If indexing was successful (i.e. valid indexing time), determine file sizes.
    index_size_byte = 0
    index_data_size_byte = 0
    if total_indexing_time != -1.0:
        if os.path.exists(diskann_index_path):
            index_size_byte = os.path.getsize(diskann_index_path)
        if os.path.exists(diskann_index_path + ".data"):
            index_data_size_byte = os.path.getsize(diskann_index_path + ".data")

    # Return results in a dictionary
    return {
        "sort_visit_order_time": sort_visit_order_time,
        "first_pass_link_time": first_pass_time,
        "last_pass_link_time": last_pass_time,
        "total_link_time": total_link_time,
        "max_degree": max_degree,
        "avg_degree": avg_degree,
        "min_degree": min_degree,
        "count_deg_lt_2": count_deg_lt_2,
        "total_indexing_time": total_indexing_time,
        "total_save_time": total_save_time,
        "index_size_byte": index_size_byte,
        "index_data_size_byte": index_data_size_byte,
    }


def index_benchmark_exist(state, build_key, index_key):
    if "index_raw_data" not in state or f"{build_key}_{index_key}" not in state["index_raw_data"]:
        return False
    return True


def run_index_step(state, temp_state, trans_state, data, l, r, alpha, saturate_graph, use_existing_index=True):
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
        data_path = os.path.join(data_path_prefix, data)
        cmd = [
            build_memory_index,
            "--data_type", "float",
            "--dist_fn", "l2",
            "--data_path", data_path,
            "--index_path_prefix", diskann_index_path,
            "-R", str(r),
            "-L", str(l),
            "--alpha", str(alpha),
            "--num_threads", ("0" if MULTI_THREADED_WRITE_BENCHMARKS else "1"),
            "--tracking_addr", f"tcp://localhost:{DEFAULT_TRACKING_PORT}",
            "--saturate_graph" if saturate_graph else "",
        ]

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


def get_ground_truth(state, temp_state, trans_state, data_file: str, query_data_file: str, k=100):
    data_path_prefix = trans_state["sift_data_path"] if data_file.startswith("sift") else trans_state["rand_data_path"]
    ground_truth_file = os.path.join(data_path_prefix, "d{}_q{}.gt_{}".format(data_file.replace(".fbin", ""),
                                                                              query_data_file.replace(".fbin", ""),
                                                                              k))
    if not os.path.exists(ground_truth_file):
        compute_groundtruth = trans_state["compute_groundtruth"]
        cmd = [
            compute_groundtruth,
            "--data_type", "float",
            "--dist_fn", "l2",
            "--base_file", os.path.join(data_path_prefix, data_file),
            "--query_file", os.path.join(data_path_prefix, query_data_file),
            "--gt_file", ground_truth_file,
            "--K", str(k)
        ]
        run_serialized_command(cmd, None)
    return ground_truth_file


def run_query_step(state, temp_state, trans_state, query_key, num_runs, data, k, l):
    project_root: str = temp_state["project_root"]
    build_key = trans_state["build_key"]
    index_key = trans_state["index_key"]
    query_base_path: str = os.path.join(project_root, "build", build_key, index_key, query_key)
    os.makedirs(query_base_path, exist_ok=True)

    search_memory_index = trans_state["search_memory_index"]
    diskann_index_path = trans_state["diskann_index_path"]
    indexed_data_file = trans_state["indexed_data_file"]
    ground_truth_file = get_ground_truth(state, temp_state, trans_state, indexed_data_file, data)
    data_path_prefix = trans_state["sift_data_path"] if data.startswith("sift") else trans_state["rand_data_path"]
    data_path = os.path.join(data_path_prefix, data)

    run_result_bin_files = []
    run_stdouts = []
    for run_i in range(num_runs):
        query_run_path = os.path.join(query_base_path, f"run_{run_i}")
        result_path = os.path.join(query_run_path, "result")
        os.makedirs(query_run_path, exist_ok=True)
        cmd = [
            search_memory_index,
            "--data_type", "float",
            "--dist_fn", "l2",
            "--index_path_prefix", diskann_index_path,
            "--query_file", data_path,
            "--gt_file", ground_truth_file,
            "-K", str(k),
            "-L", *[str(l_) for l_ in l],
            "--result_path", result_path,
            "--num_threads", "0" if MULTI_THREADED_READ_BENCHMARKS else "1",
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
        if MULTI_THREADED_WRITE_BENCHMARKS:
            print("Write benchmarks will be multi threaded...")
        if MULTI_THREADED_READ_BENCHMARKS:
            print("Read benchmarks will be multi threaded...")

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
                run_index_step(state, temp_state, trans_state, use_existing_index=use_existing_index, **index_param)
            total_elapsed_time += t.time_elapsed()

            # query step
            for query_i, (query_param, query_key) in enumerate(zip(query_params, query_keys)):
                if not query_benchmark_exist(state, build_key, index_key, query_key):
                    with Timed(binded_print(f" {i + 1:>3}/{num_benchmarks} [query_{query_i}]")) as t:
                        run_query_step(state, temp_state, trans_state, query_key, **query_param)
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
            if k not in key_to_param:
                print(f"Skipping: {k}")
                continue
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


def combine_dfs(dfs: List[pd.DataFrame]):
    columns = dict()
    for df in dfs:
        for col in df.columns:
            if col not in columns:
                columns[col] = 0
            columns[col] += 1
    common_columns = list(k for k, v in columns.items() if v == len(dfs))
    print("Joining on", common_columns)
    return pd.merge(dfs[0], dfs[1], on=common_columns, how="inner")


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


def plot_lines(df: pd.DataFrame,
               column_configs: list[
                   Tuple[Tuple[str, str | None, str | None], Tuple[str, str | None, str | None], bool, bool]],
               line_by,
               point_by,
               name_prefix,
               graph_by=None,
               row_by=None,
               get_graph_name: Callable = None,
               get_row_name: Callable = None,
               get_line_name: Callable = None,
               get_point_label: Callable = None,
               master_name=None,
               legend_style: str | None = "external",  # "external" or matplotlib legend styles
               color_dict=("blue",),
               line_style_dict=("^-",),
               fig_size_scale=(5, 5)):
    def plot_columns(row_axs_, row_data_, row_name_):
        line_groups = row_data_.groupby(line_by)
        for plot_col, ((col1_attr, col1_alt_name, col1_objective), (col2_attr, col2_alt_name, col2_objective),
                       plot_pareto, plot_bounds) in enumerate(column_configs):
            col1_name = col1_alt_name if col1_alt_name is not None else col1_attr
            col2_name = col2_alt_name if col2_alt_name is not None else col2_attr
            for line_i, (line_attr, line_data) in enumerate(line_groups):
                line_attr_dict = {k: v for k, v in zip(line_by, line_attr)}
                line_name = get_line_name(line_attr) if get_line_name is not None else generate_key(line_attr_dict)
                line_color = color_dict[line_i % len(color_dict)]
                line_line_style = line_style_dict[line_i % len(line_style_dict)]
                point_groups = line_data.groupby(point_by)
                point_values = []
                for point_attr, point_data in point_groups:
                    col1_val = point_data[col1_attr]
                    col2_val = point_data[col2_attr]
                    point_label = get_point_label(point_attr) if get_point_label is not None else None
                    point_values.append(
                        (col1_val.mean(), col2_val.mean(), col2_val.min(), col2_val.max(), point_label))
                if plot_pareto:
                    pareto_points_idx = compute_pareto([(v[0], v[1]) for v in point_values],
                                                       objectives=(col1_objective, col2_objective))
                    point_values = [point_values[idx] for idx in pareto_points_idx]
                point_values.sort(key=lambda v: (v[0], v[1]))
                x_data = [val[0] for val in point_values]
                y_data = [val[1] for val in point_values]
                if plot_bounds:
                    y_min = [val[2] for val in point_values]
                    y_max = [val[3] for val in point_values]
                    row_axs_[plot_col].fill_between(x_data, y_min, y_max, color=line_color, alpha=0.25)
                row_axs_[plot_col].plot(x_data, y_data, line_line_style, label=line_name, color=line_color)
                for x, y, _, _, label in point_values:
                    if label is None: continue
                    row_axs_[plot_col].annotate(label, (x, y), color=line_color, textcoords="offset points",
                                                xytext=(0, 0), fontsize=10)
            row_axs_[plot_col].set_xlabel(col1_name)
            row_axs_[plot_col].set_ylabel(col2_name)
            row_axs_[plot_col].set_title(row_name_)
            row_axs_[plot_col].grid(True, linestyle='--', alpha=0.7)
            if legend_style is not None and legend_style != "external":
                row_axs_[plot_col].legend(loc=legend_style)

    def plot_rows(graph_data_, graph_name_):
        if row_by is None:
            n_col = len(column_configs)
            fig, axs = plt.subplots(1, n_col, figsize=(n_col * fig_size_scale[0], fig_size_scale[1]))
            plot_columns(axs, graph_data_, master_name)
            handles, labels = axs[0].get_legend_handles_labels()
        else:
            row_groups = graph_data_.groupby(row_by)
            n_row = len(row_groups)
            n_col = len(column_configs)
            fig, axs = plt.subplots(n_row, n_col, figsize=(n_col * fig_size_scale[0], n_row * fig_size_scale[1]),
                                    squeeze=False)
            for plot_row, (row_attr, row_data) in enumerate(row_groups):
                row_attr_dict = {k: v for k, v in zip(row_by, row_attr)}
                row_name = get_row_name(row_attr) if get_row_name is not None else pprint.pformat(
                    row_attr_dict)
                plot_columns(axs[plot_row], row_data, row_name)
            handles, labels = axs[0][0].get_legend_handles_labels()
        if legend_style == "external":
            legend_width_frac = 0.07
            legend_ax = fig.add_axes((0.0, 0.0, legend_width_frac, 1))
            legend_ax.axis('off')
            legend_ax.legend(handles, labels, loc='right', title="Legend")
            fig.tight_layout(rect=(legend_width_frac, 0, 1, 1))
        else:
            fig.tight_layout()
        plt.savefig("{}{}.png".format(name_prefix.replace(" ", "_").lower(),
                                      f"_{graph_name_}" if graph_name_ is not None else ""),
                    bbox_inches="tight")

    if graph_by is None:
        plot_rows(df, master_name)
    else:
        graph_groups = df.groupby(graph_by)
        for graph_attr, graph_data in graph_groups:
            graph_attr_dict = {k: v for k, v in zip(graph_by, graph_attr)}
            graph_name = get_graph_name(graph_attr) if get_row_name is not None else generate_key(graph_attr_dict)
            plot_rows(graph_data, graph_name)


def run_benchmarks(param, **kwargs):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    state = run_serialized_benchmarks(project_root, param, **kwargs)
    return save_state(state, project_root)


def consolidate_data(param, *state_files):
    states = [load_state(state_file) for state_file in state_files]
    if len(states) == 0:
        return None, None, None
    index_df = combine_states_to_df(*[state["index_raw_data"] for state in states],
                                    params=param, params_to_key=index_param_to_key,
                                    flatten_param=flatten_index_param)
    print(f"Loaded {index_df.shape[0]} data points for index.")
    query_df = combine_states_to_df(*[state["query_raw_data"] for state in states],
                                    params=param, params_to_key=query_param_to_key,
                                    flatten_param=flatten_query_param, subkeys=["l"], subkeys_as=["query_l"])
    print(f"Loaded {query_df.shape[0]} data points for query.")
    combined_df = combine_dfs([query_df, index_df])
    print(f"Combined {combined_df.shape} data points.")
    return index_df, query_df, combined_df


def get_categorical_colors(num_category, num_sub_category, cmap="tab10", continuous=False):
    # https://stackoverflow.com/a/47232942
    if num_category > plt.get_cmap(cmap).N:
        raise ValueError("Too many categories for colormap.")
    if continuous:
        ccolors = plt.get_cmap(cmap)(np.linspace(0, 1, num_category))
    else:
        ccolors = plt.get_cmap(cmap)(np.arange(num_category, dtype=int))
    cols = np.zeros((num_category * num_sub_category, 3))
    for i, c in enumerate(ccolors):
        chsv = matplotlib.colors.rgb_to_hsv(c[:3])
        arhsv = np.tile(chsv, num_sub_category).reshape(num_sub_category, 3)
        arhsv[:, 1] = np.linspace(chsv[1], 0.25, num_sub_category)
        arhsv[:, 2] = np.linspace(chsv[2], 1, num_sub_category)
        rgb = matplotlib.colors.hsv_to_rgb(arhsv)
        cols[i * num_sub_category:(i + 1) * num_sub_category, :] = rgb
    return cols


def plot_benchmarks(df):
    # column_attr, column_name, column_objective (min/max)

    font = {'weight': 'bold', 'size': 12}
    matplotlib.rc('font', **font)

    # query performance columns
    recall_col = ("recall", "Recall@10 (%)", "max")
    qps_col = ("QPS", "Query Per Seconds", "max")
    p99_latency_col = ("p99_latency", "Tail Latency P99 (micros)", "min")
    mean_latency_col = ("mean_latency", "Average Latency (micros)", "min")
    average_distance_compare_col = ("average_distance_compare", "Average Distance Compare", "min")

    # index performance columns
    total_indexing_time_col = ("total_indexing_time", "Construction Time (second)", "min")
    first_pass_link_time_col = ("first_pass_link_time", "First Pass Time (second)", "min")
    last_pass_link_time_col = ("last_pass_link_time", "Second Pass Time (second)", "min")
    index_size_byte_col = ("index_size_byte", "Index Size (byte)", "min")
    avg_degree_col = ("avg_degree", "Average Degree", "min")
    max_degree_col = ("max_degree", "Max Degree", "min")

    # index build columns
    alpha_col = ("index_alpha", "Alpha", None)
    index_l_col = ("index_l", "Lc", None)

    index_plot_columns = ["sort_visit_order_time", "first_pass_link_time", "last_pass_link_time", "total_link_time",
                          "total_indexing_time", "total_save_time", "index_size_byte", "index_data_size_byte",
                          "max_degree", "avg_degree", "min_degree", "count_deg_lt_2"]
    query_plot_columns = ["QPS", "average_distance_compare", "mean_latency", "p99_latency", "recall"]

    # varying alpha
    plot_lines(
        df[(df["build_two_pass_indexing"] == "Off") & (df["build_sorted_visit_order"] == "Off")],
        row_by=["query_k", "query_l", "index_r", "index_l"],
        get_row_name=lambda v: f"SIFT1M, K={int(v[0])}, Lq={int(v[1])}, R={int(v[2])}, Lc={int(v[3])}",
        column_configs=[
            (alpha_col, recall_col, False, True),
            (alpha_col, qps_col, False, True),
            (alpha_col, average_distance_compare_col, False, True),
            (alpha_col, index_size_byte_col, False, True),
            (alpha_col, avg_degree_col, False, True),
        ],
        line_by=["build_isolate_alpha", "build_isolate_alpha_direct"],
        get_line_name=lambda v: "{}".format(("isolate"
                                             if v[0] == "On" else
                                             ("direct"
                                              if v[1] == "On" else "diskann"))),
        point_by=["index_alpha"],
        name_prefix="expr1_vary_alpha",
        color_dict=get_categorical_colors(3, 1),
        line_style_dict=["-o", "-s", "-^"],
        legend_style="best",
    )
    plot_lines(
        df[(df["build_two_pass_indexing"] == "Off") & (df["build_sorted_visit_order"] == "Off")
            # & (df["build_isolate_alpha"] == "Off")
            # & (df["build_isolate_alpha_direct"] == "Off")
            # & ((df["index_alpha"] == 1.0) | (df["index_alpha"] == 1.2) | (df["index_alpha"] == 1.6) | (df["index_alpha"] == 1.8))
           ],
        row_by=["query_k", "query_l", "index_r", "index_l"],
        get_row_name=lambda v: f"SIFT1M, K={int(v[0])}, Lq={int(v[1])}, R={int(v[2])}, Lc={int(v[3])}",
        column_configs=[
            (recall_col, average_distance_compare_col, True, False),
            (recall_col, qps_col, True, False),
        ],
        line_by=["build_isolate_alpha", "build_isolate_alpha_direct"],
        get_line_name=lambda v: "{}".format(("isolate"
                                             if v[0] == "On" else
                                             ("direct"
                                              if v[1] == "On" else "diskann"))),
        point_by=["index_alpha"],
        get_point_label=lambda v: f"$\\alpha={v[0]:.1f}$",
        # point_by=["build_isolate_alpha", "build_isolate_alpha_direct"],
        # get_point_label=lambda v: "{}".format(("isolate"
        #                                        if v[0] == "On" else
        #                                        ("direct"
        #                                         if v[1] == "On" else "diskann"))),
        # line_by=["index_alpha"],
        # get_line_name=lambda v: f"$\\alpha$={v[0]}",
        name_prefix="expr1_vary_alpha_compare",
        color_dict=get_categorical_colors(10, 1),
        line_style_dict=["-o", "-s", "-^"],
        legend_style="best",
    )

    # two pass
    plot_lines(
        df[(df["index_l"] != 50) &
           (df["index_alpha"] == 1.2) & (df["build_sorted_visit_order"] == "Off") &
           (df["build_isolate_alpha"] == "Off") & (df["build_isolate_alpha_direct"] == "Off")],
        row_by=["query_k", "query_l", "index_r"],
        get_row_name=lambda v: f"SIFT1M, K={int(v[0])}, Lq={int(v[1])}, R={int(v[2])}",
        column_configs=[
            (recall_col, average_distance_compare_col, False, False),
            (recall_col, qps_col, False, False),
            (recall_col, index_size_byte_col, False, False),
            (recall_col, avg_degree_col, False, False),
            (average_distance_compare_col, index_size_byte_col, False, False),
            (recall_col, total_indexing_time_col, False, False),
            (average_distance_compare_col, total_indexing_time_col, False, False),
        ],
        line_by=["build_two_pass_indexing",
                 "build_two_pass_index_scaled_l",
                 "build_two_pass_index_sampled_visit_order"],
        get_line_name=lambda v: (f"diskann"
                                 if v[0] == "Off" else
                                 ((f"full + full" if v[2] == "Off" else f"sampled + full")
                                  if v[1] == "Off" else
                                  (f"2Lc + full" if v[2] == "Off" else f"sampled 2Lc + full"))),
        point_by=["index_l"],
        get_point_label=lambda v: f"Lc={v[0]}",
        name_prefix="expr2_two_pass",
        color_dict=get_categorical_colors(5, 1),
        line_style_dict=["-o", "-s", "--s", "-^", "--^"],
        legend_style="best",
    )

    plot_lines(
        df[(df["index_l"] != 50) &
           (df["index_alpha"] == 1.2) & (df["build_sorted_visit_order"] == "Off") &
           (df["build_isolate_alpha"] == "Off") & (df["build_isolate_alpha_direct"] == "Off")],
        row_by=["query_k", "query_l", "index_r"],
        get_row_name=lambda v: f"SIFT1M, K={int(v[0])}, Lq={int(v[1])}, R={int(v[2])}",
        column_configs=[
            (index_l_col, avg_degree_col, False, False),
            (index_l_col, total_indexing_time_col, False, False),
            (index_l_col, index_size_byte_col, False, False),
            (index_l_col, first_pass_link_time_col, False, False),
            (index_l_col, last_pass_link_time_col, False, False),
        ],
        line_by=["build_two_pass_indexing",
                 "build_two_pass_index_scaled_l",
                 "build_two_pass_index_sampled_visit_order"],
        get_line_name=lambda v: (f"diskann"
                                 if v[0] == "Off" else
                                 ((f"full + full" if v[2] == "Off" else f"sampled + full")
                                  if v[1] == "Off" else
                                  (f"2Lc + full" if v[2] == "Off" else f"sampled 2Lc + full"))),
        point_by=["index_l"],
        get_point_label=None,
        name_prefix="expr2_two_pass_index_l",
        color_dict=get_categorical_colors(5, 1),
        line_style_dict=["-o", "-s", "--s", "-^", "--^"],
        legend_style="best",
    )

    # sorted order
    plot_lines(
        df[(df["index_l"] != 50) &
           (df["index_alpha"] == 1.2) &
           (df["build_isolate_alpha"] == "Off") & (df["build_isolate_alpha_direct"] == "Off")],
        row_by=["query_k", "query_l", "index_r"],
        get_row_name=lambda v: f"SIFT1M, K={int(v[0])}, Lq={int(v[1])}, R={int(v[2])}",
        column_configs=[
            (recall_col, average_distance_compare_col, False, False),
            (recall_col, index_size_byte_col, False, False),
            (average_distance_compare_col, index_size_byte_col, False, False),
            (recall_col, total_indexing_time_col, False, False),
            (average_distance_compare_col, total_indexing_time_col, False, False),
        ],
        line_by=["build_sorted_visit_order",
                 "build_two_pass_indexing",
                 "build_two_pass_index_scaled_l",
                 "build_two_pass_index_sampled_visit_order"],
        get_line_name=lambda v: ((f"one pass"
                                  if v[1] == "Off" else
                                  ((f"full + full" if v[3] == "Off" else f"sampled + full")
                                   if v[2] == "Off" else
                                   (f"2Lc + full" if v[3] == "Off" else f"sampled 2Lc + full")))
                                 if v[0] == "Off" else
                                 (f"one pass (sorted)"
                                  if v[1] == "Off" else
                                  ((f"full + full (sorted)" if v[3] == "Off" else f"sampled + full (sorted)")
                                   if v[2] == "Off" else
                                   (f"2Lc + full (sorted)" if v[3] == "Off" else
                                    f"sampled 2Lc 1 + full 2 (sorted)")))),
        point_by=["index_l"],
        get_point_label=lambda v: f"Lc={v[0]}",
        name_prefix="expr3_sorted_order",
        color_dict=get_categorical_colors(2, 5),
        legend_style="best",
    )


alpha_one_to_one_point_six = [1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6]
alpha_one_to_two = [1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, 1.55, 1.6, 1.65, 1.7,
                    1.75, 1.8, 1.85, 1.9, 1.95, 2.0]
alpha_one_to_one_point_six_whole = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]
alpha_one_to_two_whole = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]

if __name__ == '__main__':
    grouped_param = group_benchmark_params([
        # ============ SIFT1M Alpha benchmarks ============
        # {  # building parameters
        #     "build_isolate_alpha": ["On", "Off"],
        #     "build_isolate_alpha_direct": "Off",
        #     "build_two_pass_indexing": "Off",
        #     "build_two_pass_index_scaled_l_first_pass": "Off",
        #     "build_two_pass_index_scaled_l_last_pass": "Off",
        #     "build_two_pass_index_sampled_visit_order": "Off",
        #     "build_sorted_visit_order": "Off",
        #     "build_tracking": False,
        #     # indexing parameters
        #     "index_data": "sift_base.fbin",
        #     "index_l": [50, 100, 150, 200, 400, 600, 800],
        #     "index_r": 128,
        #     "index_alpha": alpha_one_to_two_whole,
        #     # query parameters
        #     "query_num_runs": 3,
        #     "query_data": "sift_query.fbin",
        #     "query_l": [10, 25, 50],
        #     "query_k": [10], },
        # {  # building parameters
        #     "build_isolate_alpha": "Off",
        #     "build_isolate_alpha_direct": "On",
        #     "build_two_pass_indexing": "Off",
        #     "build_two_pass_index_scaled_l_first_pass": "Off",
        #     "build_two_pass_index_scaled_l_last_pass": "Off",
        #     "build_two_pass_index_sampled_visit_order": "Off",
        #     "build_sorted_visit_order": "Off",
        #     "build_tracking": False,
        #     # indexing parameters
        #     "index_data": "sift_base.fbin",
        #     "index_l": [50, 100, 150, 200, 400, 600, 800],
        #     "index_r": 128,
        #     "index_alpha": alpha_one_to_two_whole,
        #     # query parameters
        #     "query_num_runs": 3,
        #     "query_data": "sift_query.fbin",
        #     "query_l": [10, 25, 50],
        #     "query_k": [10], },
        # ============ SIFT1M Two Pass Construction benchmarks (K=10) ============
        {  # building parameters
            "build_isolate_alpha": "Off",
            "build_isolate_alpha_direct": "Off",
            "build_two_pass_indexing": "Off",
            "build_two_pass_index_scaled_l_first_pass": "Off",
            "build_two_pass_index_scaled_l_last_pass": "Off",
            "build_two_pass_index_sampled_visit_order": "Off",
            "build_sorted_visit_order": ["On", "Off"],
            "build_tracking": False,
            # indexing parameters
            "index_data": "rand_128_1k.fbin",
            "index_l": [50, 100, 200, 400, 600, 800],
            "index_r": [64, 128, 197],
            "index_alpha": [1.2, 1.4],
            # query parameters
            "query_num_runs": 5,
            "query_data": "rand_128_1k.fbin",
            "query_l": [10, 50],
            "query_k": [10], },
        {  # building parameters
            "build_isolate_alpha": "Off",
            "build_isolate_alpha_direct": "Off",
            "build_two_pass_indexing": "On",
            "build_two_pass_index_scaled_l_first_pass": ["On", "Off"],
            "build_two_pass_index_scaled_l_last_pass": "Off",
            "build_two_pass_index_sampled_visit_order": ["On", "Off"],
            "build_sorted_visit_order": ["On", "Off"],
            "build_tracking": False,
            # indexing parameters
            "index_data": "rand_128_1k.fbin",
            "index_l": [50, 100, 200, 400, 600, 800],
            "index_r": [64, 197],
            "index_alpha": [1.2, 1.4],
            # query parameters
            "query_num_runs": 5,
            "query_data": "rand_128_1k.fbin",
            "query_l": [10, 50],
            "query_k": [10], },
        {  # building parameters
            "build_isolate_alpha": "Off",
            "build_isolate_alpha_direct": "Off",
            "build_two_pass_indexing": "On",
            "build_two_pass_index_scaled_l_first_pass": "Off",
            "build_two_pass_index_scaled_l_last_pass": ["On", "Off"],
            "build_two_pass_index_sampled_visit_order": ["On", "Off"],
            "build_sorted_visit_order": ["On", "Off"],
            "build_tracking": False,
            # indexing parameters
            "index_data": "rand_128_1k.fbin",
            "index_l": [50, 100, 200, 400, 600, 800],
            "index_r": [64, 197],
            "index_alpha": [1.2, 1.4],
            # query parameters
            "query_num_runs": 5,
            "query_data": "rand_128_1k.fbin",
            "query_l": [10, 50],
            "query_k": [10], },
        # ============ SIFT1M Two Pass Construction benchmarks (K=50) ============
        {  # building parameters
            "build_isolate_alpha": "Off",
            "build_isolate_alpha_direct": "Off",
            "build_two_pass_indexing": "Off",
            "build_two_pass_index_scaled_l_first_pass": "Off",
            "build_two_pass_index_scaled_l_last_pass": "Off",
            "build_two_pass_index_sampled_visit_order": "Off",
            "build_sorted_visit_order": ["On", "Off"],
            "build_tracking": False,
            # indexing parameters
            "index_data": "rand_128_1k.fbin",
            "index_l": [50, 100, 200, 400, 600, 800],
            "index_r": [64, 128, 197],
            "index_alpha": [1.2, 1.4],
            # query parameters
            "query_num_runs": 5,
            "query_data": "rand_128_1k.fbin",
            "query_l": [50],
            "query_k": [50], },
        {  # building parameters
            "build_isolate_alpha": "Off",
            "build_isolate_alpha_direct": "Off",
            "build_two_pass_indexing": "On",
            "build_two_pass_index_scaled_l_first_pass": ["On", "Off"],
            "build_two_pass_index_scaled_l_last_pass": "Off",
            "build_two_pass_index_sampled_visit_order": ["On", "Off"],
            "build_sorted_visit_order": ["On", "Off"],
            "build_tracking": False,
            # indexing parameters
            "index_data": "rand_128_1k.fbin",
            "index_l": [50, 100, 200, 400, 600, 800],
            "index_r": [64, 197],
            "index_alpha": [1.2, 1.4],
            # query parameters
            "query_num_runs": 5,
            "query_data": "rand_128_1k.fbin",
            "query_l": [50],
            "query_k": [50], },
        {  # building parameters
            "build_isolate_alpha": "Off",
            "build_isolate_alpha_direct": "Off",
            "build_two_pass_indexing": "On",
            "build_two_pass_index_scaled_l_first_pass": "Off",
            "build_two_pass_index_scaled_l_last_pass": ["On", "Off"],
            "build_two_pass_index_sampled_visit_order": ["On", "Off"],
            "build_sorted_visit_order": ["On", "Off"],
            "build_tracking": False,
            # indexing parameters
            "index_data": "rand_128_1k.fbin",
            "index_l": [50, 100, 200, 400, 600, 800],
            "index_r": [64, 197],
            "index_alpha": [1.2, 1.4],
            # query parameters
            "query_num_runs": 5,
            "query_data": "rand_128_1k.fbin",
            "query_l": [50],
            "query_k": [50], },
    ])
    run_benchmarks(
        grouped_param,
        dry_run=True,
        last_state=None,
        use_existing_index=False,
        delete_index_after_query=True
    )
    _, _, combined_df = consolidate_data(
        grouped_param,
        # "../state_20250401_132510.json",
        # "../state_20250401_181019.json",
        # "../state_20250401_212354.json",
        # "../state_20250402_023113.json",
        # "../state_20250410_043337.json",
        # "../state_20250410_090451.json",
        # "../state_20250410_182159.json",
        # "../state_20250411_044535.json",
        # "../state_20250411_083939.json",
        # "../state_20250411_144559.json",
        # "../state_20250411_202031.json",
    )
    if combined_df is not None:
        plot_benchmarks(combined_df)
    else:
        print("None plotted.")
