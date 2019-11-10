from collections import defaultdict

import numpy as np


def make_question_dict(train_df):
    question_dict = {}
    for i, row in train_df.iterrows():
        question_dict[row['qid1']] = row['question1']
        question_dict[row['qid2']] = row['question2']
    return question_dict


def create_graphs(train_df):
    positive_graph = defaultdict(set)
    negative_graph = defaultdict(set)

    for i, row in train_df.iterrows():
        if row['is_duplicate']:
            positive_graph[row['qid1']].add(row['qid2'])
            positive_graph[row['qid2']].add(row['qid1'])
        else:
            negative_graph[row['qid1']].add(row['qid2'])
            negative_graph[row['qid2']].add(row['qid1'])

    return positive_graph, negative_graph


def bfs(start, visited, graph):
    queue = [start]
    bucket = set()
    while queue:
        v = queue.pop(0)
        if v not in visited:
            visited.add(v)
            bucket.add(v)
            queue.extend(graph[v] - visited)

    return bucket


def make_buckets_with_bfs(question_dict, positive_graph, negative_graph):
    visited = set()
    buckets = []
    bucket_by_qid = {}

    for qid in question_dict:
        if qid not in visited:
            buckets.append(bfs(qid, visited, positive_graph))

    for i, bucket in enumerate(buckets):
        for qid in bucket:
            bucket_by_qid[qid] = i

    bucket_negatives = [set() for _ in buckets]

    for qid, edges in negative_graph.items():
        for neg_qid in edges:
            bucket_negatives[bucket_by_qid[qid]].add(bucket_by_qid[neg_qid])

    return buckets, bucket_negatives, bucket_by_qid


def generate_triplet_dataset(train_df, k_negatives=3):
    question_dict = make_question_dict(train_df)
    positive_graph, negative_graph = create_graphs(train_df)
    buckets, bucket_negatives, bucket_by_qid = make_buckets_with_bfs(question_dict, positive_graph, negative_graph)

    qids = list(question_dict.keys())

    for b_id, bucket in enumerate(buckets):
        if len(bucket) <= 1:
            continue

        negatives = [neg_id for neg_b in bucket_negatives[b_id] for neg_id in buckets[neg_b]]
        bucket_list = list(bucket)
        for i in range(len(bucket) - 1):
            for j in range(i + 1, len(bucket)):
                if len(negatives) >= k_negatives:
                    neg_chosen = np.random.choice(negatives, size=k_negatives)
                else:
                    neg_chosen = np.random.choice(len(qids), size=(k_negatives - len(negatives)))
                    neg_chosen = [qids[int(chosen)] for chosen in neg_chosen]
                    neg_chosen += negatives

                for neg_id in neg_chosen:
                    yield {
                        'bucket': b_id,
                        'anchor_id': bucket_list[i],
                        'anchor_q': question_dict[bucket_list[i]],
                        'pos_id': bucket_list[j],
                        'pos_q': question_dict[bucket_list[j]],
                        'neg_id': neg_id,
                        'neg_q': question_dict[neg_id]
                    }
