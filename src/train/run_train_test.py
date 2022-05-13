import random
import datetime
import pandas as pd
from pytorch_transformers import WarmupLinearSchedule

from src.utils.trec_eval import write_trec_result, get_metrics
from src.utils.load_data import load_queries, load_tables, load_qt_relations
from src.utils.reproducibility import set_random_seed
from src.model.retrieval import MatchingModel
from src.graph_construction.tabular_graph import TabularGraph
from src.utils.process_table_and_query import *
import torch
import json

import numpy as np
import os

queries = None
tables = None
qtrels = None

global_step = 0

def get_output_dir(config):
    output_dir = config['output_dir']
    return output_dir

def evaluate(epoch, step, config, model, query_id_list, mode):
    output_dir = get_output_dir(config)
    if step is not None:
        out_file_name = 'epoch_%d_step_%d_pred_%s.jsonl' % (epoch, step, mode)
    else:
        out_file_name = 'epoch_%d_pred_%s.jsonl' % (epoch, mode)
    out_pred_file = os.path.join(output_dir, out_file_name)
    f_o_pred = open(out_pred_file, 'w')
    metric_dict = {}
    top_n_lst = [1, 3, 5]
    for max_top in top_n_lst:
        metric_dict[max_top] = []
    model.eval()
    with torch.no_grad():
        for qid in tqdm(query_id_list):
            query = queries["sentence"][qid]
            query_feature = queries["feature"][qid].to("cuda")
           
            score_info_lst = [] 
            for (tid, rel) in qtrels[qid].items():
                table = tables[tid]
                dgl_graph = tables[tid]["dgl_graph"].to("cuda")
                node_features = tables[tid]["node_features"].to("cuda")

                score = model(table, query, dgl_graph, node_features, query_feature).item()
                
                score_info = {
                    'table_id':tid,
                    'label':int(rel),
                    'score':score
                }
                score_info_lst.append(score_info)
            
            sorted_idxes = get_top_pred(score_info_lst)
            correct_lst = [score_info_lst[a]['label'] for a in sorted_idxes]
            for max_top in metric_dict:
                max_correct_lst = metric_dict[max_top]
                max_correct = max(correct_lst[:max_top])
                max_correct_lst.append(max_correct)

            out_pred_item = {
                'qid':qid,
                'sorted_idxes':sorted_idxes,
                'score_info':score_info_lst
            }
            f_o_pred.write(json.dumps(out_pred_item) + '\n')
    
    f_o_pred.close()
    result = {}
    for max_top in metric_dict:
        mean_metric = np.mean(metric_dict[max_top]) * 100
        metric_key = 'p@%d' % max_top
        result[metric_key] = mean_metric
    return result

def get_top_pred(score_info_lst):
    score_lst = np.array([a['score'] for a in score_info_lst])
    sorted_idxes = np.argsort(-score_lst)
    sorted_idxes_int = [int(a) for a in sorted_idxes]
    return sorted_idxes_int 

def train(epoch, config, model, train_query_ids, optimizer, scheduler, loss_func, dev_query_ids, test_query_ids):
    random.shuffle(train_query_ids)

    model.train()

    eloss = 0
    batch_loss = 0
    n_iter = 0
    cnt = 0
   
    output_dir = get_output_dir(config) 
    global global_step
    checkpoint_steps = config['checkpoint_steps']
    pbar = tqdm(train_query_ids)
    for qid in pbar:
        cnt += 1

        query = queries["sentence"][qid]
        query_feature = queries["feature"][qid].to("cuda")

        logits = []
        label = None
        pos = 0
        for (tid, rel) in qtrels[qid].items():
            if rel == 1:
                label = torch.LongTensor([pos]).to("cuda")

            table = tables[tid]
            dgl_graph = tables[tid]["dgl_graph"].to("cuda")
            node_features = tables[tid]["node_features"].to("cuda")

            logit = model(table, query, dgl_graph, node_features, query_feature)
            logits.append(logit)

            pos += 1

        if label is None or len(logits) < 2:
            print(qid, query)
            continue

        loss = loss_func(torch.cat(logits).view(1, -1), label.view(1))

        batch_loss += loss

        n_iter += 1
        
        global_step += 1

        if n_iter % config["batch_size"] == 0 or cnt == len(train_query_ids):
            batch_loss /= config["batch_size"]
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            pbar.set_postfix(loss=batch_loss.item())

            optimizer.zero_grad()
            batch_loss = 0

        eloss += loss.item()
        
        if global_step % checkpoint_steps == 0:
            evaluate_step(epoch, global_step, config, model, dev_query_ids, test_query_ids, output_dir)

    evaluate_step(epoch, None, config, model, dev_query_ids, test_query_ids, output_dir)

    return eloss / len(train_query_ids)

def save_model(epoch, step, model, output_dir):
    if step is not None:
        file_name = 'epoc_%d_step_%d_model.bin' % (epoch, step)
    else:
        file_name = 'epoc_%d_model.bin' % epoch

    out_file = os.path.join(output_dir, file_name)
    torch.save(model.state_dict(), out_file)

def evaluate_step(epoch, step, config, model, dev_query_ids, test_query_ids, output_dir):
    if step is None:
        step_desp = ''
    else:
        step_desp = str(step)
    global best_metrics
    save_model(epoch, step, model, output_dir)
    dev_metrics = evaluate(epoch, step, config, model, dev_query_ids, 'dev')
    if best_metrics is None or dev_metrics['p@1'] > best_metrics['p@1']:
        best_metrics = dev_metrics
       
        log_msg = 'epoch=%d, step=%s, dev, %s' % (epoch, step_desp, json.dumps(best_metrics))
        f_o_log.write(log_msg + '\n')

        test_metrics = evaluate(epoch, step, config, model, test_query_ids, 'test')
        
        log_msg = 'epoch=%d, step=%s, test, %s' % (epoch, step_desp, json.dumps(test_metrics))
        f_o_log.write(log_msg + '\n')

        f_o_log.flush()

        print(datetime.datetime.now(), 'epoch', epoch, 'step', step_desp, 'dev', dev_metrics, 'test',
              test_metrics, "*", flush=True)
    else:
        print(datetime.datetime.now(), 'epoch', epoch, 'step', step_desp, 'dev', dev_metrics, flush=True)


def train_and_test(config):
    set_random_seed()
    
    out_dataset_dir = get_output_dir(config)
    if os.path.isdir(out_dataset_dir):
        print('[%s] already exists' % out_dataset_dir)
        return
    os.makedirs(out_dataset_dir)

    global f_o_log
    log_file = os.path.join(out_dataset_dir, 'log.txt')    
    f_o_log = open(log_file, 'w')

    global queries, tables, qtrels

    train_queries = load_queries(config["data_dir"], "train_query.txt")
    dev_queries = load_queries(config["data_dir"], "dev_query.txt")
    test_queries = load_queries(config["data_dir"], "test_query.txt")
    tables = load_tables(config["data_dir"])
    qtrels = load_qt_relations(config["data_dir"])

    # remove invalid queries
    train_query_ids = list(train_queries.keys())
    dev_query_ids = list(dev_queries.keys())
    test_query_ids = list(test_queries.keys())
    
    num_total_queries = len(train_query_ids) + len(dev_query_ids) + len(test_query_ids)
    queries = {}
    queries["sentence"] = {}
    queries["sentence"].update(train_queries)
    queries["sentence"].update(dev_queries)
    queries["sentence"].update(test_queries)

    del train_queries, dev_queries, test_queries
    assert(len(queries['sentence']) == num_total_queries)

    #######################################

    constructor = TabularGraph(config["fasttext"], config["merge_same_cells"])

    queries["feature"] = process_queries(queries["sentence"], constructor)
    process_tables(tables, constructor)

    model = MatchingModel(bert_dir=config["bert_dir"], do_lower_case=config["do_lower_case"],
                          bert_size=config["bert_size"], gnn_output_size=config["gnn_size"])

    if config["use_pretrained_model"]:
        pretrain_model = torch.load(config["pretrained_model_path"])
        model.load_state_dict(pretrain_model, strict=False)

    model = model.to("cuda")

    loss_func = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam([
        {'params': model.bert.parameters(), 'lr': config["bert_lr"]},
        {'params': (x for x in model.parameters() if x not in set(model.bert.parameters())), 'lr': config["gnn_lr"]}
    ])
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=config["warmup_steps"], t_total=config["total_steps"])

    global best_metrics
    best_metrics = None

    for epoch in range(config['epoch']):
        train(epoch, config, model, train_query_ids, optimizer, scheduler, loss_func, dev_query_ids, test_query_ids)

    f_o_log.close()


def run_evaluate(config):
    output_dir = get_output_dir(config)
    if os.path.isdir(output_dir):
        print('[%s] already exists' % output_dir)
        return
    os.makedirs(output_dir)
    log_file = os.path.join(output_dir, 'log.txt')
    f_o_log = open(log_file, 'w')
    global queries, tables, qtrels

    test_queries = load_queries(config["data_dir"], "test_query.txt")
    tables = load_tables(config["data_dir"])
    qtrels = load_qt_relations(config["data_dir"])

    test_query_ids = list(test_queries.keys())

    queries = {}
    queries["sentence"] = {}
    queries["sentence"].update(test_queries)

    #######################################

    constructor = TabularGraph(config["fasttext"], config["merge_same_cells"])

    queries["feature"] = process_queries(queries["sentence"], constructor)
    process_tables(tables, constructor)

    model = MatchingModel(bert_dir=config["bert_dir"], do_lower_case=config["do_lower_case"],
                          bert_size=config["bert_size"], gnn_output_size=config["gnn_size"])

    if config["use_pretrained_model"]:
        pretrain_model = torch.load(config["pretrained_model_path"])
        model.load_state_dict(pretrain_model, strict=False)

    model = model.to("cuda")
    print(config)

    test_metrics = evaluate(0, 0, config, model, test_query_ids, 'test')
    log_msg = 'test, %s' % (json.dumps(test_metrics))
    print(log_msg)
    f_o_log.write(log_msg + '\n')

    f_o_log.close()


