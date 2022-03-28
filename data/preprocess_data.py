import json
from tqdm import tqdm
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()
    return args

def read_tables(table_file):
    table_dict = {}
    with open(table_file) as f:
        for line in tqdm(f):
            src_table = json.loads(line)
            table_id = src_table['tableId']
            table_dict[table_id] = src_table
    return table_dict

def get_out_table(table_id, src_table_dict):
    src_table = src_table_dict[table_id]
    caption = src_table['documentTitle']
    sub_caption = ''
    column_data = src_table['columns']
    out_data = []
    col_names = [a['text'] for a in column_data]
    out_data.append(col_names)
    row_data = src_table['rows'] 
    for row_info in row_data:  
        cell_data = row_info['cells']
        cell_text_lst = [a['text'] for a in cell_data]
        out_data.append(cell_text_lst)
    
    out_table = {
        'caption':caption,
        'sub_caption':sub_caption,
        'table_array':out_data
    }
    return out_table

def process_query(args, table_dict, out_table_dict, qt_rels):
    out_query_file = './%s/%s_query.txt' % (args.dataset, args.mode)
    f_o = open(out_query_file, 'w')
    retr_file = './%s/fusion_retrieved_%s.jsonl' % (args.dataset, args.mode)
    with open(retr_file) as f:
        for line in tqdm(f):
            item = json.loads(line)
            qid = item['id']
            question = item['question']
            
            gold_table_lst = item['table_id_lst']
            passage_info_lst = item['ctxs']
            
            retr_table_lst = [a['tag']['table_id'] for a in passage_info_lst]
            retr_table_lst = list(set(retr_table_lst))
            labels = [int(a in gold_table_lst) for a in retr_table_lst]
           
            if args.mode == 'train': 
                if (max(labels) < 1) or (min(labels) > 0):
                    continue 

            out_query_item = [qid, question]
            out_query_text = '\t'.join(out_query_item)
            f_o.write(out_query_text + '\n')
            
            for idx, retr_table_id in enumerate(retr_table_lst):
                assert ('\t' not in retr_table_id)
                out_qt_rel_item = [qid, str(0), retr_table_id, str(labels[idx])] 
                out_qt_rel_text = '\t'.join(out_qt_rel_item)
                qt_rels.append(out_qt_rel_text)
                 
                out_table = get_out_table(retr_table_id, table_dict)
                out_table_dict[retr_table_id] = out_table

    f_o.close()

def main():
    args = get_args()
    table_file = '/home/cc/data/%s/tables/tables.jsonl' % args.dataset
    table_dict =  read_tables(table_file)
    
    out_table_dict = {}
    qt_rels = []
    mode_lst = ['train', 'dev', 'test']
    for mode in tqdm(mode_lst):
        args.mode = mode
        process_query(args, table_dict, out_table_dict, qt_rels)

    print('writing qtrels')
    out_qt_rel_file = './%s/qtrels.txt' % args.dataset
    with open(out_qt_rel_file, 'w') as f_o_rel:
        for rel in tqdm(qt_rels):
            f_o_rel.write(rel + '\n')
    
    print('wrting tables')
    out_table_file = './%s/tables.json' % args.dataset
    with open(out_table_file, 'w') as f_o_table:
        json.dump(out_table_dict, f_o_table)

if __name__ == '__main__':
    main()
