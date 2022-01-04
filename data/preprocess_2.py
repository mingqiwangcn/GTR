import json
from tqdm import tqdm

dataset = 'fetaqa'
query_file = '/home/cc/data/%s/interactions/test_qas.jsonl' % dataset
retr_file = '/home/cc/code/%s_retr_data/preds_test.json' % dataset

table_file = '/home/cc/data/%s/tables/tables.jsonl' % dataset

out_query_file = './%s/test_query.txt' % dataset
out_qrel_file = './%s/qtrels.txt' % dataset
out_table_file = './%s/tables.json' % dataset

def read_retr_data(retr_file):
    retr_dict = {}
    with open(retr_file) as f:
        for line in f:
            item = json.loads(line)
            qid = item['qid']
            tag_lst = item['passage_tags']
            table_id_lst = [a['table_id'] for a in tag_lst]
            table_set = set(table_id_lst)
            table_lst = list(table_set)
            retr_dict[qid] = table_lst
    return retr_dict

def read_queries(query_file):
    query_dict = {}
    qid_lst = []
    with open(query_file) as f:
        for line in f:
            item = json.loads(line)
            qid = item['qid']
            qid_lst.append(qid)
            query_dict[qid] = item
    
    return qid_lst, query_dict

def gen_out_queries(qid_lst, query_dict, out_file):
    with open(out_file, 'w') as f_o:
        for qid in tqdm(qid_lst):
            question = query_dict[qid]['question']
            f_o.write(qid + '\t' + question + '\n')

def gen_out_qrels(qid_lst, query_dict, retr_dict, out_file):
    count = 0
    with open(out_file, 'w') as f_o:
        for qid in tqdm(qid_lst):
            table_id_lst = retr_dict[qid]
            gold_table_set = set(query_dict[qid]['table_id_lst'])
            
            related_lst = []
            for table_id in table_id_lst:
                related = int(table_id in gold_table_set)
                f_o.write(qid + '\t' + str(0) + '\t' + table_id + '\t' + str(related) + '\n') 
                related_lst.append(related)

            count += max(related_lst)
    
    print('%.2f' % (count * 100/len(qid_lst))) 

def gen_out_tables(src_file, out_file):
    out_table_dict = {}
    with open(src_file) as f:
        for line in tqdm(f):
            src_table = json.loads(line)
            table_id = src_table['tableId']
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
            out_table_dict[table_id] = out_table

    with open(out_file, 'w') as f_o:
        json.dump(out_table_dict, f_o)

def main():
    qid_lst, query_dict = read_queries(query_file)
    retr_dict = read_retr_data(retr_file) 
   
    gen_out_queries(qid_lst, query_dict, out_query_file)
    
    gen_out_qrels(qid_lst, query_dict, retr_dict, out_qrel_file)

    gen_out_tables(table_file, out_table_file)

if __name__ == '__main__':
    main()
