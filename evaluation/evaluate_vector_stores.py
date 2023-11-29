import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from LMTutor.model.llm_langchain_tutor import LLMLangChainTutor
import pandas as pd
import numpy as np

def evaluate_vector_store_performance(lmtutor, query_df):
    query_results = []
    answer_results = []
    all_avg_query_l2 = []
    all_avg_gt_l2 = []
    for idx, row in query_df.iterrows():
        res = lmtutor.similarity_search_thres(row['questions'])
        query_results.append(res)

        all_avg_query_l2.append(np.average([d[2] for d in res]))

        # Get similarity of retreived docs and answers
        per_doc_l2 = []
        gt_embed = lmtutor.gen_vectorstore.embedding_function(row["ground_truth"])
        for data, _, _ in res:
            doc_embed = lmtutor.gen_vectorstore.embedding_function(data)
            l2_dist = np.linalg.norm(np.array(doc_embed) - np.array(gt_embed))
            per_doc_l2.append(l2_dist)
        
        all_avg_gt_l2.append(np.average(per_doc_l2))
        answer_results.append(per_doc_l2)
    
    query_df = query_df.assign(
        query_results = query_results,
        answer_results = answer_results,
        all_avg_query_l2 = all_avg_query_l2,
        all_avg_gt_l2 = all_avg_gt_l2
    )

    return query_df
    

if __name__ == '__main__':
    lmtutor = LLMLangChainTutor()
    lmtutor.load_vector_store("/home/pushkar/LMTutor-Package/DSC-291-vector")

    query_df = pd.read_csv("/home/pushkar/LMTutor/vicuna13b_gpt-3.5_answers_final.csv")
    final_eval_results = evaluate_vector_store_performance(lmtutor, query_df)
    final_eval_results.to_csv("/home/pushkar/LMTutor/vector_store_eval_results.csv")

    mean_query = final_eval_results["all_avg_query_l2"].mean()
    mean_gt = final_eval_results["all_avg_gt_l2"].mean()

    print(f"Average L2 distance between docs retrieved and query = {mean_query}")
    print(f"Average L2 distance between docs retrieved and ground truth = {mean_gt}")