import pandas as pd
from pathlib import Path

from utils import *

def claude_norag(data_dir, q_path):
    cld_no_rag_path = data_dir / Path("claude_noRAG.csv")

    cld = ClaudeAnswerGenerator(None, q_path)
    cld.generate_answers(RAG_eval=False)
    cld.save_ans_as_csv(cld_no_rag_path)

def gpt_norag(data_dir, q_path):
    gpt_no_rag_path = data_dir / Path("gpt_noRAG.csv")

    gpt = GPTAnswerGenerator(None, q_path)
    gpt.generate_answers(RAG_eval=False)
    gpt.save_ans_as_csv(gpt_no_rag_path)

def lmt_norag(data_dir, q_path):
    lmt_no_rag_path = data_dir / Path("lmt_noRAG.csv")

    lmt = LMTutorAnswerGenerator(None, q_path)
    lmt.generate_answers(RAG_eval=False)
    lmt.save_ans_as_csv(lmt_no_rag_path)

def lmt_arg(data_dir, q_path, vs_path):    
    dir_path, \
    lmtans_save_path, \
    claudeans_save_path, \
    gptans_save_path, \
    result_txt_path = set_up_dir(vs_path)

    lmt = LMTutorAnswerGenerator(vs_path, q_path)
    lmt.generate_answers(RAG_eval=True)
    lmt.save_ans_as_csv(lmtans_save_path)

def temp(data_dir, q_path):
    # vectorstore_path = Path("/home/hao.zhang/axie/LMTutor/DSC_291_vector_fineemb/")
    vectorstore_path_0 = Path("/home/hao.zhang/axie/LMTutor/DSC-291-vector_2/")
    # vectorstore_name = vectorstore_path.parts[-1]
    vectorstore_name_0 = vectorstore_path_0.parts[-1]
    
    lmt_ans_path = Path(f"/home/hao.zhang/axie/LMTutor/lmtans_{vectorstore_name_0}.csv")
    lmt_ans = pd.read_csv(lmt_ans_path)

    dir_path, \
    lmtans_save_path, \
    claudeans_save_path, \
    gptans_save_path, \
    result_txt_path = set_up_dir(vectorstore_path_0)

    pass

def eval(vs_path, data_dir):
    dir_path, \
    lmtans_save_path, \
    claudeans_save_path, \
    gptans_save_path, \
    result_txt_path = set_up_dir(vs_path)
    vs_name = vs_path.parts[-1]


    # change this part for different eval + ans_dict row names
    lmt_rag = pd.read_csv(lmtans_save_path)
    ans_col = None
    for col in lmt_rag.columns:
        if "_RAG" in col:
            ans_col = col
            break
    assert ans_col is not None
    
    claude_norag = pd.read_csv(data_dir / Path("claude_noRAG.csv"))
    base_col = 'claude_noRAG'

    ans_df = pd.merge(lmt_rag, claude_norag[['questions', 'claude_noRAG']], on='questions')
    # change this part for different eval + ans_dict row names


    rag_eval = LMTutorRAGEvaluator('chatgpt')
    response = []
    chosen_options = []
    for idx in tqdm(range(len(ans_df)), desc="evaluating"):
        row = ans_df.iloc[idx]
        ans_dict = {
            'lmtans_RAG': row[ans_col],
            'claude_noRAG': row[base_col],
            'ground_truth': row['ground_truth'],
        }

        curr_response = rag_eval.evaluate(row['questions'], ans_dict)

        response.append(curr_response)
        chosen_options.append(curr_response['option_chosen'])

    opt_count = {}
    for opt in chosen_options:
        if opt not in opt_count:
            opt_count[opt] = 1
        else:
            opt_count[opt] += 1

    result = pd.DataFrame(response)
    result_path = dir_path / Path(f"{ans_col}__{base_col}.csv")
    result.to_csv(result_path)

    with open(result_txt_path, 'a') as f:
        f.write(str(opt_count))
        f.write("\n")
        f.close()
    
    print(opt_count)


def main():
    data_dir = Path("/home/hao.zhang/axie/LMTutor/LMTutor/evaluation/data/")
    q_path = data_dir / Path("questions.csv")
    vs_path = Path("/home/hao.zhang/axie/LMTutor/DSC_291_vector_fineemb/")
    vs_path = Path("/home/hao.zhang/axie/LMTutor/DSC-291-vector_2/")
    vs_path = Path("/home/hao.zhang/axie/LMTutor/DSC-250-vector-w-291/")

    eval(vs_path, data_dir)

    # temp(data_dir, q_path)
    
    # claude_norag(data_dir, q_path)
    # gpt_norag(data_dir, q_path)
    # lmt_norag(data_dir, q_path)
    
    # lmt_arg(data_dir, q_path, vs_path)
    pass


if __name__ == "__main__":
    main()