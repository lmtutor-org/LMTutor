import sys
from pathlib import Path
sys.path.insert(0, str(Path("/home/hao.zhang/axie/LMTutor/")))
from LMTutor.model.llm_langchain_tutor import LLMLangChainTutor
from evaluation_pipeline import *
import pandas as pd
import os
import time


class LMTutorRAGEvaluator():
    def __init__(self, evaluator) -> None:
        self.evaluator = evaluator

    def evaluate(self, question, ans_dict):
        prompt = f"Consider the following answers in response to the " + \
                 f"question:\n{question} "
        i = 0
        
        for k, v in ans_dict.items():
            if k == "ground_truth":
                continue
            i += 1
            prompt += f"\n\n Answer-{i} ({k}):\n{v} "
        prompt += f"\n\n Ground Truth:\n{ans_dict['ground_truth']}"

        prompt += f"\n\n Which answer is more accurate and better-written compared to the ground truth? " + \
                  f"Please answer with one of the following options and explaine why.\n\n Options: "
        
        ans_idx = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
        i = 0
        for k in ans_dict.keys():
            if k == "ground_truth":
                continue
            prompt += f"\n({ans_idx[i]}) {k}"
            i += 1
        prompt += "\n\n"

        response = None
        if self.evaluator == 'claude':
            response = self.evaluate_by_claude(prompt)
        elif self.evaluator == 'chatgpt':
            response = self.evaluate_by_chatgpt(prompt)
        
        assert response is not None, "response is None!"
        option_chosen = self.parse_llm_judge(response, ans_dict)
        reason = response[(response.find(option_chosen) + len(option_chosen)):] if option_chosen != "None" else response

        result = {
            "question": question,
            "option_chosen": option_chosen,
            "reason": reason,
        }

        return result
    
    def parse_llm_judge(self, llm_response, ans_dict):
        # match = re.search(r"(?i)answer-(\d) \(([^)]+)\)", llm_response)

        # if match:
        #     return match.group(2)
        # else:
        #     return "None"
       
        for k in ans_dict:
            if k == "ground_truth":
                continue
            elif k in llm_response:
                return k

        return "None"
        

    def evaluate_by_claude(self, prompt):
        from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

        anthropic = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

        completion = anthropic.completions.create(
            model = 'claude-2',
            # max_retries=10,
            max_tokens_to_sample=150,
            temperature=0,
            prompt=f"{HUMAN_PROMPT} {prompt} {AI_PROMPT}"
        )

        response = completion.completion

        return response
    
    def evaluate_by_chatgpt(self, prompt, num_retries=10):
        import openai

        openai.api_key = os.environ["OPENAI_API_KEY"]

        response = None
        for attempt in range(num_retries):
            backoff = 2 ** (attempt)
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4-1106-preview",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                    max_tokens=150
                )
                break
            except openai.error.APIError as e:
                print(f"OpenAI API returned an API Error: {e}")
                if attempt == num_retries - 1:
                    raise
            except openai.error.APIConnectionError as e:
                print(f"Failed to connect to OpenAI API: {e}")
                if attempt == num_retries - 1:
                    raise
            except openai.error.RateLimitError as e:
                print(f"OpenAI API request exceeded rate limit: {e}")
                if attempt == num_retries - 1:
                    raise
            except openai.error.Timeout as e:
                print(f"OpenAI API request timed out: {e}")
                if attempt == num_retries - 1:
                    raise
            except openai.error.InvalidRequestError as e:
                print(f"Invalid request to OpenAI API: {e}")
                if attempt == num_retries - 1:
                    raise
            except openai.error.AuthenticationError as e:
                print(f"Authentication error with OpenAI API: {e}")
                if attempt == num_retries - 1:
                    raise
            except openai.error.ServiceUnavailableError as e:
                print(f"OpenAI API service unavailable: {e}")
                if attempt == num_retries - 1:
                    raise
            time.sleep(backoff)

        if response is None:
            print(f"Failed to get response after {num_retries} retries")
            return "None"

        return response['choices'][0]['message']['content'].strip()


def let_lmtutor_generate_answers(vectorstore_path, question_path, lmtans_save_path, RAG_eval=True):
    vectorstore_name = vectorstore_path.parts[-1]
    lmtutor_ans = LMTutorAnswerGenerator(vectorstore_path)
    print("********lmtutor_ans initialized********")
    lmtutor_ans.load_questions(question_path)
    # lmtutor_ans.questions = lmtutor_ans.questions[:1]
    lmtutor_answers_df = pd.DataFrame({
                "questions": lmtutor_ans.questions,
                "ground_truth": lmtutor_ans.ground_truth
                })

    if RAG_eval:
        lmtutor_ans.generate_answers(RAG_eval=True)
        lmtutor_answers_df[f"lmans_RAG_{vectorstore_name}"] = lmtutor_ans.answers
    else:
        lmtutor_ans.generate_answers(RAG_eval=False)
        lmtutor_answers_df["lmans_noRAG"] = lmtutor_ans.answers    
    
    lmtutor_answers_df.to_csv(lmtans_save_path, index=False)


def let_claude_generate_answers(vectorstore_path, question_path, claudeans_save_path, RAG_eval=True):
    vectorstore_name = vectorstore_path.parts[-1]
    claude_ans = ClaudeAnswerGenerator(vectorstore_path)
    claude_ans.load_questions(question_path)
    claude_ans.questions = claude_ans.questions[:1]
    claude_ans.ground_truth = claude_ans.ground_truth[:1]

    claude_ans_df = pd.DataFrame({
            "questions": claude_ans.questions,
            "ground_truth": claude_ans.ground_truth
            })
    
    if RAG_eval:
        claude_ans.generate_answers(RAG_eval=True)
        claude_ans_df[f"claude_RAG_{vectorstore_name}"] = claude_ans.answers
    else:
        claude_ans.generate_answers(RAG_eval=False)
        claude_ans_df["claude_no_RAG"] = claude_ans.answers
    
    claude_ans_df.to_csv(claudeans_save_path, mode='w', header=True, index=False)


def let_gpt_generate_answers(vectorstore_path, question_path, gptans_save_path, RAG_eval=True):
    vectorstore_name = vectorstore_path.parts[-1]
    gpt_ans = GPTAnswerGenerator(vectorstore_path)
    gpt_ans.load_questions(question_path)

    gpt_ans_df = pd.DataFrame({
            "questions": gpt_ans.questions,
            "ground_truth": gpt_ans.ground_truth})

    if RAG_eval:
        gpt_ans.generate_answers(RAG_eval=True)
        gpt_ans_df[f"gpt_RAG_{vectorstore_name}"] = gpt_ans.answers
    else:
        gpt_ans.generate_answers(RAG_eval=False)
        gpt_ans_df["gpt_noRAG"] = gpt_ans.answers
    
    gpt_ans_df.to_csv(gptans_save_path, mode='w', header=True, index=False)




def set_up_dir(vectorstore_path):
    vectorstore_name = vectorstore_path.parts[-1]
    dir_path = Path(f"/home/hao.zhang/axie/LMTutor/LMTutor/evaluation/{vectorstore_name}_eval")
    if not dir_path.exists():
        dir_path.mkdir()
    lmtans_save_path = dir_path / f"lmtans_{vectorstore_name}.csv"
    claudeans_save_path = dir_path / f"claudeans_{vectorstore_name}.csv"
    gptans_save_path = dir_path / f"gptans_{vectorstore_name}.csv"
    result_txt_path = dir_path / f"result_{vectorstore_name}.txt"

    return dir_path, lmtans_save_path, claudeans_save_path, gptans_save_path, result_txt_path


def exp_for_new_vs(vs_path):
    dir_path, \
    lmtans_save_path, \
    claudeans_save_path, \
    gptans_save_path = set_up_dir(vs_path)
    q_path = "/home/hao.zhang/axie/LMTutor/LMTutor/evaluation/data/questions.csv"

    lmt = LMTutorAnswerGenerator(vs_path)
    lmt.generate_answers(RAG_eval=True)


    cld = ClaudeAnswerGenerator(vs_path)
    cld.generate_answers(RAG_eval=True)


    gpt = GPTAnswerGenerator(vs_path)
    gpt.generate_answers(RAG_eval=True)


def main():
    vectorstore_path = Path("/home/hao.zhang/axie/LMTutor/DSC_291_vector_fineemb/")
    vectorstore_path_0 = Path("/home/hao.zhang/axie/LMTutor/DSC-291-vector_2/")
    vectorstore_name = vectorstore_path.parts[-1]
    vectorstore_name_0 = vectorstore_path_0.parts[-1]
    question_path = "/home/hao.zhang/axie/LMTutor/vicuna13b_gpt-3.5_answers_final.csv"
    lmtans_save_path = f"/home/hao.zhang/axie/LMTutor/lmtans_{vectorstore_name}.csv"
    claudeans_save_path = f"/home/hao.zhang/axie/LMTutor/claudeans_{vectorstore_name}.csv"
    gptans_save_path = f"/home/hao.zhang/axie/LMTutor/gptans_{vectorstore_name_0}.csv"

    # let_lmtutor_generate_answers(vectorstore_path, question_path, lmtans_save_path)
    let_claude_generate_answers(vectorstore_path, question_path, claudeans_save_path)
    # # let_gpt_generate_answers(vectorstore_path, question_path, gptans_save_path)
    exit(0)
    
    rag_eval = LMTutorRAGEvaluator("chatgpt")
    no_vs_ans = pd.read_csv("/home/hao.zhang/axie/LMTutor/lmtutor_answers_RAG_dsc291vs_eval.csv")
    vs_ans = pd.read_csv(f"/home/hao.zhang/axie/LMTutor/lmtans_{vectorstore_name}.csv")
    # claude_RAG_ans = pd.read_csv(f"/home/hao.zhang/axie/LMTutor/claudeans_{vectorstore_name}.csv")
    # claude_RAG_ans.columns = ["", "claude_RAG", "questions", "ground_truth"]
    # claude_ans = pd.read_csv(f"/home/hao.zhang/axie/LMTutor/claudeans_DSC-291-vector.csv")
    # claude_ans.columns = ["", "claude_RAG", "claude_no_RAG", "questions", "ground_truth"]
    # ans = pd.merge(claude_RAG_ans, claude_ans[["questions", "claude_no_RAG"]], on='questions')
    gpt_ans = pd.read_csv(gptans_save_path)
    ans = pd.merge(vs_ans, gpt_ans[["questions", f"gpt_noRAG"]], on='questions')
    # ans = gpt_ans
    # ans = pd.merge(vs_ans, gpt_ans[["questions", f"gpt_RAG_{vectorstore_name}"]], on='questions')
    # ans = lmtutor_ans.merge(vs0_ans[["lmtutor_answers_RAG"]], left_index=True, right_index=True, suffixes=('', '_vs_new'))
    
    response = []
    chosen_options = []
    for idx, row in tqdm(ans.iterrows(), desc="evaluating with LLM"):
    # for idx, row in ans.iterrows():
        print("evaluating question: ", idx)
        ans_dict = {
            "lmtans_RAG": row[f"lmans_RAG_{vectorstore_name}"],
            "gpt_answers_noRAG": row["gpt_noRAG"],
            "ground_truth": row["ground_truth"],
            # "chatgpt": row["gpt3"]
            # "gpt_answers_RAG": row[f"gpt_RAG_{vectorstore_name}"]
        }

        curr_response = rag_eval.evaluate(
            row["questions"],
            ans_dict)
        response.append(curr_response)
        chosen_options.append(curr_response["option_chosen"])
    
    opt_count = {}
    for opt in chosen_options:
        if opt not in opt_count:
            opt_count[opt] = 1
        else:
            opt_count[opt] += 1

    result = pd.DataFrame(response)
    result.to_csv(f"/home/hao.zhang/axie/LMTutor/{vectorstore_name}_lmtRAG_gptnoRAG_by_claude.csv")

    f = open("/home/hao.zhang/axie/LMTutor/{vectorstore_name}_lmtRAG_gptnoRAG_by_claude_results.txt", "w")
    f.write(str(opt_count))
    f.close()

    print(opt_count)
    
    



if __name__ == "__main__":
    main()