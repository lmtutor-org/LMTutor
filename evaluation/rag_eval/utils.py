import re, json, argparse, openai, os, time, torch, sys, tiktoken

import pandas as pd
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/hao.zhang/axie/LMTutor/")
from LMTutor.model.llm_langchain_tutor import LLMLangChainTutor


class AnswerGenerator:
    def __init__(self, vs_path, question_path) -> None:
        self.system_prompt = "You are a teaching assistant for the Advanced Data Mining course. " + \
                                "You are asked to answer questions from students."
        self.vs_path = vs_path
        self.vs_name = self.vs_path.parts[-1] if self.vs_path is not None else None
        self.question_path = question_path
        self.load_questions(self.question_path)
        self.ans_df = pd.DataFrame({
            "questions": self.questions,
            "ground_truth": self.ground_truth})
        self.model_name = "base"
        self.lmtutor = None

    def load_questions(self, question_path):
        self.questions = []
        self.ground_truth = []
        ### txt files
        if question_path.suffix == ('.txt'):
            with open(question_path, "r") as f:
                for each in f:
                    self.questions.append(re.sub(r"^\d+\.\s*", "", each))
        elif question_path.suffix == ('.csv'):
            q_file = pd.read_csv(question_path)
            for _, each in q_file.iterrows():
                self.questions.append(each['questions'])
                self.ground_truth.append(each['ground_truth'])

    def load_vs_only(self):
        self.lmtutor = LLMLangChainTutor(
            embedding="instruct_embedding",
            llm="hf_lmsys/vicuna-13b-v1.5",
            embed_device="cuda:6",
            llm_device="cuda:7",
        )
        self.lmtutor.load_vector_store(self.vs_path)

    def save_answers(self, save_path):
        with open(save_path, "w") as f:
            for each in self.answers:
                json.dump(each, f)
                f.write("\n")

    def save_ans_as_csv(self, save_path):
        self.ans_df.to_csv(save_path)

    def similarity_search_statistics(self):
        raise NotImplementedError

    def generate_answers(self, RAG_eval=False):
        # generates answers for all self.questions
        if RAG_eval:
            if self.lmtutor is None:
                self.load_vs_only()

            self.answers = []
            for q in tqdm(self.questions, desc=f"{self.model_name} answering questions with RAG"):
                answer = self.generate(q, RAG=RAG_eval)
                self.answers.append(answer)
            
            self.ans_df[f"{self.model_name}_{self.vs_name}_RAG"] = self.answers

        if not RAG_eval:
            self.answers = []
            for q in tqdm(self.questions, desc=f"{self.model_name} answering questions without RAG"):
                answer = self.generate(q, RAG=RAG_eval)
                self.answers.append(answer)
            
            self.ans_df[f"{self.model_name}_noRAG"] = self.answers


class LMTutorAnswerGenerator(AnswerGenerator):
    def __init__(self, *args) -> None:
        super().__init__(*args)
        self.lmtutor = LLMLangChainTutor(
            embedding="finetuned",
            embedding_path = "/home/hao.zhang/axie/LMTutor/exp-finetune",
            llm="hf_lmsys/vicuna-13b-v1.5",
            embed_device="cuda:6",
            llm_device="cuda:7",
        )
        if self.vs_path is not None:
            self.lmtutor.load_vector_store(self.vs_path)
        self.lmtutor.initialize_hf_llm()
        self.model_name = "lmt"

    def generate(self, question, RAG=True):
        self.lmtutor.first_conversation = True
        self.lmtutor.memory.clear()
        return self.lmtutor.conversational_qa(question, use_rag=RAG)

    def get_context_similarity(self):
        for answer in tqdm(self.answers, desc="Generating context similarity"):
            context_sim = []
            for cxt in answer["lmtutor_context"]:
                pass

    def similarity_search_statistics(self):
        result = []
        for q in tqdm(self.questions, desc="processing questions"):
            result.append([each[1] for each in self.lmtutor.similarity_search_thres(query=q)])

        result = pd.DataFrame(result)

        return result

class ClaudeAnswerGenerator(AnswerGenerator):
    def __init__(self, *args) -> None:
        super().__init__(*args)
        self.model_name = "claude"

    def generate(self, question, RAG=True):
        if RAG:
            prompt = self.system_prompt + "\n"
            prompt += "Please answer the following question only based on the context.\n"
            prompt += "CONTEXT: {context} \n Question: {question}"
            context = "\n".join([each.page_content for each in self.lmtutor.similarity_search_topk(question, prompt, k=5)])
            prompt = prompt.format(context=context, question=question)
            
            return self.retrieve_from_claude(prompt)
        else:
            prompt = self.system_prompt + "\n"
            prompt += "Please answer the following question. \n"
            prompt += "Question: {question}"
            prompt = prompt.format(question=question)

            return self.retrieve_from_claude(prompt)


class GPTAnswerGenerator(AnswerGenerator):
    def __init__(self, *args) -> None:
        super().__init__(*args)
        self.model_name = "gpt"

    def generate(self, question, RAG=True):
        if RAG:
            prompt = self.system_prompt + "\n"
            prompt += "Please answer the following question only based on the context.\n"
            prompt += "CONTEXT: {context} \n Question: {question}"
            context = "\n".join([each.page_content for each in self.lmtutor.similarity_search_topk(question, prompt, k=5)])
            prompt = prompt.format(context=context, question=question)
            
            return self.retrieve_from_gpt(prompt)
        else:
            prompt = self.system_prompt + "\n"
            prompt += "Please answer the following question. \n"
            prompt += "Question: {question}"
            prompt = prompt.format(question=question)

            return self.retrieve_from_gpt(prompt)

class LMTutorRAGEvaluator():
    def __init__(self, evaluator_model) -> None:
        self.evaluator_model = evaluator_model

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
    
    def unbiased_evaluate(self, ans_df, ans_col, base_col):
        possible_choices = ["A", "B"]
        choices = []
        reasons_a = []
        reasons_b = []
        ans_len = []
        base_len = []
        for idx in tqdm(range(len(ans_df)), desc="evaluating"):
            row = ans_df.iloc[idx]
            choice_a, response_a = self.pairwise_comparison(
                                    row['questions'], 
                                    row['ground_truth'],
                                    row[ans_col], 
                                    row[base_col])
            choice_b, response_b = self.pairwise_comparison(
                                    row['questions'],
                                    row['ground_truth'],
                                    row[base_col],
                                    row[ans_col])
            
            reasons_a.append(response_a)
            reasons_b.append(response_b)
            ans_len.append(len(tiktoken.encoding_for_model(self.evaluator_model).encode(row[ans_col])))
            base_len.append(len(tiktoken.encoding_for_model(self.evaluator_model).encode(row[base_col])))
            

            if choice_a not in possible_choices or choice_b not in possible_choices:  # CA, CB, CC, AC, BC
                choices.append(f"[{choice_a}]#[{choice_b}]")
                continue
            else:
                if choice_a != choice_b:    # AB BA  becasue position is switched
                    choices.append(choice_a)
                else:
                    choices.append("C")     # AA BB
            
            pass
            
        opt_count = {}
        for opt in choices:
            if opt not in opt_count:
                opt_count[opt] = 1
            else:
                opt_count[opt] += 1
        
        if "A" not in list(opt_count.keys()):
            opt_count["A"] = 0
        if "B" not in list(opt_count.keys()):
            opt_count["B"] = 0

        opt_count[ans_col] = opt_count["A"]
        opt_count[base_col] = opt_count["B"]
        
        ans_df['llm_judge_choice'] = choices
        ans_df['llm_judge_reason_a'] = reasons_a
        ans_df['llm_judge_reason_b'] = reasons_b
        ans_df[f"{ans_col}_len"] = ans_len
        ans_df[f"{base_col}_len"] = base_len

        return ans_df, opt_count
            

    def pairwise_comparison(self, question, reference_ans, ans1, ans2):
        sys_prompt = """Please act as an impartial judge and evaluate the quality of the responses provided by two
AI teaching assistants to the student question displayed below. Your evaluation should consider
correctness and helpfulness. You will be given a reference answer, assistant A's answer,
and assistant B's answer. Your job is to evaluate which assistant's answer is better.
Begin your evaluation by comparing both assistants' answers with the reference answer.
Identify and correct any mistakes. Avoid any position biases and ensure that the order in
which the responses were presented does not influence your decision. Do not allow the
length of the responses to influence your evaluation. Do not favor certain names of the
assistants. Be as objective as possible. After providing your explanation, output your
final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]"
if assistant B is better, and "[[C]]" for a tie."""

        user_prompt_template = """[Student Question]\n{student_question}\n\n
[The Start of Reference Answer]\n{reference_ans}\n[The End of Reference Answer]\n\n
[The Start of Assistant A's Answer]\n{ans1}\n[The End of Assistant A's Answer]\n\n
[The Start of Assistant B's Answer]\n{ans2}\n[The End of Assistant B's Answer]"""
        user_prompt = user_prompt_template.format(student_question=question, reference_ans=reference_ans, ans1=ans1, ans2=ans2)

        llm_judge_response = retrieve_from_gpt(user_prompt, sys_prompt=sys_prompt, model_name=self.evaluator_model)
        llm_judge_choice = self.parse_llm_judge(llm_judge_response)

        return llm_judge_choice, llm_judge_response


    def parse_llm_judge(self, llm_response, ans_dict=None):
        # match = re.search(r"(?i)answer-(\d) \(([^)]+)\)", llm_response)

        # if match:
        #     return match.group(2)
        # else:
        #     return "None"
       
       #########################

        # for k in ans_dict:
        #     if k == "ground_truth":
        #         continue
        #     elif k in llm_response:
        #         return k

        # return "None"

        #########################

        pattern = r'\[\[([A-C])\]\]'
        match = re.search(pattern, llm_response)

        if match:
            return match.group(1)
        else:
            return "None"

def retrieve_from_claude(prompt, sys_prompt=None):
        prompt = sys_prompt + "\n\n" + prompt if sys_prompt is not None else prompt
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

def retrieve_from_gpt(prompt, num_retries=10, sys_prompt=None, model_name="gpt-4-1106-preview"):
        sys_prompt = "You are a helpful assistant." if sys_prompt is None else sys_prompt
        import openai

        openai.api_key = os.environ["OPENAI_API_KEY"]

        response = None
        for attempt in range(num_retries):
            backoff = 2 ** (attempt)
            try:
                response = openai.ChatCompletion.create(
                    # model="gpt-4-1106-preview",
                    model=model_name,
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                    max_tokens=4096
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

def plot_dist_graph(dir_path, f_path, ans_col, base_col):
    df = pd.read_csv(f_path)
    
    len_list = {
        'A': [],
        'B': [],
        'C': [],
    }

    for idx, row in df.iterrows():
        if row['llm_judge_choice'] == 'A':
            len_list['A'].append(row[f"{ans_col}_len"])
        elif row['llm_judge_choice'] == 'B':
            len_list['B'].append(row[f"{base_col}_len"])

    plt.hist(len_list['A'], bins=20, alpha=0.5, label='A')
    plt.hist(len_list['B'], bins=20, alpha=0.5, label='B')

    plt.xlabel('Lengths')
    plt.ylabel('Frequency')
    plt.title('Answer length distribution')
    plt.legend(loc='upper right')

    plt.savefig(dir_path / Path("dist.png"))