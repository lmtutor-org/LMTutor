from LMTutor.model.llm_langchain_tutor import LLMLangChainTutor

import re, json, argparse, openai, os, time, torch

import pandas as pd
from tqdm import tqdm
from pathlib import Path


class AnswerGenerator:
    def __init__(self, vs_path, question_path) -> None:
        self.system_prompt = "You are a teaching assistant for the Advanced Data Mining course. You are asked to answer questions from students."
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
            embedding="instruct_embedding",
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
        return 

    def evaluate_by_chatgpt(self, prompt):
        return retrieve_from_gpt(prompt)


def retrieve_from_claude(prompt):
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

def retrieve_from_gpt(prompt, num_retries=10):
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