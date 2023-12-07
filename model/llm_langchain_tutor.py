from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import Chroma, FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain.memory import ChatMessageHistory

from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader

from langchain.memory import ConversationBufferMemory

from langchain.llms import HuggingFacePipeline

import openai
import os
import sys

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from transformers import pipeline

PIPELINE_TYPE = {
    'lmsys/vicuna-13b-v1.5': 'text-generation'
}

class QuestionBasedRecursiveTextSplitter(RecursiveCharacterTextSplitter):
    def __init__(self, *args, **kwargs):
        super(QuestionBasedRecursiveTextSplitter, self).__init__(*args, **kwargs)
        anthropic_key = os.environ['ANTHROPIC_API']
        self.llm = Anthropic(model="claude-2", anthropic_api_key=anthropic_key)

    def create_documents(
        self, texts, metadatas=None
    ):
        """Create documents from a list of texts."""
        _metadatas = metadatas or [{}] * len(texts)
        documents = []
        for i, text in enumerate(tqdm(texts)):
            index = -1
            for chunk in self.split_text(text):
                metadata = copy.deepcopy(_metadatas[i])
                if self._add_start_index:
                    index = text.find(chunk, index + 1)
                    metadata["start_index"] = index
                metadata["page_content"] = chunk

                questions = self.topics_from_chunk(chunk)

                for question in questions:
                    new_doc = Document(page_content=question, metadata=metadata)
                    documents.append(new_doc)
        return documents

    def topics_from_chunk(self, doc):
        # Create the template
        template_string = '''Generate three questions that could be answered
        using information conatined in the following text: {doc}. The questions must
        be different and should focus on different topics covered in this text. Strictly
        do not provide any introduction or conclusion. Only provide a list of questions
        using the following template.
        - (Question 1 based on text)
        - (Question 2 based on text)
        - (Question 3 based on text)
        \n\nAssistant:
        '''

        # LLM call
        prompt_template = ChatPromptTemplate.from_template(template_string)

        try:
            chain = LLMChain(llm=self.llm, prompt=prompt_template)
            response = chain.run({
                "doc" : doc,
                })
            
            questions = [q.strip("- ") for q in response.split("\n\n")[1:]]
        except:
            questions = []
        
        print(questions)
        return questions


class LLMLangChainTutor():
    def __init__(self, doc_loader='dir', embedding='instruct_embedding', llm='hf_lmsys/vicuna-13b-v1.5', vector_store='faiss', langchain_mod='conversational_retrieval_qa', openai_key=None, embed_device='cuda',llm_device='cuda') -> None:
        self.openai_key = openai_key
        self.llm_name = llm
        self.embed_device = embed_device
        self.llm_device = llm_device

        self._document_loader(doc_loader=doc_loader)
        self._embedding_loader(embedding=embedding)
        self._vectorstore_loader(vector_store=vector_store)
        self._memory_loader()

    def _document_loader(self, doc_loader):
        if doc_loader == 'dir':
            self.doc_loader = DirectoryLoader
    
    def _embedding_loader(self, embedding):
        if embedding == 'openai':
            os.environ['OPENAI_API_KEY'] = self.openai_key
            self.embedding_model = OpenAIEmbeddings()
        
        elif embedding == 'instruct_embedding':
            self.embedding_model = HuggingFaceInstructEmbeddings(query_instruction="Represent the query for retrieval: ", model_kwargs={'device':self.embed_device}, encode_kwargs={'batch_size':32})
    
    
    def _vectorstore_loader(self, vector_store):
        if vector_store == 'faiss':
            self.vector_store = FAISS

    def _memory_loader(self):
        self.memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    def conversational_qa_init(self):
        if self.llm_name == 'openai':
            self.qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0), self.gen_vectorstore.as_retriever(), memory=self.memory, return_source_documents=True)
        
        elif self.llm_name.startswith('hf'):
            llm_name = self.llm_name.split('_')[-1]
            llm = HuggingFacePipeline.from_model_id(
                model_id=llm_name,
                task="text-generation",#"text2text-generation",#,
                model_kwargs={"temperature": 0.7, "max_length": 32, "torch_dtype": torch.float16},
                pipeline_kwargs={'max_new_tokens':32},
                device=self.llm_device ### self.device
            )
            self.memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')
            self.qa = ConversationalRetrievalChain.from_llm(llm, self.gen_vectorstore.as_retriever(), memory=self.memory, return_source_documents=True)

    def load_document(self, doc_path, glob='*.pdf', chunk_size=400, chunk_overlap=0):
        docs = self.doc_loader(doc_path, glob=glob, show_progress=True, use_multithreading=True, max_concurrency=16).load() ### many doc loaders

        text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap) ### hyperparams
        self.splitted_documents = text_splitter.split_documents(docs)
    
    def generate_vector_store(self):
        self.gen_vectorstore = self.vector_store.from_documents(self.splitted_documents, self.embedding_model)
    
    def save_vector_store(self, folder_path):
        self.gen_vectorstore.save_local(folder_path=folder_path)
    
    def load_vector_store(self, folder_path):
        self.gen_vectorstore = self.vector_store.load_local(folder_path=folder_path, embeddings=self.embedding_model)

    def similarity_search_topk(self, query, FIRST_PROMPT, k=4):
        retrieved_docs = self.gen_vectorstore.similarity_search(query, k=k)

        return self.limit_ctx_len(retrieved_docs, query, FIRST_PROMPT)
    
    # def similarity_search_thres(self, query, FIRST_PROMPT, thres=0.8):
    #     retrieval_result  = self.gen_vectorstore.similarity_search_with_score(query, k=10)
    #     retrieval_result = [d[0] for d in retrieval_result]
        
    #     return retrieval_result

    def limit_ctx_len(self, retrieved_docs, query, FIRST_PROMPT):
        # model length, context length, question length
        num_ctx = len(retrieved_docs)
        context = retrieved_docs[:num_ctx]
        context_len = len(" \n ".join([each.page_content for each in context]))
        model_len = self.tokenizer.model_max_length if self.tokenizer is not None else 1024*32
        query_len = len(query)
        pt_len = len(FIRST_PROMPT)
        input_len = sum([context_len, query_len, pt_len])

        while(input_len > 0.8 * model_len):
            num_ctx -= 1
            context = retrieved_docs[:num_ctx]
            context_len = len(" \n ".join([each.page_content for each in context]))
            input_len = sum([context_len, query_len, pt_len])
            
        return retrieved_docs[:num_ctx]

    def conversational_qa(self, user_input, use_rag=True):
        if use_rag:
            # return self.qa({'question': user_input})
            FIRST_PROMPT = "A chat between a student user and a teaching assistant. The assistant gives helpful, detailed, and polite answers to the user's questions based on the context.\n"
            PROMPT_TEMPLTATE = "CONTEXT: {context} \n USER: {user_input} \n ASSISTANT:"
            context = " \n ".join([each.page_content for each in self.similarity_search_topk(user_input, FIRST_PROMPT, k=5)])
            if self.first_conversation:
                prompt = FIRST_PROMPT + PROMPT_TEMPLTATE.format(context=context, user_input=user_input)
                self.first_conversation = False
            else:
                prompt = self.memory.messages[-1] + "\n\n " + PROMPT_TEMPLTATE.format(context=context, user_input=user_input)
        else:
            FIRST_PROMPT = "A chat between a student user and a teaching assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n"
            PROMPT_TEMPLTATE = "USER: {user_input} \n ASSISTANT:"
            if self.first_conversation:
                prompt = FIRST_PROMPT + PROMPT_TEMPLTATE.format(user_input=user_input)
                self.first_conversation = False
            else:
                prompt = self.memory.messages[-1] + "\n\n " + PROMPT_TEMPLTATE.format(user_input=user_input)

        # inputs = self.tokenizer(prompt, return_tensors="pt").to(self.llm_device)

        # # Generate
        # generate_ids = self.llm.generate(inputs.input_ids, max_length=30)
        # output = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        output = self.gen_pipe(prompt)[0]['generated_text']
        self.memory.add_message(prompt+output)
        # torch.cuda.empty_cache()

        return output

    def initialize_hf_llm(self):
        if self.llm_name.startswith('hf'):
            llm_name = self.llm_name.split('_')[-1]
            self.llm = AutoModelForCausalLM.from_pretrained(llm_name, temperature=0.7, torch_dtype=torch.float16).to(self.llm_device)
            self.tokenizer = AutoTokenizer.from_pretrained(llm_name)

            self.gen_pipe = pipeline(PIPELINE_TYPE[llm_name], model=self.llm, tokenizer=self.tokenizer, device=self.llm_device, max_new_tokens=512, return_full_text=False)

            self.memory = ChatMessageHistory()
            self.first_conversation = True
        else:
            self.tokenizer = None
    
    def get_embedding(self, data):
        return self.gen_vectorstore.embedding_function(data)

    def similarity_search_thres(self, query, thres=0.8):
        retrieval_result  = self.gen_vectorstore.similarity_search_with_score(query, k=10)
        retrieval_result = [(d[0].metadata['page_content'], d[0].page_content, d[0].metadata['source'], d[1]) for d in retrieval_result]
        return retrieval_result


if __name__ == '__main__':
    lmtutor = LLMLangChainTutor()
    lmtutor.load_vector_store("/home/haozhang/axie/LMTutor/data/DSC-291-vector")
    lmtutor.conversational_qa("What's the course?")

        
    