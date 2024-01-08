import subprocess

sub_p_res = subprocess.run(['pip', 'install', "langchain" ,"sentence-transformers " , "chromadb"], stdout=subprocess.PIPE).stdout.decode('utf-8') #<cc-cm>
print("pip install downloded ", sub_p_res)


command = 'CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python'

sub_p_res = subprocess.run(command, shell=True, check=True)
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from huggingface_hub import hf_hub_download
from langchain.llms import LlamaCpp
import time
from langchain.prompts import PromptTemplate
import streamlit as st
import time

loader = DirectoryLoader('./link_to_data_folder', glob="**/*.txt", loader_cls=TextLoader)
pages = loader.load()

def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    document = chunks[10]
    print(document.page_content)
    print(document.metadata)

    return chunks

chunks_text = split_text(pages)

embedding = HuggingFaceEmbeddings(model_name='intfloat/e5-small-v2')

db = Chroma.from_documents(chunks_text, embedding=embedding)

MODEL_ID = "TheBloke/Mistral-7B-OpenOrca-GGUF"
MODEL_BASENAME = "mistral-7b-openorca.Q4_K_M.gguf"

model_path = hf_hub_download(
            repo_id=MODEL_ID,
            filename=MODEL_BASENAME,
            resume_download=True,
        )

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])


CONTEXT_WINDOW_SIZE = 1900
MAX_NEW_TOKENS = 2500
N_BATCH = 1
n_gpu_layers = 40
kwargs = {
          "model_path": model_path,
          "n_ctx": CONTEXT_WINDOW_SIZE,
          "max_tokens": MAX_NEW_TOKENS,
          "n_batch": N_BATCH,
          "n_gpu_layers": n_gpu_layers,
          "callback_manager": callback_manager,
          "verbose":True,
      }

llm = LlamaCpp(**kwargs)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 5}),
    return_source_documents=True,
    verbose=False,
)

template = """ 
You are and expert idea validator in terms of circular economy principles to address climate change.
Carefully evaluate the provided business idea based on six key metrics— circular economy (which include resource efficiency, waste management, long term viability, closed loop system, life cycle analysis), feasibility, scalability, innovation, and social impact and generate a score (0 to 10) for each metric, avoiding assumptions.
Ensure the evaluation is grounded in the context's content, avoiding assumptions.
Provide a comprehensive score and insights for each metric to guide investors in aligning their ambitions with the most promising business opportunities.
Add at the end the overall score for the evalution .
remember you don't read any buisness idea from the context , your only interest is the buisness idea provided by the user  
If the user provided something other than a business idea  , just say that you cannot answer politly in just few words , don't try to make up an answer from something in the context.
Context: {context}
Business Idea: {question}
your answer with the evaluation and the short description for each metric and ways of improving the business idea :  """


NEW_PROMPT = PromptTemplate(template=template, input_variables=['context', 'question'])
qa.combine_documents_chain.llm_chain.prompt = NEW_PROMPT





#---------------------------------------------------------  
# App title
st.set_page_config(page_title="🤖 MOAA advisor is Here")

# Replicate Credentials
with st.sidebar:
    st.title('MOAA Advisor is Here 🤖🌍')
    st.markdown(""" 
🌍 MOAA Advisor: Circular Economy Assessment

Invest with purpose using MOAA Advisor, your AI-powered guide to evaluate business proposals aligned with circular economy principles and climate change solutions.

📈 Metrics Evaluated:

Resource Efficiency
Waste Reduction
Environmental Impact
Innovation and Technology
Social Impact
Long-Term Viability
🔍 Insights for Investors:
Receive clear, metric-specific insights to inform your investment decisions. Backed by examples and observations, MOAA Advisor ensures comprehensive evaluations.

💡 Suggestions for Enhancement:
MOAA Advisor doesn't just score; it suggests improvements to align proposals better with circular economy principles.

🚀 Overall Score:
Effortlessly calculate an overall score for a quick snapshot of a proposal's sustainability.

🌱 Why MOAA Advisor:

Efficiency: Quick and precise evaluations.
Insightful Recommendations: Receive actionable suggestions.
Objective Assessments: AI-driven, unbiased evaluations.
Invest responsibly. Choose MOAA Advisor for sustainable and impactful investment decisions.
""")

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    if message["role"] == "user" :
      with st.chat_message(message["role"], avatar="user.png"):
          st.write(message["content"])
    else : 
      with st.chat_message(message["role"], avatar="logo.png"):
          st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

# Function for generating LLaMA2 response
def generate_llm_response(prompt_input):
    res = qa(prompt_input)
    return res['result']

# User-provided prompt
if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="user.png"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
  with st.chat_message("assistant", avatar="logo.png"):
      with st.spinner("Thinking..."):
          response = generate_llm_response(st.session_state.messages[-1]["content"])
      placeholder = st.empty()
      full_response = ''
      for item in response:
          full_response += item
          placeholder.markdown(full_response)
      placeholder.markdown(full_response)
  message = {"role": "assistant", "content": full_response}
  st.session_state.messages.append(message)

# Example prompt
with st.sidebar : 
  st.title('Examples :')

def promptExample1():
    prompt = "'Full Turn Fashion' is a comprehensive circular economy model designed to tackle fashion waste and pollution. Ushering the paradigm shift from usual 'produce-use-dispose' to 'reduce-reuse-recycle, we encourage consumers to return their worn-out clothing which we will collect, sort and recycle into unique, locally produced garments. Wastes unsuitable for direct recycling will be decomposed to produce organic dyes and materials. Operating specifically within fast-fashion importing countries, these processes involve local SMEs and communities, promoting local economy growth and creating jobs. Financial feasibility is achieved through revenue from the sale of upcycled apparel and premium value added to Geographical Indication. Our model is scalable, with potential for network expansion across other fast-fashion importing countries, steering the global fashion industry towards sustainability."
    st.session_state.messages.append({"role": "user", "content": prompt})

# Example prompt
def promptExample2():
    prompt = "My proposal is the 'Plastics Profit Cycle' a solution that not only encourages businesses to recycle and use recycled plastics but also creates substantial profit from it. In this model, businesses still collaborate to create a circular plastic economy, similar to the original model. However, we introduce strong incentives for participating businesses such as tax breaks, reduction in waste disposal fees, and an option to profit from the waste by selling it to recycling companies. This solution goes beyond merely reusing and recycling to create a system where plastic waste becomes an asset rather than a cost. Furthermore, we include a sustainability rating system that provides businesses with a ranking based on their participation in the Plastics Profit Cycle. This ranking could be used in marketing efforts, reinforcing the company's commitment to sustainable practices and boosting their public image. The structured incentivization combined with a rating system could prompt more businesses to join the cycle, enhancing its financial and environmental impact while increasing its feasibility and scalability."
    st.session_state.messages.append({"role": "user", "content": prompt})

# Example prompt
def promptExample3():
    prompt = "The ""Sustainable Packaging Rewards Program"" proposes a two-fold solution. First, it applies the principles of gamification, making sustainability an engaging pursuit. Customers return used packaging to stores in exchange for credits. These credits, accumulated over time, offer rewards ranging from discount vouchers to technology gadgets, thus providing immediate gratification. Second, businesses are involuntarily propelled towards using eco-friendly packaging as they bear the responsibility of recycling returned materials. The exact logistics of the operational ecosystem would need to be shaped through partnerships with retail outlets, businesses, and government bodies. With wide implementation, it is expected to dramatically decrease packaging waste while enhancing customer loyalty, thus marrying environmental sustenance with economic profitability. Challenges such as setting up efficient collection and recycling systems must be factored into feasibility and scalability evaluations."
    st.session_state.messages.append({"role": "user", "content": prompt})


st.sidebar.button('Solution example 1', on_click=promptExample1)
st.sidebar.button('Solution example 2', on_click=promptExample2)
st.sidebar.button('Solution example 3', on_click=promptExample3)


with st.sidebar:
    st.title('Disclaimer ⚠️:')
    st.markdown('May introduce false information')
    st.markdown('Some solutions may require additional analysis ')
