
![MOAA Advisor](https://github.com/QoutiOussama13/AI-EarthHack-MOAA-/assets/81428754/15d1e580-c1ad-454d-8152-c0fc7fc50708)

# 🌍 MOAA Advisor: Circular Economy Assessment

Invest with purpose using **MOAA Advisor**, your AI-powered guide to evaluate business proposals aligned with circular economy principles and climate change solutions.

## 📈 Metrics Evaluated:

- Resource Efficiency
- Waste Reduction
- Environmental Impact
- Innovation and Technology
- Social Impact
- Long-Term Viability

## 🔍 Insights for Investors:
Receive clear, metric-specific insights to inform your investment decisions. Backed by examples and observations, **MOAA Advisor** ensures comprehensive evaluations.

## 💡 Suggestions for Enhancement:
**MOAA Advisor** doesn't just score; it suggests improvements to align proposals better with circular economy principles.

## 🚀 Overall Score:
Effortlessly calculate an overall score for a quick snapshot of a proposal's sustainability.

## 🌱 Why MOAA Advisor:

- **Efficiency:** Quick and precise evaluations.
- **Insightful Recommendations:** Receive actionable suggestions.
- **Objective Assessments:** AI-driven, unbiased evaluations.


## Approach:
We utilized the Retrieval-Augmented Generation [RAG](https://arxiv.org/abs/2005.11401) technique to enhance the accuracy and reliability of our generative AI models. RAG optimizes the output of a large language model by referencing an authoritative knowledge base outside of its training data sources before generating a response.

The main tool we employed for making the RAG is [Langchain 🦜️⛓️](https://python.langchain.com/docs/get_started/introduction) for performence and simplicity .

### Phase 1: Data Collection and Embedding
1. **Data Collection:**
   - Gathered detailed data from different sources containing information about companies in the circular economy and metrics used to evaluate the circular economy business ideas

2. **Embedding:**
   - Utilized a model from [Hugging Face](https://huggingface.co/intfloat/e5-small-v2) to transform data into vector representation.

3. **Vector Storage:**
   - Stored embeddings in the vector database [Chroma](https://www.trychroma.com).

### Phase 2: User Interaction
1. **User Interaction:**
   - Users input prompts through the [Streamlit](https://streamlit.io/) interface.

2. **Prompting:**
   - we gave the model this prompt template :
```pyhton
You are and expert idea validator in terms of circular economy principles to address climate change.
Carefully evaluate the provided business idea based on six key metrics— circular economy (which include resource efficiency, waste management, long term viability, closed loop system, life cycle analysis), feasibility, scalability, innovation, and social impact and generate a score (0 to 10) for each metric, avoiding assumptions.
Ensure the evaluation is grounded in the context's content, avoiding assumptions.
Provide a comprehensive score and insights for each metric to guide investors in aligning their ambitions with the most promising business opportunities.
Add at the end the overall score for the evalution .
remember you don't read any buisness idea from the context , your only interest is the buisness idea provided by the user  
If the user provided something other than a business idea  , just say that you cannot answer politly in just few words , don't try to make up an answer from something in the context.
Context: {context}
Business Idea: {question}
your answer with the evaluation and the short description for each metric and ways of improving the business idea :  
```
   - Embedded the template + the user prompt using the same model used for data embedding.

4. **Similarity Search:**
   - Compared embedded user prompts with vector database data to find the best match.

### Phase 3: Large Language Model (LLM)
1. **LLM Selection:**
   - Utilized the [Mistral-7B-OpenOrca](https://huggingface.co/Open-Orca/Mistral-7B-OpenOrca) language model, incorporating [Q4 quantization](https://huggingface.co/TheBloke/Mistral-7B-OpenOrca-GGUF).

2. **Answers Generation:**
   - Fed results of similarity search to the LLM to generate answers based on user-specific needs and available information

3. **User Interface (UI):**
   - Presented answers in the Streamlit UI for user consumption.

## Global architechture :
![Design sans titre](https://github.com/QoutiOussama13/AI-EarthHack-MOAA-/assets/81428754/cfb028ba-426e-4c47-92ef-c777f40b6fed)

## How to use?

1. Make sure that you have the `Data` folder, and you've replaced its location in the code: `./path_to_data`.

2. If you prefer a guided approach, you can make a copy of the Colab notebook called `Idea_Evaluator.ipynb`. This notebook provides a step-by-step guide on using the MOAA Advisor for evaluating circular economy proposals.

3. To run the full application in a collaborative environment, navigate to the demo file and execute `Streamlit_template.ipynb` to run the `app.py` in Colab. This allows you to experience the interactive features of the application within the Colab environment.

4. For local execution (not recommended but possible), follow these steps:
   - Install the required dependencies by running:
     ```bash
     pip install -r requirements.txt
     ```
   - Run the application using the following command in the terminal:
     ```bash
     streamlit run app.py
     ```
   This will launch the application locally, allowing you to interact with it through your web browser.

Note: Ensure that you have the necessary permissions and configurations set up for the successful execution of the application.

## What is next?
![image](https://github.com/QoutiOussama13/AI-EarthHack-MOAA-/assets/81428754/91fd170d-1296-472e-936c-e950fe7b83cd)


## Invest responsibly. Choose **MOAA Advisor** for sustainable and impactful investment decisions.
