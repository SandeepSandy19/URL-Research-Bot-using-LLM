# URL-Research-Bot-using-LLM
Research Bot is a user-friendly news research tool designed for effortless information retrieval. Users can input article URLs and ask questions to receive relevant insights from the any Domain data
## Features
- Load URLs or upload text files containing URLs to fetch article content.

- Process article content through LangChain's UnstructuredURL Loader

- Construct an embedding vector using OpenAI's embeddings and leverage FAISS, a powerful similarity search library, to enable swift and effective retrieval of relevant information

- Interact with the LLM's (Chatgpt) by inputting queries and receiving answers along with source URLs.

### 1.Clone this repository to your local machine using:

'''bash
git clone https://github.com/SandeepSandy19/URL-Research-Bot-using-LLM.git
'''

### 2.Navigate to the project directory:


Install the required dependencies using pip:
'''bash
pip install -r requirements.txt
'''

3. Set up your OpenAI API key by creating a .env file in the project root and adding your API

> OPENAI_API_KEY=your_api_key_here


## Run the Streamlit app by executing:
> streamlit run main.py
