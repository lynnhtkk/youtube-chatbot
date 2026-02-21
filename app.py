# Import necessary libraries for the YouTube bot
import os
import gradio as gr
import re  # For extracting video id
from youtube_transcript_api import YouTubeTranscriptApi  # For extracting transcripts from YouTube videos
from langchain_text_splitters import RecursiveCharacterTextSplitter  # For splitting text into manageable segments
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS  # For efficient vector storage and similarity search
from langchain_core.prompts import PromptTemplate  # For defining prompt templates
from langchain_core.output_parsers import StrOutputParser  # NEW: For parsing LLM outputs in LCEL


# TRANSCRIPT PROCESSING
def get_video_id(url):
    """Extract the 11-character video ID from a YouTube URL."""
    pattern = r'https:\/\/www\.youtube\.com\/watch\?v=([a-zA-Z0-9_-]{11})'
    match = re.search(pattern, url)
    return match.group(1) if match else None


def get_transcript(url):
    """Fetch the English transcript for a YouTube video, preferring manually created over auto-generated."""
    # Extracts the video ID from the URL
    video_id = get_video_id(url)

    # Create a YouTubeTranscriptApi() object
    ytt_api = YouTubeTranscriptApi()

    # Fetch the list of available transcripts for the given YouTube video
    transcripts = ytt_api.list(video_id)

    transcript = ""
    for t in transcripts:
        # Check if the transcript's language is English
        if t.language_code == 'en':
            if t.is_generated: # auto-generated
                # If no transcript has been set yet, use the auto-generated one
                if len(transcript) == 0:
                    transcript = t.fetch()
            else:
                # If a manually created transcript is found, use it (overrides auto-generated)
                transcript = t.fetch()
                break  # Prioritize the manually created transcript, exit the loop
    
    return transcript if transcript else None


def process(transcript):
    """Convert a transcript object into a plain formatted string of text and timestamps."""
    # Initialize an empty string to hold the formatted transcript
    txt = ""

    # Loop through each entry in the transcript
    for t in transcript:
        try:
            # Append the text and its start time to the output string
            txt += f"Text: {t.text}, Start: {t.start}\n"
        except:
            # If there is an issue accessing 'text' or 'start', skip this entry
            pass

    # Return the processed transcript as a single string
    return txt


def chunk_transcript(processed_transcript, chunk_size=500, chunk_overlap=20):
    """Split a processed transcript string into overlapping text chunks."""
    # Initialize the RecursiveCharacterTextSplitter with specified chunk size and overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    # Split the transcript into chunks
    chunks = text_splitter.split_text(processed_transcript)
    return chunks


# MODEL INITIALIZATION
def setup_credentials():
    """Return the Gemini model ID, GCP project ID, and location from environment variables."""
    # Define the Gemini model ID for Vertex AI
    model_id = "gemini-2.5-flash"

    # Read GCP project and location from environment variables
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT", "hale-woodland-485616-m2")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "europe-west3") # Frankfurt

    return model_id, project_id, location


def define_parameters():
    """Return a dictionary of generation parameters for the LLM."""
    # Return a dictionary containing the parameters for the Vertex AI model
    return {
        "temperature": 0.3,
        "max_output_tokens": 900
    }


def initialize_vertexai_llm():
    """Instantiate and return a ChatGoogleGenerativeAI LLM with configured credentials and parameters."""
    # Create and return a VertexAI LLM instance with the specified configuration
    model_id, project_id, location = setup_credentials()
    llm = ChatGoogleGenerativeAI(
        model=model_id,             # Gemini model to use
        project=project_id,         # GCP project ID
        location=location,          # GCP region
        **define_parameters()       # temperature, max_output_tokens, etc.
    )
    return llm


def setup_embedding_model():
    """Instantiate and return a GoogleGenerativeAIEmbeddings model for vector indexing."""
    # Create and return a VertexAIEmbeddings instance
    _, project_id, location = setup_credentials()
    embed_model = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",  # Updated to modern Gemini embedding model string format
        project=project_id,
        location=location
    )
    return embed_model


llm = initialize_vertexai_llm()
embed_model = setup_embedding_model()


# PROMPTS & CHAINS
def create_summary_chain():
    """Create an LCEL pipeline for generating summaries."""
    template = """
    You are an AI assistant tasked with summarizing YouTube video transcripts.
    Provide concise, informative summaries that capture the main points of the video content.

    Instructions:
    1. Summarize the transcript in a single concise paragraph.
    2. Ignore any timestamps in your summary.
    3. Focus on the spoken content (Text) of the video.

    Note: In the transcript, "Text" refers to the spoken words in the video, and "start" indicates the timestamp when that part begins.

    Please summarize the following YouTube video transcript:

    {transcript}
    """

    prompt = PromptTemplate(
        input_variables=['transcript'],
        template=template
    )
    
    return prompt | llm | StrOutputParser()


def create_qa_chain():
    """Create an LCEL pipeline for question answering using video context and chat history."""
    qa_template = """
    You are an expert assistant providing detailed and accurate answers based on the following video content.
    Your responses should be:
    1. Precise and free from repetition
    2. Consistent with the information provided in the video
    3. Well-organized and easy to understand
    4. Focused on addressing the user's question directly

    If you encounter conflicting information in the video content, use your best judgment to provide the most likely correct answer based on context.

    Previous Conversation History:
    {chat_history}

    Relevant Video Context: 
    {context}

    Based on the above context, please answer the following question:
    {question}
    """
    prompt = PromptTemplate(
        input_variables=['chat_history', 'context', 'question'],
        template=qa_template
    )
    return prompt | llm | StrOutputParser()


summary_chain = create_summary_chain()
qa_chain = create_qa_chain()


# APPLICATION LOGIC
def build_or_get_index(url, session_state):
    """Helper function to build FAISS index only when URL changes."""
    if url != session_state.get("current_url"): # If the url changed
        fetched_transcript = get_transcript(url)
        processed_transcript = process(fetched_transcript)

        if not processed_transcript:
            return False, "Could not fetch transcript."

        chunks = chunk_transcript(processed_transcript, 500, 20)
        faiss_index = FAISS.from_texts(chunks, embed_model)

        # Update session state with new data
        session_state['current_url'] = url
        session_state['processed_transcript'] = processed_transcript
        session_state['faiss_index'] = faiss_index
        
        return True, "Success!"
    
    return True, "Already Cached!"


def summarize_video(url, session_state):
    """Generates a summary of the video using the preprocessed transcript."""
    if not url:
        return "Please provide a valid YouTube URL."

    success, msg = build_or_get_index(url, session_state)
    if not success:
        return msg
    
    transcript = session_state['processed_transcript']
    return summary_chain.invoke({"transcript": transcript})


def chat_logic(message, history, url, session_state):
    """Retrieve relevant transcript context and generate an answer using conversation history."""
    if not url:
        return "Please provide a valid YouTube URL in the box above first."

    success, msg = build_or_get_index(url, session_state)
    if not success:
        return f"Error building knowledge base: {msg}"
    
    faiss_index = session_state['faiss_index']

    formatted_history = ""
    for turn in history:
        if turn["role"] == "user":
            formatted_history += f"User: {turn['content'][0]['text']}\n"
        elif turn["role"] == "assistant":
            formatted_history += f"Assistant: {turn['content'][0]['text']}\n"

    relevant_document = faiss_index.similarity_search(message, k=7)
    formatted_context = "\n\n".join([doc.page_content for doc in relevant_document])

    answer = qa_chain.invoke({
        "chat_history": formatted_history,
        "context": formatted_context,
        "question": message
    })

    return answer


def main():
    """Build and launch the Gradio interface."""
    with gr.Blocks() as interface:
        gr.Markdown("<h2 style='text-align: center;'>YouTube Video Summarizer and Q&A</h2>")

        session_state = gr.State({})

        # Video Summary Section
        video_url = gr.Textbox(label="YouTube Video URL", placeholder="Enter valid YouTube Video URL here.")
        summarize_btn = gr.Button("Summarize Video")

        summary_output = gr.Textbox(label="Video Summary", lines=4, interactive=False)

        summarize_btn.click(
            fn=summarize_video,
            inputs=[video_url, session_state],
            outputs=summary_output
        )

        # Q&A Section
        chat = gr.ChatInterface(
            fn=chat_logic,
            additional_inputs=[video_url, session_state]
        )

    interface.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Ocean())


if __name__ == "__main__":
    main()