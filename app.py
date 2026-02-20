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


# Initialize global variables
fetched_transcript = ""
processed_transcript = ""
current_url = ""


def get_video_id(url):
    pattern = r'https:\/\/www\.youtube\.com\/watch\?v=([a-zA-Z0-9_-]{11})'
    match = re.search(pattern, url)
    return match.group(1) if match else None


def get_transcript(url):
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


def chunk_transcript(processed_transcript, chunk_size=200, chunk_overlap=20):
    # Initialize the RecursiveCharacterTextSplitter with specified chunk size and overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    # Split the transcript into chunks
    chunks = text_splitter.split_text(processed_transcript)
    return chunks


def setup_credentials():
    # Define the Gemini model ID for Vertex AI
    model_id = "gemini-2.5-flash"

    # Read GCP project and location from environment variables
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT", "hale-woodland-485616-m2")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "europe-west3") # Frankfurt

    return model_id, project_id, location


def define_parameters():
    # Return a dictionary containing the parameters for the Vertex AI model
    return {
        "temperature": 0.3,
        "max_output_tokens": 900
    }


def initialize_vertexai_llm(model_id, project_id, location, parameters):
    # Create and return a VertexAI LLM instance with the specified configuration
    llm = ChatGoogleGenerativeAI(
        model=model_id,            # Gemini model to use
        project=project_id,        # GCP project ID
        location=location,          # GCP region
        **parameters               # temperature, max_output_tokens, etc.
    )
    return llm


def setup_embedding_model(project_id, location):
    # Create and return a VertexAIEmbeddings instance
    embed_model = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",  # Updated to modern Gemini embedding model string format
        project=project_id,
        location=location
    )
    return embed_model


def create_faiss_index(chunks, embedding_model):
    """Create a FAISS index from text chunks using the specified embedding model."""
    return FAISS.from_texts(chunks, embedding_model)


def retrieve(faiss_index, query, k=7):
    """Retrieve relevant context from the FAISS index based on the user's query."""
    relevant_context = faiss_index.similarity_search(query, k=k)
    return relevant_context


def create_summary_prompt():
    """Create a PromptTemplate for summarizing a YouTube video transcript."""
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
    return prompt

def create_summary_chain(llm, prompt):
    """Create an LCEL pipeline for generating summaries."""
    # NEW: Modern LCEL implementation
    return prompt | llm | StrOutputParser()


def summarize_video(url):
    """Generates a summary of the video using the preprocessed transcript."""
    global fetched_transcript, processed_transcript, current_url

    if url:
        if url != current_url:  # Only fetch if the URL has changed
            fetched_transcript = get_transcript(url)
            processed_transcript = process(fetched_transcript)
            current_url = url
    else:
        return "Please provide a valid YouTube URL."

    if processed_transcript:
        model_id, project_id, location = setup_credentials()
        llm = initialize_vertexai_llm(model_id, project_id, location, define_parameters())

        summary_prompt = create_summary_prompt()
        summary_chain = create_summary_chain(llm, summary_prompt)

        # NEW: Execute the chain using .invoke() instead of .run()
        summary = summary_chain.invoke({
            "transcript": processed_transcript
        })

        return summary
    else:
        return "No transcript available. Please fetch the transcript first."


def create_qa_prompt():
    """Create a PromptTemplate for question answering based on video content."""
    qa_template = """
    You are an expert assistant providing detailed and accurate answers based on the following video content.
    Your responses should be:
    1. Precise and free from repetition
    2. Consistent with the information provided in the video
    3. Well-organized and easy to understand
    4. Focused on addressing the user's question directly

    If you encounter conflicting information in the video content, use your best judgment to provide the most likely correct answer based on context.

    Relevant Video Context: 
    {context}

    Based on the above context, please answer the following question:
    {question}
    """
    prompt = PromptTemplate(
        input_variables=['context', 'question'],
        template=qa_template
    )
    return prompt

def create_qa_chain(llm, prompt):
    """Create an LCEL pipeline for question answering."""
    # NEW: Modern LCEL implementation
    return prompt | llm | StrOutputParser()


def generate_answer(question, faiss_index, qa_chain, k=7):
    """Retrieve relevant context and generate an answer based on user input."""
    # Retrieve relevant context (Returns a list of Document objects)
    relevant_documents = retrieve(faiss_index, question, k=k)

    # NEW: Extract the text content from the Document objects to feed into the prompt
    formatted_context = "\n\n".join(doc.page_content for doc in relevant_documents)

    # NEW: Generate answer using the LCEL .invoke() method
    answer = qa_chain.invoke({
        "context": formatted_context,
        "question": question
    })

    return answer


def answer_question(url, question):
    """Retrieves relevant context and generates an answer using the preprocessed transcript."""
    global fetched_transcript, processed_transcript, current_url
    if not processed_transcript or url != current_url:  # Fetch if no transcript or URL changed
        try:
            fetched_transcript = get_transcript(url)
            processed_transcript = process(fetched_transcript)
            current_url = url
        except:
            return "Please provide a valid YouTube URL."
        
    
    if processed_transcript and question:
        chunks = chunk_transcript(processed_transcript, 200, 20)

        model_id, project_id, location = setup_credentials()
        llm = initialize_vertexai_llm(model_id, project_id, location, define_parameters())

        embed_model = setup_embedding_model(project_id, location)
        faiss_index = create_faiss_index(chunks, embed_model)

        qa_prompt = create_qa_prompt()
        qa_chain = create_qa_chain(llm, qa_prompt)

        # Generate the answer using FAISS index
        answer = generate_answer(question, faiss_index, qa_chain)
        return answer
    else:
        return "Please provide a valid question and ensure the transcript has been fetched."



# --- Gradio Interface Setup ---
with gr.Blocks() as interface:
    gr.Markdown("<h2 style='text-align: center;'>YouTube Video Summarizer and Q&A</h2>")

    video_url = gr.Textbox(label="YouTube Video Url", placeholder="Enter the YouTube Video URL")
    summary_output = gr.Textbox(label="Video Summary", lines=6)

    question_input = gr.Textbox(label="Ask a Question about the Video", placeholder="Ask your question")
    answer_output = gr.Textbox(label="Answer to your Question", lines=6)

    summarize_btn = gr.Button("Summarize Video")
    question_btn = gr.Button("Ask a Question")

    summarize_btn.click(summarize_video, inputs=video_url, outputs=summary_output)
    question_btn.click(answer_question, inputs=[video_url, question_input], outputs=answer_output)


if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=7860)