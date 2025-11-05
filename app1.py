#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 20:23:29 2025

@author: bridgetnevel
"""

"""
Digital Twin RAG System - Streamlit Interface with Image Analysis
Interactive web UI for conversing with an art historian's digital twin
Now supports image analysis using RAG context!
"""

import streamlit as st
from pathlib import Path
from typing import List, Dict, Optional
import chromadb
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings as LlamaSettings,
    PromptTemplate,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core import Settings
import os
from dotenv import load_dotenv
from PIL import Image
import requests
from io import BytesIO
import base64

load_dotenv()

# Page config
st.set_page_config(
    page_title="Digital Twin - Art Historian",
    page_icon="🎨",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .expert-response {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 10px 0;
    }
    .source-box {
        background-color: #e8f4f8;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        font-size: 0.9em;
    }
    .image-analysis-box {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


class ArtHistorianTwin:
    """Digital twin with Streamlit integration and image analysis"""
    
    def __init__(
        self,
        expert_name: str,
        expert_voice_samples: List[str],
        expert_tone: str,
        expert_style: str,
        google_api_key: str,
        chroma_persist_dir: str = "./chroma_db"
    ):
        self.expert_name = expert_name
        self.expert_voice_samples = expert_voice_samples
        self.expert_tone = expert_tone
        self.expert_style = expert_style
        self.chroma_persist_dir = chroma_persist_dir
        self.google_api_key = google_api_key
        
        # Initialize Gemini models
        self.embed_model = GoogleGenAIEmbedding(
            model_name="models/text-embedding-004",
            api_key=google_api_key
        )
        
        self.llm = GoogleGenAI(
            model="gemini-2.5-flash",
            api_key=google_api_key,
            temperature=0.7
        )
        
        # Configure LlamaIndex
        LlamaSettings.embed_model = self.embed_model
        LlamaSettings.llm = self.llm
        LlamaSettings.chunk_size = 1024
        LlamaSettings.chunk_overlap = 100
        
        self.chroma_client = None
        self.index = None
        self.query_engine = None
    
    def ingest_documents(self, documents_dir: str, force_reload: bool = False):
        """Ingest and index documents"""
        # Check if index exists
        if not force_reload and Path(self.chroma_persist_dir).exists():
            st.info("📚 Loading existing index...")
            self._load_existing_index()
            return True
        
        # Load documents
        with st.spinner("📄 Loading documents..."):
            try:
                documents = SimpleDirectoryReader(
                    documents_dir,
                    recursive=True,
                    required_exts=[".pdf", ".txt", ".md"]
                ).load_data()
                
                if not documents:
                    st.error("No documents found!")
                    return False
                
                st.success(f"✅ Loaded {len(documents)} documents")
            except Exception as e:
                st.error(f"Error loading documents: {e}")
                return False
        
        # Create text splitter
        text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=100)
        
        # Initialize Chroma
        self.chroma_client = chromadb.PersistentClient(path=self.chroma_persist_dir)
        chroma_collection = self.chroma_client.get_or_create_collection(
            name=f"{self.expert_name.replace(' ', '_').lower()}_collection"
        )
        
        # Create vector store
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Build index
        with st.spinner("🔄 Creating embeddings and building index..."):
            try:
                self.index = VectorStoreIndex.from_documents(
                    documents,
                    storage_context=storage_context,
                    transformations=[text_splitter],
                    show_progress=True
                )
                st.success(f"✅ Index created!")
            except Exception as e:
                st.error(f"Error creating index: {e}")
                return False
        
        self._setup_query_engine()
        return True
    
    def _load_existing_index(self):
        """Load existing index"""
        self.chroma_client = chromadb.PersistentClient(path=self.chroma_persist_dir)
        chroma_collection = self.chroma_client.get_or_create_collection(
            name=f"{self.expert_name.replace(' ', '_').lower()}_collection"
        )
        
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        self.index = VectorStoreIndex.from_vector_store(
            vector_store,
            embed_model=self.embed_model
        )
        
        self._setup_query_engine()
    
    def _setup_query_engine(self):
        """Setup query engine with persona"""
        voice_examples = "\n".join([
            f"    - \"{sample}\""
            for sample in self.expert_voice_samples
        ])
        
        self.llm.model = "gemini-2.5-flash"

        system_prompt = f"""You are the digital twin of {self.expert_name}, a world-renowned expert on Indian and Islamic art. Your purpose is to assess and discuss art with their voice, style, and deep knowledge.

**YOUR KNOWLEDGE:**
You will be provided with "CONTEXT" which consists of relevant passages retrieved directly from {self.expert_name}'s own writings. You MUST base your factual answers *only* on this provided context. If the context does not contain the answer, you must say, "My writings do not cover that specific detail, but I can offer some general thoughts based on my expertise."

**YOUR VOICE:**
You must adopt their specific persona.
- **Tone:** {self.expert_tone}
- **Style:** {self.expert_style}
- **Example Passages:** Here are samples of their voice:
{voice_examples}

**GUIDELINES:**
1. Always stay in character as {self.expert_name}
2. Reference the context naturally (don't say "according to the context")
3. Use technical terminology authentically
4. If expressing uncertainty, do so in their style
5. Show the depth of knowledge and passion they had for the subject

CONTEXT:
{{context_str}}

USER'S QUESTION:
{{query_str}}

YOUR RESPONSE (as {self.expert_name}):"""
        
        self.query_engine = self.index.as_query_engine(
            similarity_top_k=5,
            response_mode="compact"
        )
        
        qa_template = PromptTemplate(system_prompt)
        
        self.query_engine.update_prompts(
            {"response_synthesizer:text_qa_template": qa_template}
        )
    
    def query(self, question: str) -> Dict:
        """Query the digital twin"""
        if self.query_engine is None:
            raise ValueError("Must ingest documents first")
        
        response = self.query_engine.query(question)
        
        sources = []
        if hasattr(response, 'source_nodes'):
            for node in response.source_nodes:
                sources.append({
                    'text': node.node.text[:1000] + "...",
                    'score': node.score,
                    'metadata': node.node.metadata
                })
        
        return {
            'response': str(response),
            'sources': sources
        }
    
    def get_initial_image_description(self, image_pil: Image.Image) -> str:
        """Get initial detailed description of image for caching"""
        try:
            import google.generativeai as genai
            
            genai.configure(api_key=self.google_api_key)
            
            generation_config = {
                'temperature': 0.4,
                'max_output_tokens': 1024,
            }
            
            vision_model = genai.GenerativeModel(
                'gemini-2.5-flash',
                generation_config=generation_config
            )
            
            # Resize image if needed
            max_size = 1024
            if max(image_pil.size) > max_size:
                image_pil.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            prompt = """Provide a detailed description of this artwork covering:
- Visual elements (composition, colors, figures, objects, architectural features)
- Art style and technique (brushwork, materials, perspective)
- Cultural/regional characteristics
- Symbolic or iconographic elements
- Period indicators

Be thorough but concise (3-4 paragraphs)."""
            
            response = vision_model.generate_content(
                [prompt, image_pil],
                request_options={'timeout': 45}
            )
            
            response.resolve()
            return response.text.strip()
            
        except Exception as e:
            raise Exception(f"Error getting image description: {str(e)}")
    
    def answer_with_image_context(self, question: str, image_description: str) -> Dict:
        """Answer question using cached image description + RAG"""
        if self.query_engine is None:
            raise ValueError("Must ingest documents first")
        
        # Create RAG query from the question
        rag_query = f"{question} {' '.join(image_description.split()[:20])}"
        
        # Retrieve relevant context
        rag_response = self.query(rag_query)
        context = rag_response['response'][:2000]
        sources = rag_response['sources']
        
        # Build prompt with image context and RAG
        voice_examples = "\n".join([f"    - \"{sample[:200]}...\"" for sample in self.expert_voice_samples[:2]])
        
        prompt = f"""You are {self.expert_name}. Answer the question about this artwork.

ARTWORK DESCRIPTION:
{image_description}

RELEVANT SCHOLARLY CONTEXT:
{context}

YOUR VOICE: {self.expert_tone}
YOUR STYLE: {self.expert_style}

USER'S QUESTION: {question}

Provide a focused answer in your characteristic voice. Reference the artwork's visual elements and scholarly context naturally.

YOUR RESPONSE:"""
        
        # Use text-only LLM since we already have the image description
        try:
            from llama_index.core.llms import ChatMessage
            
            messages = [ChatMessage(role="user", content=prompt)]
            response = self.llm.chat(messages)
            
            return {
                'response': str(response.message.content),
                'sources': sources,
                'rag_query': rag_query
            }
        except Exception as e:
            raise Exception(f"Error generating response: {str(e)}")
    def analyze_image_with_rag(self, image_pil: Image.Image, user_question: str = None) -> Dict:
        """
        Initial image analysis - creates cached description
        """
        if self.query_engine is None:
            raise ValueError("Must ingest documents first")
        
        try:
            import google.generativeai as genai
            
            genai.configure(api_key=self.google_api_key)
            
            generation_config = {
                'temperature': 0.7,
                'top_p': 0.95,
                'top_k': 40,
                'max_output_tokens': 2048,
            }
            
            vision_model = genai.GenerativeModel(
                'gemini-2.5-flash',
                generation_config=generation_config
            )
            
            # Resize image if too large
            max_size = 1024
            if max(image_pil.size) > max_size:
                image_pil.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            # Step 1: Get detailed image description (will be cached)
            description_prompt = """Describe this artwork in detail covering:
- Visual elements, composition, colors
- Art style and techniques
- Cultural/regional characteristics
- Symbolic elements
Be thorough but concise (3-4 paragraphs)."""
            
            desc_response = vision_model.generate_content(
                [description_prompt, image_pil],
                request_options={'timeout': 45}
            )
            desc_response.resolve()
            image_description = desc_response.text.strip()
            
            # Step 2: Generate RAG query
            query_prompt = "In 3-5 keywords, describe: art style, period, cultural origin visible in this image."
            
            query_response = vision_model.generate_content(
                [query_prompt, image_pil],
                request_options={'timeout': 30}
            )
            query_response.resolve()
            rag_query = query_response.text.strip()
            
            # Step 3: Retrieve context from RAG
            rag_response = self.query(rag_query)
            context = rag_response['response'][:2000]
            sources = rag_response['sources']
            
            # Step 4: Generate initial analysis
            if user_question:
                analysis_prompt = f"""You are {self.expert_name}. 

ARTWORK DESCRIPTION:
{image_description}

SCHOLARLY CONTEXT:
{context}

YOUR VOICE: {self.expert_tone}

Answer this question: {user_question}

YOUR RESPONSE:"""
            else:
                analysis_prompt = f"""You are {self.expert_name}.

ARTWORK DESCRIPTION:
{image_description}

SCHOLARLY CONTEXT:
{context}

YOUR VOICE: {self.expert_tone}

Provide an art historical analysis covering: style, period, cultural context, techniques, and significance.
Write in your characteristic voice (3-4 paragraphs).

YOUR ANALYSIS:"""
            
            final_response = vision_model.generate_content(
                [analysis_prompt, image_pil],
                request_options={'timeout': 45}
            )
            final_response.resolve()
            
            return {
                'response': final_response.text,
                'rag_query': rag_query,
                'sources': sources,
                'has_image': True,
                'image_description': image_description  # Return for caching
            }
            
        except Exception as e:
            raise Exception(f"Image analysis error: {str(e)}")


def load_image_from_url(url: str) -> Image.Image:
    """Load image from URL"""
    response = requests.get(url)
    return Image.open(BytesIO(response.content))


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string"""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


# Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "twin" not in st.session_state:
    st.session_state.twin = None
if "index_built" not in st.session_state:
    st.session_state.index_built = False
if "current_image" not in st.session_state:
    st.session_state.current_image = None
if "image_context" not in st.session_state:
    st.session_state.image_context = ""


# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Key
    google_api_key = st.text_input(
        "Google API Key",
        type="password",
        help="Get your key from https://aistudio.google.com/app/apikey"
    )
    
    st.markdown("---")
    
    # Expert Configuration
    st.subheader("👤 Expert Profile")
    
    expert_name = st.text_input(
        "Expert's Name",
        value="Stewart Cary Welch",
        help="Name of the art historian"
    )
    
    expert_tone = st.text_area(
        "Expert's Tone",
        value="Formal yet passionate, professorial but accessible, deeply scholarly with occasional flashes of wit",
        height=80
    )
    
    expert_style = st.text_area(
        "Expert's Style",
        value="Uses complex, carefully constructed sentences; employs extensive art-historical terminology; frequently draws connections between artistic elements and broader cultural contexts",
        height=100
    )
    
    # Voice samples
    st.subheader("📝 Voice Samples")
    num_samples = st.number_input("Number of samples", min_value=1, max_value=5, value=2)
    v_sample = ["""Showing pictures to appreciative friends is often catalytic. One day in a corridor of the Fogg Museum, I stopped Sidney Freedberg and aired a recent Persian arrival, Sprig of Rose Blossoms (cat. 15). Sidney rose to the occasion. “A Leonardo fleshed out!” he proclaimed. The lustrously black late picture had caught my eye in Paris, where it was offered to me by Charles Ratton, the legendarily perceptive dealer specializing in tribal art, to whom I had been introduced by George Ortiz. He was also the indirect source of Two Sufis: Two Temperaments (cat. 26). After a superb luncheon with George at his parents’ house, during which I met Bruce Chatwin, who was to become a close friend, I urged Bruce to accompany me to Charles Ration’s nongallery, an apartment filled with enticing African, Pacific Island, and American Indian objects.

Monsieur Ratton, whose eyes opened widely to all art, brought out, and immediately sold to me, a superb early-seventeenth-century Mughal portrait of an infant, signed by Emperor Jahangir’s omnitalented artist Abu’l-Hasan. Ten days later, Bruce returned alone to Monsieur R’s and was shown Two Sufis. He bought it, and a few days later sold it to a mutual friend, Howard Hodgkin. Howard soon sent me a photograph, with a letter asking my opinion of the picture and of its attribution to Dawlat. Moved by the picture’s humanity and extraordinary conjunction of mystics with a metaphor-laden sarus crane devouring a fish, I replied at once, not only agreeing with the attribution, but adding that if he ever became bored by it, I should be thrilled to have it. A year or so later, this unusually intense psychological analysis of two temperamentally opposite holy men—one edging upon masochism, the other on sadism—came to me 

Trained to imbue Turkman, Safavid, and Golconda dragons with divine power, the Kotah Master, enthusiastically nurtured by the raos, his Rajput patrons, brought his talents to bear on the animals and birds of Rajasthan’s jungles and forts. How exciting it was when I soon found visual evidence of this brilliant artist’s stylistic roots in his Dragons and Birds from the Chambal River (cat. 41; fig. 6). Later more substantial proof of the Turkman appeared when I was shown this artistic wizard’s Worldly and Other-worldly Birds, painted in full flight across the ceiling of Kotah Fort’s Chhattar Mahal. Bordering it, moreover, are equally vital Golconda-style arabesques. Further proof turned up still later when I found the Kotah Master’s version of A Hero Topples a Demon, a fifteenth-century Turkman picture from Tabriz, preserved in the library of the Topkapı Sarayı Müzesi, Istanbul.""",
"""TIE FIRST FOLIOS IN THIS REmarkable
album were initiated in about I 620 by
Nuruddin Muhammad Jahangir (r. r6os-r627), the
fourth Mughal emperor of Hindus tan (Northern India).
It appears that imperial albums were designed to bring
together portraits of family, of friends, and of a few
members of rival dynasties. For greater delight specimens
of calligraphy (qui tea), other miniatures-including
a series of extraordinary natural history studies-and
illuminations were added, and all were set within
magnificent borders decorated with flowers and arabesques.
1 The emperor inscribed many of the portraits
in a hand of imperial aplomb. Intimate as one of our
own family albums, it was intended to be contemplated
in private or to be leafed through with family and very
close friends. The mood is tranquil.
When Emperor Jahangir died in r627, he was on his
way to Lahore from Kashmir, his favorite retreat, noted
for lakes, prospects of mountains, and flowers of the
sort the emperor asked artists to paint. His albums
were inherited by his son, Shahjahan (r. r628-58), who
enhanced them with further portraits, calligraphies,
and illuminations, many of which bear his elegant
nastacliq script. They became the property of his third
son, Abu'z-Zafar Muhyi'ddin Muhammad Aurangzeb,
who imprisoned Shahjahan, seized the throne, and ruled
as cAlamgir I (r6s8-1707). He made a single addition
to the Kevorkian Album: a small, black impression of
his seal stamped at the center of one of his father's
noble rosettes (shamsa) (MMA fol. 40r; pl. s).
Many royal albums left the imperial library, probably
in the early nineteenth century (sec below, "The
Royal Albums in the Nineteenth Century"). The
Kevorkian Album itself was created in about r 820, presumably
by a Delhi art dealer, who commissioned a
number of miniatures and calligraphies to supplement
the seventeenth-century originals he had obtained. The
lustrous folios were edged in crimson silk then bound
in papicr-mache decorated with floral arabesques and
birds. The album, containing no owner's or librarian's
comments tracing its history, was ready for the next
stage of its peregrinations."""]
    
    voice_samples = []
    for i in range(num_samples):
        sample = st.text_area(
            f"Sample {i+1}",
            value = v_sample[i],
            key=f"sample_{i}",
            height=100,
            help="Paste a characteristic passage from the expert's writing"
        )
        if sample:
            voice_samples.append(sample)
    
    st.markdown("---")
    
    # Documents Directory
    st.subheader("📚 Document Management")
    documents_dir = st.text_input(
        "Documents Directory",
        value="./documents",
        help="Path to directory containing expert's books (PDF/TXT)"
    )
    
    force_rebuild = st.checkbox(
        "Force Rebuild Index",
        help="Check to rebuild even if index exists"
    )
    
    # Build/Load Index Button
    if st.button("🔨 Build/Load Index", use_container_width=True):
        if not google_api_key:
            st.error("Please enter Google API Key")
        elif not voice_samples:
            st.error("Please add at least one voice sample")
        elif not Path(documents_dir).exists():
            st.error(f"Documents directory not found: {documents_dir}")
        else:
            try:
                # Initialize twin
                st.session_state.twin = ArtHistorianTwin(
                    expert_name=expert_name,
                    expert_voice_samples=voice_samples,
                    expert_tone=expert_tone,
                    expert_style=expert_style,
                    google_api_key=google_api_key
                )
                
                # Ingest documents
                success = st.session_state.twin.ingest_documents(
                    documents_dir,
                    force_reload=force_rebuild
                )
                
                if success:
                    st.session_state.index_built = True
                    st.success("✅ Ready to chat!")
                    
            except Exception as e:
                st.error(f"Error: {e}")
    
    st.markdown("---")
    
    # Reset conversation
    if st.button("🔄 Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.current_image = None
        st.session_state.image_context = ""
        st.rerun()
    
    st.markdown("---")
    st.caption("💡 **Tip:** You can now upload images or provide image URLs for analysis!")


# Main Content
st.title("🎨 Digital Twin - Art Historian")
st.markdown(f"### Conversing with {expert_name}")

if not st.session_state.index_built:
    st.info("👈 Configure the expert profile and build the index in the sidebar to begin")
    
    # Show setup instructions
    st.markdown("### 📋 Setup Instructions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **1. Prepare Documents**
        - Create a `documents/` directory
        - Add expert's books (PDF/TXT)
        - Organize in subdirectories if needed
        
        **2. Configure API**
        - Get Google API key
        - Enter in sidebar
        """)
    
    with col2:
        st.markdown("""
        **3. Set Expert Profile**
        - Enter expert's name
        - Describe their tone & style
        - Add 2-3 writing samples
        
        **4. Build Index**
        - Click "Build/Load Index"
        - Start chatting or analyzing images!
        """)
    
    st.markdown("---")
    st.markdown("### ✨ New Feature: Image Analysis")
    st.info("Upload artwork images or provide URLs, and the digital twin will analyze them using RAG-retrieved scholarly context!")

else:
    # Image Upload Section
    st.markdown("---")
    st.subheader("🖼️ Image Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        image_source = st.radio(
            "Choose image source:",
            ["Upload Image", "Image URL"],
            horizontal=True
        )
        
        uploaded_image = None
        image_url = None
        
        if image_source == "Upload Image":
            uploaded_file = st.file_uploader(
                "Upload an artwork image",
                type=['png', 'jpg', 'jpeg'],
                help="Upload an image of artwork for analysis"
            )
            if uploaded_file:
                uploaded_image = Image.open(uploaded_file)
                st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)
        
        else:
            image_url = st.text_input(
                "Enter image URL",
                placeholder="https://example.com/artwork.jpg"
            )
            if image_url:
                try:
                    uploaded_image = load_image_from_url(image_url)
                    st.image(uploaded_image, caption="Image from URL", use_container_width=True)
                except Exception as e:
                    st.error(f"Error loading image: {e}")
    
    with col2:
        if uploaded_image:
            image_question = st.text_area(
                "Specific question about the image (optional):",
                placeholder="E.g., 'What period is this from?' or 'Explain the symbolism'",
                height=100
            )
            
            if st.button("🔍 Analyze Image", use_container_width=True):
                with st.spinner(f"🎨 {expert_name} is analyzing the artwork..."):
                    try:
                        # Analyze with RAG (pass PIL image directly)
                        result = st.session_state.twin.analyze_image_with_rag(
                            uploaded_image,
                            image_question if image_question else None
                        )
                        
                        # Cache the image and its description
                        st.session_state.current_image = uploaded_image
                        st.session_state.image_context = result.get('image_description', '')
                        
                        # Add to chat history
                        st.session_state.messages.append({
                            "role": "user",
                            "content": f"[Image uploaded] {image_question if image_question else 'Please analyze this artwork'}",
                            "image": uploaded_image
                        })
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": result['response'],
                            "sources": result['sources'],
                            "rag_query": result.get('rag_query', ''),
                            "is_image_analysis": True
                        })
                        
                        st.success("✅ Image cached! You can now ask follow-up questions about this artwork.")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Analysis error: {e}")
    
    st.markdown("---")
    
    # Show active image context
    if st.session_state.current_image is not None:
        st.info("🖼️ **Active Image Context:** Questions will reference the uploaded artwork with fresh RAG lookups")
        col_img, col_clear = st.columns([3, 1])
        with col_img:
            st.image(st.session_state.current_image, caption="Current artwork in context", width=200)
        with col_clear:
            if st.button("🗑️ Clear Image Context"):
                st.session_state.current_image = None
                st.session_state.image_context = ""
                st.rerun()
    
    # Chat Interface
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                # Show image if present
                if "image" in message:
                    st.image(message["image"], width=300)
                
                st.markdown(message["content"])
                
                # Show RAG query for image analysis
                if message.get("is_image_analysis") and "rag_query" in message:
                    st.markdown(f"**🔍 RAG Query Generated:** `{message['rag_query']}`")
                
                # Show sources if available
                if "sources" in message and message["sources"]:
                    with st.expander("📚 View Retrieved Sources"):
                        for i, source in enumerate(message["sources"], 1):
                            st.markdown(f"**Source {i}** (Relevance: {source['score']:.3f})")
                            st.markdown(f"```\n{source['text']}\n```")
                            if source['metadata']:
                                st.caption(f"Metadata: {source['metadata']}")
    
    # Chat input
    if prompt := st.chat_input(f"Ask {expert_name} about {'the artwork or ' if st.session_state.current_image else ''}Islamic art..."):
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })
        
        # Generate response
        with st.spinner(f"🤔 {expert_name} is thinking..."):
            try:
                # Check if there's an active image context
                if st.session_state.current_image is not None and st.session_state.image_context:
                    # Use image context + fresh RAG lookup
                    result = st.session_state.twin.answer_with_image_context(
                        prompt,
                        st.session_state.image_context
                    )
                else:
                    # Regular text-only query
                    result = st.session_state.twin.query(prompt)
                
                # Add to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result['response'],
                    "sources": result['sources'],
                    "rag_query": result.get('rag_query', '')
                })
                
                # Force rerun to display the new messages
                st.rerun()
                
            except Exception as e:
                st.error(f"Error: {e}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"I apologize, but I encountered an error: {str(e)}"
                })

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "Powered by LlamaIndex, Chroma & Google Gemini | RAG + Vision + Persona-Based Generation"
    "</div>",
    unsafe_allow_html=True
)