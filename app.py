import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import os
import glob
import re
import numpy as np
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings("ignore")

# ================================
# CONFIGURACIÓN INICIAL
# ================================
st.set_page_config(
    page_title="🤖 ChatPDF IA",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ================================
# TEMA OSCURO ELEGANTE 🎨
# ================================
st.markdown("""
<style>
    /* Fondo principal oscuro */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        color: #e8e8f0;
    }
    
    /* Sidebar elegante */
    .css-1d391kg {
        background: linear-gradient(180deg, #2d2d44 0%, #3a3a5c 100%);
        border-right: 2px solid #6c63ff33;
    }
    
    /* Mensajes del chat */
    .chat-message {
        padding: 1.2rem;
        border-radius: 16px;
        margin: 0.8rem 0;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    /* Mensaje del usuario */
    .user-message {
        background: linear-gradient(135deg, #667eea44, #764ba244);
        border-left: 4px solid #a8b2ff;
        margin-left: 2rem;
    }
    
    /* Mensaje del asistente */
    .assistant-message {
        background: linear-gradient(135deg, #f093fb22, #f5576c22);
        border-left: 4px solid #ff9a9e;
        margin-right: 2rem;
    }
    
    /* Input del chat */
    .stChatInput > div > div > input {
        background: rgba(255,255,255,0.08) !important;
        border: 2px solid rgba(255,255,255,0.1) !important;
        border-radius: 25px !important;
        color: #e8e8f0 !important;
        padding: 12px 20px !important;
        backdrop-filter: blur(10px) !important;
    }
    
    .stChatInput > div > div > input:focus {
        border-color: #a8b2ff !important;
        box-shadow: 0 0 20px rgba(168,178,255,0.3) !important;
    }
    
    /* Botones elegantes */
    .stButton > button {
        background: linear-gradient(45deg, #667eea, #764ba2) !important;
        border: none !important;
        border-radius: 12px !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 0.6rem 1.5rem !important;
        transition: all 0.3s ease !important;
        backdrop-filter: blur(10px) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(102,126,234,0.4) !important;
    }
    
    /* Selectbox elegante */
    .stSelectbox > div > div {
        background: rgba(255,255,255,0.08) !important;
        border: 2px solid rgba(255,255,255,0.1) !important;
        border-radius: 12px !important;
        backdrop-filter: blur(10px) !important;
    }
    
    /* Métricas elegantes */
    .metric-card {
        background: linear-gradient(135deg, rgba(168,178,255,0.2), rgba(255,154,158,0.2));
        padding: 1.2rem;
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
        margin: 0.5rem 0;
        text-align: center;
    }
    
    /* Títulos elegantes */
    h1, h2, h3 {
        background: linear-gradient(45deg, #a8b2ff, #ff9a9e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
    
    /* Spinner personalizado */
    .stSpinner > div {
        border-top-color: #a8b2ff !important;
    }
    
    /* Eliminar padding extra */
    .block-container {
        padding-top: 2rem !important;
    }
    
    /* Ocultar elementos innecesarios */
    .stDeployButton, footer, header {
        visibility: hidden !important;
    }
    
    /* Alertas elegantes */
    .stAlert {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 12px !important;
        backdrop-filter: blur(10px) !important;
    }
</style>
""", unsafe_allow_html=True)

# ================================
# CONFIGURACIÓN DE ALMACENAMIENTO EN DISCO D
# ================================
BASE_VECTOR_PATH = "D:/chatpdf_vectorstore"
CACHE_PATH = "D:/chatpdf_cache"
TEMP_PATH = "D:/chatpdf_temp"
HF_CACHE_PATH = "D:/huggingface_cache"
TRANSFORMERS_CACHE = "D:/transformers_cache"

# 🚨 FORZAR TODO EL ALMACENAMIENTO A DISCO D
os.makedirs(BASE_VECTOR_PATH, exist_ok=True)
os.makedirs(CACHE_PATH, exist_ok=True)
os.makedirs(TEMP_PATH, exist_ok=True)
os.makedirs(HF_CACHE_PATH, exist_ok=True)
os.makedirs(TRANSFORMERS_CACHE, exist_ok=True)

# 🔥 VARIABLES DE ENTORNO CRÍTICAS - TODO A DISCO D
os.environ["HF_HOME"] = HF_CACHE_PATH
os.environ["HUGGINGFACE_HUB_CACHE"] = HF_CACHE_PATH
os.environ["TRANSFORMERS_CACHE"] = TRANSFORMERS_CACHE
os.environ["XDG_CACHE_HOME"] = CACHE_PATH
os.environ["TMPDIR"] = TEMP_PATH
os.environ["TMP"] = TEMP_PATH
os.environ["TEMP"] = TEMP_PATH

print(f"🎯 CONFIGURACIÓN DE ALMACENAMIENTO:")
print(f"📁 Vectorstore: {BASE_VECTOR_PATH}")
print(f"🔧 Cache general: {CACHE_PATH}")
print(f"🤗 HuggingFace: {HF_CACHE_PATH}")
print(f"🤖 Transformers: {TRANSFORMERS_CACHE}")
print(f"⚡ Temporales: {TEMP_PATH}")
print(f"✅ TODO EN DISCO D - DISCO C LIBRE!")

# Verificar que los directorios se crearon correctamente
for path_name, path in [
    ("Vectorstore", BASE_VECTOR_PATH),
    ("Cache", CACHE_PATH),
    ("HuggingFace", HF_CACHE_PATH),
    ("Transformers", TRANSFORMERS_CACHE),
    ("Temp", TEMP_PATH)
]:
    if os.path.exists(path):
        print(f"✅ {path_name}: {path}")
    else:
        print(f"❌ ERROR: {path_name} no se pudo crear en {path}")

# ================================
# CARGA OPTIMIZADA DE MODELOS
# ================================
@st.cache_resource
def cargar_modelos():
    """Carga modelos FORZANDO todo el almacenamiento a disco D"""
    try:
        print("🚀 Iniciando carga de modelos - TODO EN DISCO D")
        
        # 🤗 Embeddings con cache FORZADO a disco D
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={'device': 'cpu'},
            cache_folder=HF_CACHE_PATH,  # FORZADO a D:/huggingface_cache
            encode_kwargs={'normalize_embeddings': True}
        )
        print(f"✅ Embeddings cargados en: {HF_CACHE_PATH}")
        
        # 🤖 Modelo de generación con cache FORZADO a disco D
        model_name = "google/flan-t5-base"
        
        # FORZAR descarga y cache a disco D
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            cache_dir=TRANSFORMERS_CACHE,  # FORZADO a D:/transformers_cache
            local_files_only=False
        )
        
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, 
            cache_dir=TRANSFORMERS_CACHE,  # FORZADO a D:/transformers_cache
            local_files_only=False
        )
        
        print(f"✅ Modelo generativo cargado en: {TRANSFORMERS_CACHE}")
        
        generator = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=300,
            temperature=0.7,
            do_sample=True,
            top_p=0.95,
            repetition_penalty=1.15
        )
        
        print("🎯 TODOS LOS MODELOS ALMACENADOS EN DISCO D")
        return embeddings, generator, True
        
    except Exception as e:
        st.error(f"❌ Error cargando modelos: {e}")
        print(f"❌ Error detallado: {e}")
        return None, None, False

# ================================
# PROCESAMIENTO INTELIGENTE DE DOCUMENTOS
# ================================
def procesar_documento_inteligente(archivo_pdf):
    """Procesamiento avanzado FORZANDO todo el almacenamiento a disco D"""
    nombre_archivo = os.path.splitext(os.path.basename(archivo_pdf))[0]
    ruta_indice = os.path.join(BASE_VECTOR_PATH, nombre_archivo)
    
    print(f"📁 Procesando: {archivo_pdf}")
    print(f"💾 Índice se guardará en: {ruta_indice}")
    
    embeddings, _, success = cargar_modelos()
    if not success:
        return None, None
    
    # Verificar si ya existe el índice EN DISCO D
    if os.path.exists(ruta_indice):
        try:
            print(f"📂 Cargando índice existente desde: {ruta_indice}")
            vectorstore = FAISS.load_local(
                ruta_indice, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            stats = obtener_estadisticas_rapidas(vectorstore)
            print("✅ Índice cargado exitosamente desde disco D")
            return vectorstore, stats
        except Exception as e:
            print(f"⚠️ Error cargando índice existente: {e}")
            print("🔄 Re-procesando documento...")
    
    # Procesar documento con archivos temporales EN DISCO D
    with st.spinner("🔄 Procesando documento..."):
        try:
            # 📂 FORZAR que PyPDFLoader use directorio temporal en disco D
            temp_pdf_path = os.path.join(TEMP_PATH, f"temp_{nombre_archivo}.pdf")
            
            # Copiar PDF a disco D temporalmente si es necesario
            if not archivo_pdf.startswith("D:"):
                import shutil
                shutil.copy2(archivo_pdf, temp_pdf_path)
                archivo_a_procesar = temp_pdf_path
                print(f"📋 PDF copiado a disco D: {temp_pdf_path}")
            else:
                archivo_a_procesar = archivo_pdf
            
            # Cargar con PyPDFLoader
            loader = PyPDFLoader(file_path=archivo_a_procesar)
            documents = loader.load()
            print(f"📖 Cargadas {len(documents)} páginas")
            
            # Chunking inteligente y adaptativo
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=150,
                separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
                length_function=len
            )
            
            docs = text_splitter.split_documents(documents)
            print(f"✂️ Creados {len(docs)} chunks iniciales")
            
            # Filtrado inteligente de chunks
            docs_filtrados = []
            for doc in docs:
                content = doc.page_content.strip()
                
                # Filtros de calidad
                if (len(content.split()) >= 20 and  # Mínimo 20 palabras
                    len(content) >= 100 and         # Mínimo 100 caracteres
                    not re.match(r'^[\s\d\W]*$', content) and  # No solo números/símbolos
                    content.count(' ') > 10):       # Contenido real
                    docs_filtrados.append(doc)
            
            print(f"🔍 Chunks filtrados de calidad: {len(docs_filtrados)}")
            
            # Crear vectorstore y GUARDAR EN DISCO D
            print(f"🧠 Creando vectorstore en: {ruta_indice}")
            vectorstore = FAISS.from_documents(docs_filtrados, embeddings)
            vectorstore.save_local(ruta_indice)
            print(f"💾 Vectorstore GUARDADO en disco D: {ruta_indice}")
            
            # Limpiar archivo temporal si se creó
            if temp_pdf_path != archivo_a_procesar and os.path.exists(temp_pdf_path):
                os.remove(temp_pdf_path)
                print("🧹 Archivo temporal limpiado")
            
            # Estadísticas
            stats = {
                "chunks": len(docs_filtrados),
                "palabras": sum(len(doc.page_content.split()) for doc in docs_filtrados),
                "paginas": len(documents)
            }
            
            print(f"📊 Estadísticas: {stats}")
            return vectorstore, stats
            
        except Exception as e:
            st.error(f"❌ Error procesando: {e}")
            print(f"❌ Error detallado: {e}")
            return None, None

def obtener_estadisticas_rapidas(vectorstore):
    """Obtiene estadísticas básicas del vectorstore"""
    try:
        # Muestra pequeña para estadísticas
        sample_docs = vectorstore.similarity_search("", k=min(50, vectorstore.index.ntotal))
        total_words = sum(len(doc.page_content.split()) for doc in sample_docs)
        
        # Extrapolar para total aproximado
        factor = vectorstore.index.ntotal / len(sample_docs) if sample_docs else 1
        
        return {
            "chunks": vectorstore.index.ntotal,
            "palabras": int(total_words * factor),
            "paginas": int((total_words * factor) // 250)  # Aprox 250 palabras por página
        }
    except:
        return {"chunks": 0, "palabras": 0, "paginas": 0}

# ================================
# BÚSQUEDA Y RESPUESTA INTELIGENTE
# ================================
def buscar_contexto_inteligente(pregunta, vectorstore, k=6):
    """Búsqueda híbrida con filtrado de calidad"""
    try:
        # Búsqueda por similaridad
        docs_relevantes = vectorstore.similarity_search_with_score(pregunta, k=k*2)
        
        # Filtrar por score de calidad
        docs_filtrados = [
            doc for doc, score in docs_relevantes 
            if score < 1.2  # Threshold de similaridad
        ]
        
        # Tomar los mejores
        docs_finales = docs_filtrados[:k]
        
        if docs_finales:
            contexto = "\n\n".join([doc.page_content for doc in docs_finales])
            confianza = min(0.9, 1.0 - (docs_relevantes[0][1] / 2.0)) if docs_relevantes else 0.5
            return contexto, confianza
        
        return "", 0.0
        
    except Exception as e:
        st.error(f"Error en búsqueda: {e}")
        return "", 0.0

def generar_respuesta_clara(contexto, pregunta, confianza):
    """Genera respuestas claras y precisas"""
    _, generator, success = cargar_modelos()
    
    if not success or not contexto.strip():
        return "❌ No encontré información relevante para tu pregunta. Intenta reformularla."
    
    # Prompt optimizado para claridad
    prompt = f"""Basándote únicamente en el siguiente contexto, responde de manera clara y directa a la pregunta.

Contexto:
{contexto[:2000]}

Pregunta: {pregunta}

Instrucciones:
- Sé claro y directo
- Usa solo información del contexto
- Si no hay información suficiente, dilo claramente
- Evita repeticiones

Respuesta:"""
    
    try:
        respuesta = generator(
            prompt,
            max_length=250,
            min_length=30,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            repetition_penalty=1.2
        )[0]['generated_text']
        
        # Limpiar respuesta
        respuesta = respuesta.replace(prompt, "").strip()
        
        # Añadir indicador de confianza si es baja
        if confianza < 0.5:
            respuesta += "\n\n💡 *La información encontrada puede no ser completamente relevante.*"
            
        return respuesta
        
    except Exception as e:
        return f"❌ Error generando respuesta: {str(e)}"

# ================================
# GESTIÓN DE MEMORIA AUTOMÁTICA
# ================================
def limpiar_memoria_inteligente():
    """Limpieza automática e inteligente de memoria"""
    if len(st.session_state.messages) > 30:
        # Mantener solo los últimos 20 mensajes
        st.session_state.messages = st.session_state.messages[-20:]
        import gc
        gc.collect()

# ================================
# INICIALIZACIÓN DE ESTADO
# ================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "archivo_actual" not in st.session_state:
    st.session_state.archivo_actual = ""
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "stats" not in st.session_state:
    st.session_state.stats = None

# ================================
# INTERFAZ PRINCIPAL
# ================================

# Header elegante
st.markdown("""
<div style='text-align: center; padding: 1rem 0 2rem 0;'>
    <h1 style='font-size: 3rem; margin-bottom: 0.5rem;'>🤖 ChatPDF IA</h1>
    <p style='color: #a8b2ff; font-size: 1.2rem; opacity: 0.8;'>Análisis inteligente de documentos PDF</p>
</div>
""", unsafe_allow_html=True)

# Sidebar minimalista
with st.sidebar:
    st.markdown("### 📂 Documento")
    
    # Selector de archivos
    archivos_pdf = glob.glob("*.pdf")
    
    if archivos_pdf:
        archivo_seleccionado = st.selectbox(
            "Seleccionar PDF:",
            options=archivos_pdf,
            index=0,
            label_visibility="collapsed"
        )
        
        # Procesar si cambió el archivo
        if archivo_seleccionado != st.session_state.archivo_actual:
            st.session_state.archivo_actual = archivo_seleccionado
            st.session_state.messages = []
            
            # Procesar documento
            vectorstore, stats = procesar_documento_inteligente(archivo_seleccionado)
            st.session_state.vectorstore = vectorstore
            st.session_state.stats = stats
            
            if vectorstore:
                st.success("✅ Documento cargado")
        
        # Mostrar estadísticas
        if st.session_state.stats:
            st.markdown("### 📊 Estadísticas")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class='metric-card'>
                    <h4>📄 Páginas</h4>
                    <h2>{st.session_state.stats['paginas']}</h2>
                </div>
                """, unsafe_allow_html=True)
                
            with col2:
                st.markdown(f"""
                <div class='metric-card'>
                    <h4>📝 Palabras</h4>
                    <h2>{st.session_state.stats['palabras']:,}</h2>
                </div>
                """, unsafe_allow_html=True)
        
        # Botón de limpieza con verificación de espacio
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Limpiar Chat", type="secondary"):
                st.session_state.messages = []
                st.rerun()
        
        with col2:
            if st.button("💾 Ver Espacio D:", type="secondary"):
                try:
                    import shutil
                    total, used, free = shutil.disk_usage("D:/")
                    
                    st.info(f"""
                    💿 **Disco D:**
                    - Libre: {free//1024**3:.1f} GB
                    - Usado: {used//1024**3:.1f} GB
                    - Total: {total//1024**3:.1f} GB
                    """)
                except:
                    st.info("💿 Espacio en disco D disponible")
    
    else:
        st.warning("⚠️ No se encontraron archivos PDF")
        st.info("💡 Coloca archivos PDF en el directorio del script")

# Área de chat principal
if st.session_state.vectorstore and st.session_state.archivo_actual:
    
    # Limpieza automática
    limpiar_memoria_inteligente()
    
    # Mostrar conversación
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Input del usuario
    if prompt := st.chat_input("💬 Pregúntame sobre el documento..."):
        # Mostrar pregunta del usuario
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Generar respuesta
        with st.chat_message("assistant"):
            with st.spinner("🤔 Analizando..."):
                # Buscar contexto
                contexto, confianza = buscar_contexto_inteligente(
                    prompt, st.session_state.vectorstore
                )
                
                # Generar respuesta
                respuesta = generar_respuesta_clara(contexto, prompt, confianza)
                
                st.markdown(respuesta)
        
        st.session_state.messages.append({"role": "assistant", "content": respuesta})

else:
    # Pantalla de bienvenida
    st.markdown("""
    <div style='text-align: center; padding: 3rem 2rem;'>
        <div style='background: linear-gradient(135deg, rgba(168,178,255,0.1), rgba(255,154,158,0.1)); 
                    padding: 3rem; border-radius: 24px; border: 1px solid rgba(255,255,255,0.1); 
                    backdrop-filter: blur(10px);'>
            
            <h2 style='margin-bottom: 2rem;'>🚀 Bienvenido a ChatPDF IA</h2>
            
            <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); 
                        gap: 2rem; margin: 2rem 0;'>
                
                <div style='background: rgba(255,255,255,0.05); padding: 2rem; border-radius: 16px; 
                           border: 1px solid rgba(255,255,255,0.1);'>
                    <h3>🧠 IA Inteligente</h3>
                    <p>Análisis avanzado con modelos de lenguaje optimizados</p>
                </div>
                
                <div style='background: rgba(255,255,255,0.05); padding: 2rem; border-radius: 16px; 
                           border: 1px solid rgba(255,255,255,0.1);'>
                    <h3>⚡ Ultra Rápido</h3>
                    <p>Búsqueda híbrida y cache inteligente para respuestas instantáneas</p>
                </div>
                
                <div style='background: rgba(255,255,255,0.05); padding: 2rem; border-radius: 16px; 
                           border: 1px solid rgba(255,255,255,0.1);'>
                    <h3>🎯 Respuestas Precisas</h3>
                    <p>Contexto optimizado para máxima relevancia y claridad</p>
                </div>
            </div>
            
            <div style='margin-top: 3rem; padding: 2rem; background: rgba(168,178,255,0.1); 
                        border-radius: 16px; border-left: 4px solid #a8b2ff;'>
                <h3>📝 Para empezar:</h3>
                <p style='font-size: 1.1rem; line-height: 1.6;'>
                    1. Coloca un archivo PDF en el directorio<br>
                    2. Selecciónalo en la barra lateral<br>
                    3. ¡Comienza a hacer preguntas!<br><br>
                    <strong style='color: #a8b2ff;'>🎯 TODO se almacena en disco D - Tu disco C queda libre!</strong>
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Footer minimalista
st.markdown("""
<div style='text-align: center; padding: 2rem 0; opacity: 0.6; font-size: 0.9rem;'>
    Powered by Transformers & LangChain
</div>
""", unsafe_allow_html=True)