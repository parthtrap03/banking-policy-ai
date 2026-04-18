"""
Banking Policy Intelligence Platform - Core RAG System

Upgraded from generic legal RAG to banking-specific policy intelligence:
- Semantic chunking that respects document structure
- Banking-aware metadata enrichment
- Hybrid retrieval (Vector + BM25 + Reranking)
- Conversational memory for multi-turn chat
- Query intent routing with specialized prompts
- Analytics integration
"""

import os
from dotenv import load_dotenv
load_dotenv()  # Load .env BEFORE any os.getenv() calls
import json
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime

# Environment config
from dotenv import load_dotenv
load_dotenv()

# Core RAG dependencies
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Additional tools
import chromadb

# Local modules
from hybrid_retriever import HybridRetriever
from banking_prompts import (
    INTENT_CLASSIFIER_PROMPT,
    PROMPT_REGISTRY,
    GENERAL_PROMPT,
    SUGGESTED_QUESTIONS,
    SIMPLIFY_INSTRUCTIONS,
)
from analytics import AnalyticsEngine


# ============================================================================
# PART 1: BANKING DOCUMENT PROCESSOR
# ============================================================================

class BankingDocumentProcessor:
    """Process banking policy documents with domain-specific metadata and smart chunking"""

    def __init__(self, documents_dir: str = None):
        self.documents_dir = documents_dir or os.getenv('DOCUMENTS_DIR', './legal_documents')

        # Banking-specific section classifiers
        self.banking_sections = {
            'upi_payments': r'(upi|unified.?payment|bhim|imps|neft|rtgs)',
            'transaction_limits': r'(transaction.?limit|per.?transaction|maximum|cap|ceiling)',
            'kyc_verification': r'(kyc|know.?your.?customer|customer.?due.?diligence|verification)',
            'digital_lending': r'(digital.?lend|loan|borrower|lending.?service|disbursement)',
            'data_protection': r'(data.?protect|personal.?data|consent|data.?principal|data.?fiduciary)',
            'grievance_redressal': r'(grievance|complaint|ombudsman|dispute|resolution)',
            'penalties': r'(penalty|fine|penalt|non.?compliance|breach)',
            'ppi': r'(prepaid.?payment|ppi|wallet|e.?wallet)',
            'authentication': r'(authentication|pin|biometric|two.?factor|mfa|otp)',
            'data_storage': r'(data.?storage|local[iz]ation|data.?retention|data.?sharing)',
            'payment_terms': r'(payment|fee|billing|invoice|charges)',
            'confidentiality': r'(confidential|nda|trade.?secret)',
            'liability': r'(liability|indemnify|damages|warranty)',
            'termination': r'(termination|cancel|end|renewal)',
            'governing_law': r'(governing.?law|jurisdiction|arbitration)',
            # Factoring-specific sections
            'factoring': r'(factor(ing)?|invoice.?discount|receivable|assign(ment|or|ee))',
            'treds': r'(treds|trade.?receivable.?discount|rxil|m1xchange|invoicemart)',
            'recourse_factoring': r'(recourse.?factor|non.?recourse)',
            'reverse_factoring': r'(reverse.?factor|buyer.?led|supply.?chain.?financ)',
            'international_factoring': r'(international.?factor|export.?factor|import.?factor|fci|two.?factor)',
            'msme': r'(msme|micro.?small.?medium|udyam|small.?enterprise)',
            'npa_classification': r'(npa|non.?performing.?asset|overdue|default)',
            'security_interest': r'(security.?interest|sarfaesi|collateral|pledge|lien)',
            'ucc_article9': r'(ucc|uniform.?commercial.?code|article.?9|financing.?statement)',
            'fees_charges': r'(processing.?fee|discount.?rate|service.?charge|mclr|repo.?rate)',
        }

        # Document type classifiers
        self.doc_type_patterns = {
            'RBI Circular': r'(rbi|reserve.?bank|digital.?payment.?guideline)',
            'Data Protection Act': r'(dpdp|data.?protection.?act|digital.?personal)',
            'IT Act': r'(information.?technology.?act|it.?act)',
            'Service Agreement': r'(service.?agreement|master.?service)',
            'Privacy Policy': r'(privacy.?policy)',
            'Terms of Service': r'(terms.?of.?service|terms.?and.?conditions)',
            'Factoring Regulation Act': r'(factoring.?regulat|assignment.?of.?receivable)',
            'Bank Factoring Terms': r'(bank.?factoring.?terms|factoring.?services.?terms)',
            'International Factoring Reference': r'(international.?factoring|unidroit|ucc.?article)',
        }

        # Issuing authority mapping
        self.authority_patterns = {
            'Reserve Bank of India (RBI)': r'(rbi|reserve.?bank)',
            'Parliament of India': r'(parliament|act.?no|presidential.?assent)',
            'NPCI': r'(npci|national.?payments.?corporation)',
            'Ministry of Finance': r'(ministry.?of.?finance)',
            'UNIDROIT': r'(unidroit|ottawa.?convention)',
            'US Congress (UCC)': r'(uniform.?commercial.?code|ucc)',
            'European Union': r'(eu|european.?union|eur.?lex|directive)',
            'Factors Chain International (FCI)': r'(fci|factors.?chain)',
        }

    def load_documents(self) -> List[Document]:
        """Load PDFs and text files from directory"""
        documents = []

        # Load PDFs
        pdf_loader = DirectoryLoader(
            self.documents_dir,
            glob='**/*.pdf',
            loader_cls=PyPDFLoader,
            silent_errors=True,
        )
        try:
            documents.extend(pdf_loader.load())
        except Exception:
            pass

        # Load text files
        text_loader = DirectoryLoader(
            self.documents_dir,
            glob='**/*.txt',
            loader_cls=TextLoader,
            silent_errors=True,
        )
        try:
            documents.extend(text_loader.load())
        except Exception:
            pass

        print(f"✓ Loaded {len(documents)} documents from {self.documents_dir}")
        return documents

    def enrich_metadata(self, documents: List[Document],
                        business_info: Dict = None) -> List[Document]:
        """Add banking-specific metadata to documents"""
        if business_info is None:
            business_info = {}

        for doc in documents:
            content = doc.page_content
            source_file = doc.metadata.get('source', '')

            # Classify document type
            doc.metadata['document_type'] = self._classify_doc_type(content, source_file)

            # Identify issuing authority
            doc.metadata['issuing_authority'] = self._identify_authority(content)

            # Identify all relevant banking sections
            doc.metadata['banking_sections'] = self._identify_sections(content)

            # Add business context
            doc.metadata['company'] = business_info.get('company_name', 'Banking Institution')
            doc.metadata['industry'] = business_info.get('industry', 'Banking / Financial Services')

            # Extract effective date if present
            doc.metadata['effective_date'] = self._extract_date(content)

            # Load timestamp
            doc.metadata['load_date'] = datetime.now().isoformat()

        return documents

    def _classify_doc_type(self, content: str, source_file: str = '') -> str:
        """Classify document type based on content and filename"""
        text_to_search = (content[:500] + ' ' + source_file).lower()
        for doc_type, pattern in self.doc_type_patterns.items():
            if re.search(pattern, text_to_search, re.IGNORECASE):
                return doc_type
        return 'Banking Policy Document'

    def _identify_authority(self, content: str) -> str:
        """Identify the issuing authority"""
        content_sample = content[:1000].lower()
        for authority, pattern in self.authority_patterns.items():
            if re.search(pattern, content_sample, re.IGNORECASE):
                return authority
        return 'Unknown'

    def _identify_sections(self, content: str) -> List[str]:
        """Identify banking-relevant sections in document"""
        identified = []
        for section_name, pattern in self.banking_sections.items():
            if re.search(pattern, content, re.IGNORECASE):
                identified.append(section_name)
        return identified

    def _extract_date(self, content: str) -> str:
        """Extract effective/amendment date from document"""
        date_patterns = [
            r'effective\s+date[:\s]+(\w+\s+\d{1,2},?\s+\d{4})',
            r'last\s+updated[:\s]+(\w+\s+\d{1,2},?\s+\d{4})',
            r'(\d{1,2}\s+\w+\s+\d{4})',
            r'(\w+\s+\d{4})',
        ]
        for pattern in date_patterns:
            match = re.search(pattern, content[:500], re.IGNORECASE)
            if match:
                return match.group(1)
        return 'Not specified'

    def smart_chunk(self, documents: List[Document],
                    chunk_size: int = 1000,
                    chunk_overlap: int = 200) -> List[Document]:
        """
        Semantic chunking that respects banking document structure.
        
        - Keeps sections/clauses together
        - Preserves headers for context
        - Adds contextual prepending (document + section info in each chunk)
        """
        splitter = RecursiveCharacterTextSplitter(
            separators=[
                "\n## ",          # Major section breaks (highest priority)
                "\n### ",         # Sub-section breaks
                "\n#### ",        # Sub-sub-section breaks
                "\n\n",           # Paragraph breaks
                "\n",             # Line breaks
                ". ",             # Sentence ends
                " ",              # Words
                ""                # Characters (fallback)
            ],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

        chunks = splitter.split_documents(documents)

        # Enhance each chunk with contextual metadata
        enhanced_chunks = []
        for i, chunk in enumerate(chunks):
            # Add chunk indexing
            chunk.metadata['chunk_index'] = i
            chunk.metadata['chunk_size'] = len(chunk.page_content)

            # Identify the primary banking section
            for section_name, pattern in self.banking_sections.items():
                if re.search(pattern, chunk.page_content, re.IGNORECASE):
                    chunk.metadata['section'] = section_name
                    break
            else:
                chunk.metadata['section'] = 'general'

            # Contextual prepending — add document context to chunk content
            doc_type = chunk.metadata.get('document_type', 'Document')
            section = chunk.metadata.get('section', 'general').replace('_', ' ').title()
            authority = chunk.metadata.get('issuing_authority', 'Unknown')

            context_header = f"[Source: {doc_type} | Section: {section} | Authority: {authority}]\n\n"
            chunk.page_content = context_header + chunk.page_content

            enhanced_chunks.append(chunk)

        print(f"✓ Created {len(enhanced_chunks)} smart chunks (avg {chunk_size} chars)")
        return enhanced_chunks


# ============================================================================
# PART 2: VECTOR STORE MANAGER
# ============================================================================

class HybridStoreManager:
    """Manage vector storage with upgraded embeddings"""

    def __init__(self, persist_dir: str = None):
        self.persist_dir = persist_dir or os.getenv('CHROMA_PERSIST_DIR', './chroma_legal_db')
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)

    def get_embeddings(self, model_name: str = None):
        """Get embedding model instance"""
        model_name = model_name or os.getenv('EMBEDDING_MODEL', 'BAAI/bge-large-en-v1.5')

        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True},
        )
        print(f"✓ Using embeddings: {model_name}")
        return embeddings

    def create_vector_store(self, chunks: List[Document],
                            embedding_model: str = None) -> Chroma:
        """Create and persist vector store"""
        embeddings = self.get_embeddings(embedding_model)

        # Clear existing data for clean rebuild
        if Path(self.persist_dir).exists():
            import shutil
            shutil.rmtree(self.persist_dir, ignore_errors=True)
            Path(self.persist_dir).mkdir(parents=True, exist_ok=True)

        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=self.persist_dir,
            collection_name='banking_policies',
        )

        vectordb.persist()
        print(f"✓ Vector store created at {self.persist_dir} ({len(chunks)} chunks)")
        return vectordb

    def load_vector_store(self, embedding_model: str = None) -> Chroma:
        """Load existing vector store"""
        embeddings = self.get_embeddings(embedding_model)

        vectordb = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=embeddings,
            collection_name='banking_policies',
        )

        print(f"✓ Loaded vector store from {self.persist_dir}")
        return vectordb


# ============================================================================
# PART 3: BANKING POLICY RAG ENGINE
# ============================================================================

class BankingPolicyRAG:
    """
    Banking policy RAG with:
    - Query intent classification and routing
    - Hybrid retrieval (vector + BM25 + reranking)
    - Conversational memory
    - Analytics integration
    """

    def __init__(
        self,
        vectordb: Chroma,
        chunks: List[Document] = None,
        llm_model: str = None,
        enable_analytics: bool = True,
    ):
        self.vectordb = vectordb
        self.chunks = chunks or []
        self.chat_history: List[Dict] = []
        self.max_history = 5

        # Initialize LLM
        llm_model = llm_model or os.getenv('LLM_MODEL', 'llama-3.1-8b-instant')
        api_key = os.getenv('GROQ_API_KEY')

        self.llm = ChatGroq(
            model=llm_model,
            api_key=api_key,
            temperature=float(os.getenv('LLM_TEMPERATURE', '0.2')),
            max_tokens=int(os.getenv('LLM_MAX_TOKENS', '600')),
        )
        print(f"✓ LLM initialized: Groq ({llm_model})")

        # Initialize hybrid retriever if chunks available
        if self.chunks:
            self.retriever = HybridRetriever(
                vectordb=vectordb,
                chunks=chunks,
                enable_bm25=True,
                enable_reranker=True,
            )
        else:
            self.retriever = None
            print("⚠ No chunks provided. Using vector-only retrieval.")

        # Initialize analytics
        self.analytics = AnalyticsEngine() if enable_analytics else None

    def _classify_intent(self, question: str) -> str:
        """Classify query intent using fast regex rules (no LLM call = saves 3-5s)"""
        q = question.lower().strip()

        # COMPARISON — contains comparison keywords
        if re.search(r'(differ|compar|vs\.?|versus|between .+ and)', q):
            return 'COMPARISON'

        # PROCEDURAL — asks "how to" / "steps" / "process"
        if re.search(r'(how (do|can|does|to|is)|step|process|procedure|file a|apply for|register)', q):
            return 'PROCEDURAL'

        # COMPLIANCE — mentions obligation / penalty / requirement
        if re.search(r'(must|required|obligat|comply|complian|penalty|penalt|mandatory|prohibit|allowed|permitted|legal)', q):
            return 'COMPLIANCE'

        # FACTUAL — asks "what is" / "how much" / specific numbers
        if re.search(r'(what (is|are|was|were)|how (much|many|often|long)|limit|fee|charge|rate|amount|list|define|meaning)', q):
            return 'FACTUAL'

        return 'GENERAL'

    def _get_chat_history_text(self) -> str:
        """Format recent chat history for context"""
        if not self.chat_history:
            return "No previous conversation."

        history_lines = []
        for entry in self.chat_history[-self.max_history:]:
            history_lines.append(f"User: {entry['question']}")
            history_lines.append(f"Assistant: {entry['answer'][:200]}...")
            history_lines.append("")

        return "\n".join(history_lines)

    def _extract_confidence(self, answer: str) -> str:
        """Extract confidence level from the answer text"""
        answer_lower = answer.lower()
        if 'confidence: high' in answer_lower or '**high**' in answer_lower:
            return 'High'
        elif 'confidence: low' in answer_lower or '**low**' in answer_lower:
            return 'Low'
        elif "don't have enough information" in answer_lower:
            return 'Low'
        return 'Medium'

    def query(self, question: str, k: int = None, example_style: str = 'None') -> Dict:
        """
        Full query pipeline:
        1. Classify intent
        2. Retrieve relevant chunks (hybrid)
        3. Select appropriate prompt template
        4. Append simplification instructions based on example_style
        5. Generate answer with citations
        6. Log to analytics
        """
        start_time = time.time()
        k = k or int(os.getenv('RETRIEVAL_K', '5'))

        # Step 1: Classify intent (instant regex, no LLM call)
        intent = self._classify_intent(question)

        # Step 2: Retrieve documents
        if self.retriever:
            initial_k = int(os.getenv('RERANK_INITIAL_K', '12'))
            retrieval_results = self.retriever.retrieve(question, k=k, initial_k=initial_k)
            source_docs = [r['document'] for r in retrieval_results]
            retrieval_method = self.retriever.get_retrieval_stats()['retrieval_method']
        else:
            # Fallback: vector-only
            retriever = self.vectordb.as_retriever(search_kwargs={'k': k})
            source_docs = retriever.invoke(question)
            retrieval_results = [
                {
                    'document': doc,
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'relevance_score': 0.5,
                    'document_type': doc.metadata.get('document_type', 'Unknown'),
                    'section': doc.metadata.get('section', 'General'),
                    'source_file': doc.metadata.get('source', 'Unknown'),
                    'issuing_authority': doc.metadata.get('issuing_authority', 'Unknown'),
                }
                for doc in source_docs
            ]
            retrieval_method = 'Vector Only'

        # Step 3: Build context
        context = "\n\n---\n\n".join(doc.page_content for doc in source_docs)

        # Step 4: Select prompt and generate answer
        prompt_template = PROMPT_REGISTRY.get(intent, GENERAL_PROMPT)
        chat_history = self._get_chat_history_text()

        formatted_prompt = prompt_template.format(
            context=context,
            question=question,
            chat_history=chat_history,
        )

        # Append simplification / example-style instructions
        simplify_instruction = SIMPLIFY_INSTRUCTIONS.get(example_style, '')
        if simplify_instruction:
            formatted_prompt += simplify_instruction

        response = self.llm.invoke(formatted_prompt)
        answer = response.content

        # Step 5: Extract confidence
        confidence = self._extract_confidence(answer)

        # Step 6: Process sources
        sources = []
        for r in retrieval_results:
            content_preview = r['content']
            # Remove contextual header from preview
            if content_preview.startswith('[Source:'):
                content_preview = content_preview.split(']\n\n', 1)[-1]

            sources.append({
                'content': content_preview[:300] + '...' if len(content_preview) > 300 else content_preview,
                'metadata': r['metadata'],
                'document_type': r['document_type'],
                'section': r.get('section', 'General').replace('_', ' ').title(),
                'issuing_authority': r.get('issuing_authority', 'Unknown'),
                'source_file': r.get('source_file', 'Unknown'),
                'relevance_score': r.get('relevance_score', 0),
            })

        # Step 7: Update conversation memory
        self.chat_history.append({
            'question': question,
            'answer': answer,
            'intent': intent,
        })

        # Calculate response time
        response_time_ms = int((time.time() - start_time) * 1000)

        # Step 8: Log to analytics
        query_id = None
        if self.analytics:
            query_id = self.analytics.log_query(
                question=question,
                answer=answer,
                sources=sources,
                intent_category=intent,
                response_time_ms=response_time_ms,
                confidence=confidence,
                retrieval_method=retrieval_method,
            )

        return {
            'answer': answer,
            'sources': sources,
            'num_retrieved': len(sources),
            'intent': intent,
            'confidence': confidence,
            'response_time_ms': response_time_ms,
            'retrieval_method': retrieval_method,
            'query_id': query_id,
        }

    def submit_feedback(self, query_id: int, rating: int, comment: str = '') -> None:
        """Submit feedback for a query (1 = thumbs up, -1 = thumbs down)"""
        if self.analytics and query_id:
            self.analytics.log_feedback(query_id, rating, comment)

    def clear_history(self):
        """Clear conversation memory"""
        self.chat_history = []

    def get_suggested_questions(self) -> Dict:
        """Return banking-specific suggested questions"""
        return SUGGESTED_QUESTIONS


# ============================================================================
# PART 4: TESTING & EVALUATION
# ============================================================================

class BankingRAGEvaluator:
    """Test and evaluate RAG system with banking-specific questions"""

    def __init__(self, rag: BankingPolicyRAG):
        self.rag = rag
        self.test_results = []

    def run_test_suite(self, test_questions: List[str] = None) -> Dict:
        """Run comprehensive banking test suite"""

        if test_questions is None:
            test_questions = self._banking_test_questions()

        print("\n" + "=" * 70)
        print("BANKING POLICY RAG - EVALUATION SUITE")
        print("=" * 70)

        for i, question in enumerate(test_questions, 1):
            print(f"\n[Test {i}/{len(test_questions)}]")
            print(f"Question: {question}\n")

            result = self.rag.query(question, k=5)

            print(f"Intent: {result['intent']}")
            print(f"Confidence: {result['confidence']}")
            print(f"Response Time: {result['response_time_ms']}ms")
            print(f"Retrieval: {result['retrieval_method']}")
            print(f"\nAnswer:\n{result['answer'][:300]}...\n")
            print(f"Retrieved {result['num_retrieved']} sources:")

            for j, source in enumerate(result['sources'][:3], 1):
                print(f"  [{j}] {source['document_type']} | "
                      f"Section: {source['section']} | "
                      f"Authority: {source['issuing_authority']}")

            self.test_results.append({
                'question': question,
                'answer': result['answer'],
                'intent': result['intent'],
                'confidence': result['confidence'],
                'num_sources': result['num_retrieved'],
                'response_time_ms': result['response_time_ms'],
                'retrieval_method': result['retrieval_method'],
                'timestamp': datetime.now().isoformat(),
            })

            print("\n" + "-" * 70)

        return self._generate_report()

    def _banking_test_questions(self) -> List[str]:
        """Banking-specific test questions covering all document types"""
        return [
            # UPI & Digital Payments
            "What is the UPI transaction limit for person-to-person transfers?",
            "What are the UPI Lite wallet balance and per-transaction limits?",
            "How does the grievance redressal framework work for UPI complaints?",

            # KYC
            "What documents are accepted as Officially Valid Documents for KYC?",
            "How often must KYC be updated for high-risk customers?",

            # DPDP Act
            "What are the penalties for failure to prevent a data breach under DPDP Act?",
            "What consent requirements exist for processing personal data?",
            "Can personal data be transferred outside India?",

            # Digital Lending
            "What must a Key Fact Statement contain for digital loans?",
            "What is the FLDG cap for lending service providers?",

            # Service Agreement
            "What are the payment terms in the service agreement?",
            "How can the agreement be terminated?",

            # Cross-document
            "What is the liability framework for unauthorized digital transactions?",
        ]

    def _generate_report(self) -> Dict:
        """Generate evaluation report"""
        if not self.test_results:
            return {'total_tests': 0}

        report = {
            'total_tests': len(self.test_results),
            'timestamp': datetime.now().isoformat(),
            'results': self.test_results,
            'summary': {
                'avg_sources_retrieved': sum(r['num_sources'] for r in self.test_results) / len(self.test_results),
                'avg_response_time_ms': sum(r['response_time_ms'] for r in self.test_results) / len(self.test_results),
                'confidence_distribution': {
                    'High': sum(1 for r in self.test_results if r['confidence'] == 'High'),
                    'Medium': sum(1 for r in self.test_results if r['confidence'] == 'Medium'),
                    'Low': sum(1 for r in self.test_results if r['confidence'] == 'Low'),
                },
                'intent_distribution': {},
            }
        }

        # Count intents
        for r in self.test_results:
            intent = r['intent']
            report['summary']['intent_distribution'][intent] = \
                report['summary']['intent_distribution'].get(intent, 0) + 1

        return report

    def save_report(self, filename: str = 'banking_rag_eval_report.json'):
        """Save evaluation report to file"""
        report = self._generate_report()
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n✓ Report saved to {filename}")


# ============================================================================
# PART 5: MAIN EXECUTION
# ============================================================================

def setup_business_context() -> Dict:
    """Define your banking business context"""
    return {
        'company_name': 'Banking Policy Intelligence',
        'industry': 'Banking / Financial Services',
        'jurisdiction': 'India',
        'regulatory_bodies': ['RBI', 'NPCI', 'MeitY'],
    }


def build_system(rebuild_db: bool = True) -> Tuple[BankingPolicyRAG, List[Document]]:
    """
    Build the complete system: load docs → chunk → embed → create RAG.
    
    Args:
        rebuild_db: If True, rebuild the vector database from scratch
    
    Returns:
        Tuple of (BankingPolicyRAG instance, list of chunks)
    """
    print("Banking Policy Intelligence Platform - System Build\n")

    # Step 1: Prepare documents
    print("=" * 70)
    print("STEP 1: DOCUMENT PREPARATION")
    print("=" * 70)

    processor = BankingDocumentProcessor()
    business_info = setup_business_context()

    documents = processor.load_documents()
    documents = processor.enrich_metadata(documents, business_info)
    chunks = processor.smart_chunk(documents)

    # Step 2: Create/load vector store
    print("\n" + "=" * 70)
    print("STEP 2: VECTOR STORE")
    print("=" * 70)

    store_manager = HybridStoreManager()

    if rebuild_db:
        vectordb = store_manager.create_vector_store(chunks)
    else:
        vectordb = store_manager.load_vector_store()

    # Step 3: Initialize RAG
    print("\n" + "=" * 70)
    print("STEP 3: RAG ENGINE INITIALIZATION")
    print("=" * 70)

    rag = BankingPolicyRAG(vectordb=vectordb, chunks=chunks)

    print("\n" + "=" * 70)
    print("✓ System Ready!")
    print("=" * 70)

    return rag, chunks


def main():
    """Complete workflow: Build → Test → Report"""
    rag, chunks = build_system(rebuild_db=True)

    # Run evaluation
    print("\n" + "=" * 70)
    print("STEP 4: EVALUATION")
    print("=" * 70)

    evaluator = BankingRAGEvaluator(rag)
    report = evaluator.run_test_suite()
    evaluator.save_report('banking_rag_eval_report.json')

    print(f"\nEvaluation Summary:")
    print(f"  Total tests: {report['summary'].get('avg_sources_retrieved', 'N/A')}")
    print(f"  Avg response time: {report['summary'].get('avg_response_time_ms', 'N/A')}ms")
    print(f"  Confidence: {report['summary'].get('confidence_distribution', {})}")

    print(f"\nTo start the interactive UI:")
    print(f"  streamlit run app.py")


if __name__ == '__main__':
    main()
