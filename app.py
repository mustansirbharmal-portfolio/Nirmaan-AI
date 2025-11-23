from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
import json
import re
from typing import Dict, List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import AzureOpenAI
from azure.cognitiveservices.speech import SpeechConfig, SpeechRecognizer, AudioConfig
from azure.cosmos import CosmosClient, exceptions, PartitionKey
import logging
import uuid
from datetime import datetime
import hashlib

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Azure OpenAI client
try:
    azure_openai_client = AzureOpenAI(
        api_key=os.getenv('AZURE_OPENAI_API_KEY'),
        api_version=os.getenv('AZURE_OPENAI_API_VERSION', '2023-05-15'),
        azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT')
    )
    logger.info("Azure OpenAI client initialized successfully")
except Exception as e:
    logger.warning(f"Azure OpenAI client initialization failed: {e}")
    azure_openai_client = None

# Initialize sentence transformer for semantic similarity
try:
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("Sentence transformer model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load sentence transformer: {e}")
    sentence_model = None

# Initialize Cosmos DB client
try:
    cosmos_client = CosmosClient(
        url=os.getenv('COSMOS_ENDPOINT'),
        credential=os.getenv('COSMOS_KEY')
    )
    database = cosmos_client.get_database_client(os.getenv('COSMOS_DATABASE_NAME'))
    container = database.get_container_client(os.getenv('COSMOS_CONTAINER_NAME'))
    logger.info("Cosmos DB client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Cosmos DB client: {e}")
    cosmos_client = None
    database = None
    container = None

# Rubric configuration based on the provided images
RUBRIC_CONFIG = {
    "criteria": {
        "content_structure": {
            "weight": 40,
            "subcriteria": {
                "salutation_level": {
                    "weight": 5,
                    "keywords": ["hi", "hello", "good morning", "good afternoon", "good evening", "good day"],
                    "scoring": {
                        "no_salutation": 0,
                        "normal": 2,
                        "good": 4,
                        "excellent": 5
                    }
                },
                "key_word_presence": {
                    "weight": 30,
                    "must_have_keywords": ["name", "age", "school", "class", "family", "hobbies", "interests", "goals", "ambition", "origin", "location"],
                    "good_to_have_keywords": ["about family", "interesting thing", "fun fact", "strengths", "achievements"],
                    "scoring": {
                        "must_have_each": 4,
                        "good_to_have_each": 2
                    }
                },
                "flow_order": {
                    "weight": 5,
                    "expected_order": ["salutation", "name", "age", "class/school", "place", "additional_details", "closing"],
                    "scoring": {
                        "order_followed": 5,
                        "order_not_followed": 0
                    }
                }
            }
        },
        "speech_rate": {
            "weight": 10,
            "optimal_wpm": {"min": 111, "max": 140},
            "scoring": {
                "too_fast": {"min": 161, "score": 2},
                "fast": {"min": 141, "max": 160, "score": 6},
                "ideal": {"min": 111, "max": 140, "score": 10},
                "slow": {"min": 81, "max": 110, "score": 6},
                "too_slow": {"max": 80, "score": 2}
            }
        },
        "language_grammar": {
            "weight": 20,
            "subcriteria": {
                "grammar_errors": {
                    "weight": 10,
                    "scoring": {
                        "0.7_to_0.89": 8,
                        "0.5_to_0.69": 6,
                        "0.3_to_0.49": 4,
                        "below_0.3": 2
                    }
                },
                "vocabulary_richness": {
                    "weight": 10,
                    "ttr_scoring": {
                        "0.9_to_1.0": 10,
                        "0.7_to_0.89": 8,
                        "0.5_to_0.69": 6,
                        "0.3_to_0.49": 4,
                        "0_to_0.29": 2
                    }
                }
            }
        },
        "clarity": {
            "weight": 15,
            "filler_word_rate": {
                "excellent": {"max": 6, "score": 12},
                "good": {"min": 7, "max": 9, "score": 9},
                "average": {"min": 10, "max": 12, "score": 6},
                "poor": {"min": 13, "score": 3}
            }
        },
        "engagement": {
            "weight": 15,
            "sentiment_analysis": {
                "positive_threshold": 0.9,
                "scoring": {
                    "highly_positive": 15,
                    "positive": 12,
                    "neutral": 9,
                    "negative": 6,
                    "very_negative": 3
                }
            }
        }
    }
}

class CommunicationScorer:
    def __init__(self):
        self.rubric = RUBRIC_CONFIG
    
    def generate_azure_openai_embeddings(self, text: str) -> List[float]:
        """Generate vector embeddings using Azure OpenAI."""
        try:
            response = azure_openai_client.embeddings.create(
                model=os.getenv('AZURE_OPENAI_EMBEDDING_MODEL'),
                input=text
            )
            embeddings = response.data[0].embedding
            logger.info(f"Generated embeddings of dimension: {len(embeddings)}")
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return []
    
    def store_embeddings_in_cosmos(self, text: str, embeddings: List[float], metadata: Dict = None) -> str:
        """Store text embeddings in Azure Cosmos DB."""
        if not container:
            logger.error("Cosmos DB container not initialized")
            return None
        
        try:
            # Generate unique ID and hash for the text
            text_hash = hashlib.md5(text.encode()).hexdigest()
            doc_id = str(uuid.uuid4())
            
            # Create document
            document = {
                'id': doc_id,
                'text_hash': text_hash,
                'text': text,
                'embeddings': embeddings,
                'embedding_dimension': len(embeddings),
                'created_at': datetime.utcnow().isoformat(),
                'source': 'communication_scorer',
                'metadata': metadata or {}
            }
            
            # Store in Cosmos DB
            container.create_item(body=document)
            logger.info(f"Stored embeddings in Cosmos DB with ID: {doc_id}")
            return doc_id
            
        except exceptions.CosmosResourceExistsError:
            logger.warning(f"Document with similar content already exists")
            return None
        except Exception as e:
            logger.error(f"Error storing embeddings in Cosmos DB: {e}")
            return None
    
    def retrieve_similar_embeddings(self, query_embeddings: List[float], top_k: int = 5) -> List[Dict]:
        """Retrieve similar embeddings from Cosmos DB using cosine similarity."""
        if not container:
            logger.error("Cosmos DB container not initialized")
            return []
        
        try:
            # Query all documents (in production, you'd want to implement vector search)
            query = "SELECT * FROM c"
            items = list(container.query_items(query=query, enable_cross_partition_query=True))
            
            similarities = []
            query_embeddings = np.array(query_embeddings)
            
            for item in items:
                if 'embeddings' in item and item['embeddings']:
                    stored_embeddings = np.array(item['embeddings'])
                    
                    # Calculate cosine similarity
                    similarity = np.dot(query_embeddings, stored_embeddings) / (
                        np.linalg.norm(query_embeddings) * np.linalg.norm(stored_embeddings)
                    )
                    
                    similarities.append({
                        'document': item,
                        'similarity': float(similarity)
                    })
            
            # Sort by similarity and return top_k
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"Error retrieving similar embeddings: {e}")
            return []
    
    def process_and_store_transcript(self, text: str, metadata: Dict = None) -> Dict:
        """Process transcript, generate embeddings, and store in Cosmos DB."""
        result = {
            'success': False,
            'embeddings_id': None,
            'embeddings_dimension': 0,
            'similarity_matches': [],
            'error': None
        }
        
        try:
            # Generate embeddings using Azure OpenAI
            logger.info("Generating embeddings using Azure OpenAI...")
            embeddings = self.generate_azure_openai_embeddings(text)
            
            if not embeddings:
                result['error'] = "Failed to generate embeddings"
                return result
            
            result['embeddings_dimension'] = len(embeddings)
            
            # Store embeddings in Cosmos DB
            logger.info("Storing embeddings in Cosmos DB...")
            embeddings_id = self.store_embeddings_in_cosmos(text, embeddings, metadata)
            
            if embeddings_id:
                result['embeddings_id'] = embeddings_id
                result['success'] = True
                
                # Find similar embeddings
                logger.info("Finding similar embeddings...")
                similar_embeddings = self.retrieve_similar_embeddings(embeddings, top_k=3)
                result['similarity_matches'] = [
                    {
                        'id': match['document']['id'],
                        'similarity': match['similarity'],
                        'text_preview': match['document']['text'][:100] + '...' if len(match['document']['text']) > 100 else match['document']['text'],
                        'created_at': match['document']['created_at']
                    }
                    for match in similar_embeddings
                    if match['document']['id'] != embeddings_id  # Exclude the current document
                ]
            else:
                result['error'] = "Failed to store embeddings in Cosmos DB"
            
        except Exception as e:
            logger.error(f"Error in process_and_store_transcript: {e}")
            result['error'] = str(e)
        
        return result
        
    def calculate_words_per_minute(self, text: str, duration_seconds: int = None) -> float:
        """Calculate words per minute. If duration not provided, estimate based on average speech rate."""
        word_count = len(text.split())
        if duration_seconds:
            return (word_count / duration_seconds) * 60
        else:
            # Estimate duration based on average reading speed (150 WPM)
            estimated_duration = (word_count / 150) * 60
            return (word_count / estimated_duration) * 60
    
    def score_salutation(self, text: str) -> Tuple[int, str]:
        """Score salutation level based on keywords."""
        text_lower = text.lower()
        salutation_keywords = self.rubric["criteria"]["content_structure"]["subcriteria"]["salutation_level"]["keywords"]
        
        found_salutations = [keyword for keyword in salutation_keywords if keyword in text_lower]
        
        if not found_salutations:
            return 0, "No salutation found"
        elif len(found_salutations) == 1 and found_salutations[0] in ["hi", "hello"]:
            return 2, f"Basic salutation found: {', '.join(found_salutations)}"
        elif len(found_salutations) >= 1 and any(sal in ["good morning", "good afternoon", "good evening"] for sal in found_salutations):
            return 4, f"Good salutation found: {', '.join(found_salutations)}"
        else:
            return 5, f"Excellent salutation found: {', '.join(found_salutations)}"
    
    def score_keyword_presence(self, text: str) -> Tuple[int, str]:
        """Score based on presence of must-have and good-to-have keywords."""
        text_lower = text.lower()
        must_have = self.rubric["criteria"]["content_structure"]["subcriteria"]["key_word_presence"]["must_have_keywords"]
        good_to_have = self.rubric["criteria"]["content_structure"]["subcriteria"]["key_word_presence"]["good_to_have_keywords"]
        
        found_must_have = [kw for kw in must_have if kw in text_lower]
        found_good_to_have = [kw for kw in good_to_have if kw in text_lower]
        
        score = len(found_must_have) * 4 + len(found_good_to_have) * 2
        max_score = len(must_have) * 4 + len(good_to_have) * 2
        normalized_score = min(20, score)  # Cap at 20 points
        
        feedback = f"Must-have keywords found: {', '.join(found_must_have) if found_must_have else 'None'}\n"
        feedback += f"Good-to-have keywords found: {', '.join(found_good_to_have) if found_good_to_have else 'None'}"
        
        return normalized_score, feedback
    
    def score_speech_rate(self, text: str, duration_seconds: int = None) -> Tuple[int, str]:
        """Score speech rate based on words per minute."""
        wpm = self.calculate_words_per_minute(text, duration_seconds)
        scoring = self.rubric["criteria"]["speech_rate"]["scoring"]
        
        if wpm >= 161:
            return 2, f"Too fast: {wpm:.1f} WPM"
        elif 141 <= wpm <= 160:
            return 6, f"Fast: {wpm:.1f} WPM"
        elif 111 <= wpm <= 140:
            return 10, f"Ideal pace: {wpm:.1f} WPM"
        elif 81 <= wpm <= 110:
            return 6, f"Slow: {wpm:.1f} WPM"
        else:
            return 2, f"Too slow: {wpm:.1f} WPM"
    
    def calculate_ttr(self, text: str) -> float:
        """Calculate Type-Token Ratio for vocabulary richness."""
        words = text.lower().split()
        if not words:
            return 0
        unique_words = set(words)
        return len(unique_words) / len(words)
    
    def count_filler_words(self, text: str) -> int:
        """Count filler words in the text."""
        filler_words = ["um", "uh", "like", "you know", "so", "actually", "basically", "right", "i mean", "well", "kinda", "sort of", "okay", "hmm", "ah"]
        text_lower = text.lower()
        count = 0
        for filler in filler_words:
            count += text_lower.count(filler)
        return count
    
    def analyze_sentiment_with_azure(self, text: str) -> float:
        """Analyze sentiment using Azure OpenAI."""
        try:
            response = azure_openai_client.chat.completions.create(
                model=os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME'),
                messages=[
                    {"role": "system", "content": "You are a sentiment analyzer. Analyze the sentiment of the given text and return a score between 0 and 1, where 0 is very negative, 0.5 is neutral, and 1 is very positive. Return only the numeric score."},
                    {"role": "user", "content": text}
                ],
                max_tokens=10,
                temperature=0
            )
            
            sentiment_score = float(response.choices[0].message.content.strip())
            return max(0, min(1, sentiment_score))  # Ensure score is between 0 and 1
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return 0.5  # Default to neutral
    
    def semantic_similarity_score(self, text: str, reference_descriptions: List[str]) -> float:
        """Calculate semantic similarity between text and reference descriptions."""
        if not sentence_model:
            return 0.5  # Default score if model not available
        
        try:
            text_embedding = sentence_model.encode([text])
            ref_embeddings = sentence_model.encode(reference_descriptions)
            
            similarities = []
            for ref_emb in ref_embeddings:
                similarity = np.dot(text_embedding[0], ref_emb) / (
                    np.linalg.norm(text_embedding[0]) * np.linalg.norm(ref_emb)
                )
                similarities.append(similarity)
            
            return max(similarities) if similarities else 0.5
        except Exception as e:
            logger.error(f"Error in semantic similarity calculation: {e}")
            return 0.5
    
    def score_transcript(self, text: str, duration_seconds: int = None) -> Dict:
        """Main scoring function that combines all approaches."""
        results = {
            "overall_score": 0,
            "word_count": len(text.split()),
            "criteria_scores": {},
            "detailed_feedback": {},
            "embeddings_info": {}
        }
        
        # Process and store embeddings
        embeddings_result = self.process_and_store_transcript(
            text, 
            metadata={
                "word_count": len(text.split()),
                "duration_seconds": duration_seconds,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        results["embeddings_info"] = embeddings_result
        
        total_weighted_score = 0
        total_weight = 0
        
        # Content & Structure (40%)
        content_score = 0
        content_feedback = {}
        
        # Salutation scoring
        sal_score, sal_feedback = self.score_salutation(text)
        content_score += sal_score
        content_feedback["salutation"] = {"score": sal_score, "feedback": sal_feedback}
        
        # Keyword presence scoring
        kw_score, kw_feedback = self.score_keyword_presence(text)
        content_score += kw_score
        content_feedback["keywords"] = {"score": kw_score, "feedback": kw_feedback}
        
        # Flow order (simplified - just check if it starts with salutation)
        flow_score = 5 if any(sal in text.lower()[:50] for sal in ["hi", "hello", "good"]) else 0
        content_score += flow_score
        content_feedback["flow"] = {"score": flow_score, "feedback": "Good opening flow" if flow_score > 0 else "Consider starting with a greeting"}
        
        content_normalized = min(40, content_score)
        results["criteria_scores"]["content_structure"] = content_normalized
        # Enhanced content structure feedback with examples
        content_feedback["overall_analysis"] = self.generate_content_analysis_with_examples(text, content_score)
        results["detailed_feedback"]["content_structure"] = content_feedback
        total_weighted_score += content_normalized
        total_weight += 40
        
        # Speech Rate (10%)
        speech_score, speech_feedback = self.score_speech_rate(text, duration_seconds)
        results["criteria_scores"]["speech_rate"] = speech_score
        
        # Enhanced speech rate feedback with examples
        if duration_seconds:
            wpm = len(text.split()) / (duration_seconds / 60)
            speech_analysis = self.generate_speech_rate_analysis_with_examples(wpm, speech_score)
            results["detailed_feedback"]["speech_rate"] = speech_analysis
        else:
            results["detailed_feedback"]["speech_rate"] = {
                "score": f"{speech_score}/10 points",
                "feedback": speech_feedback,
                "note": "Provide duration for detailed speech rate analysis"
            }
        total_weighted_score += speech_score
        total_weight += 10
        
        # Language & Grammar (20%)
        ttr = self.calculate_ttr(text)
        ttr_score = 10 if ttr >= 0.9 else 8 if ttr >= 0.7 else 6 if ttr >= 0.5 else 4 if ttr >= 0.3 else 2
        
        # Simplified grammar scoring (in real implementation, use language tool)
        grammar_score = 8  # Default good score
        
        lang_total = ttr_score + grammar_score
        results["criteria_scores"]["language_grammar"] = lang_total
        
        # Enhanced language analysis with examples
        language_analysis = self.generate_language_analysis_with_examples(text, ttr, grammar_score)
        language_analysis["total_score"] = f"{lang_total}/20 points"
        language_analysis["ttr_score"] = f"{ttr_score}/10 points"
        language_analysis["grammar_score"] = f"{grammar_score}/10 points"
        results["detailed_feedback"]["language_grammar"] = language_analysis
        total_weighted_score += lang_total
        total_weight += 20
        
        # Clarity (15%)
        filler_count = self.count_filler_words(text)
        filler_rate = (filler_count / len(text.split())) * 100 if text.split() else 0
        
        if filler_rate <= 6:
            clarity_score = 12
        elif filler_rate <= 9:
            clarity_score = 9
        elif filler_rate <= 12:
            clarity_score = 6
        else:
            clarity_score = 3
        
        clarity_normalized = min(15, clarity_score)
        results["criteria_scores"]["clarity"] = clarity_normalized
        
        # Enhanced clarity feedback with examples
        clarity_analysis = self.generate_clarity_analysis_with_examples(text, filler_count, filler_rate, clarity_normalized)
        results["detailed_feedback"]["clarity"] = clarity_analysis
        total_weighted_score += clarity_normalized
        total_weight += 15
        
        # Engagement (15%)
        sentiment_score = self.analyze_sentiment_with_azure(text)
        if sentiment_score >= 0.9:
            engagement_score = 15
        elif sentiment_score >= 0.7:
            engagement_score = 12
        elif sentiment_score >= 0.5:
            engagement_score = 9
        elif sentiment_score >= 0.3:
            engagement_score = 6
        else:
            engagement_score = 3
        
        results["criteria_scores"]["engagement"] = engagement_score
        
        # Enhanced engagement feedback with examples
        engagement_analysis = self.generate_engagement_analysis_with_examples(text, sentiment_score, engagement_score)
        results["detailed_feedback"]["engagement"] = engagement_analysis
        total_weighted_score += engagement_score
        total_weight += 15
        
        # Calculate overall score
        results["overall_score"] = round((total_weighted_score / total_weight) * 100) if total_weight > 0 else 0
        
        return results
    
    def generate_content_analysis_with_examples(self, text: str, score: int) -> Dict:
        """Generate detailed content analysis with examples and suggestions."""
        analysis = {
            "score_breakdown": f"{score}/40 points",
            "performance_level": "",
            "strengths": [],
            "improvements": [],
            "examples": {}
        }
        
        # Determine performance level
        if score >= 35:
            analysis["performance_level"] = "Excellent"
            analysis["strengths"].append("Strong content structure with all key elements present")
        elif score >= 25:
            analysis["performance_level"] = "Good"
            analysis["strengths"].append("Good content structure with most key elements")
        elif score >= 15:
            analysis["performance_level"] = "Fair"
            analysis["improvements"].append("Content structure needs improvement")
        else:
            analysis["performance_level"] = "Needs Improvement"
            analysis["improvements"].append("Significant improvement needed in content structure")
        
        # Analyze specific elements with examples
        words = text.lower()
        
        # Greeting analysis
        greetings = ["hello", "hi", "good morning", "good afternoon", "good evening"]
        has_greeting = any(greeting in words for greeting in greetings)
        if has_greeting:
            analysis["strengths"].append("✓ Proper greeting used")
            analysis["examples"]["greeting"] = "Good use of greeting to start the introduction"
        else:
            analysis["improvements"].append("• Add a proper greeting (e.g., 'Hello everyone', 'Good morning')")
            analysis["examples"]["greeting_suggestion"] = "Example: 'Hello everyone, my name is...'"
        
        # Name introduction
        name_indicators = ["my name is", "i am", "myself", "i'm"]
        has_name = any(indicator in words for indicator in name_indicators)
        if has_name:
            analysis["strengths"].append("✓ Clear name introduction")
        else:
            analysis["improvements"].append("• Include clear name introduction")
            analysis["examples"]["name_suggestion"] = "Example: 'My name is [Your Name]' or 'I am [Your Name]'"
        
        # Personal details
        personal_details = ["years old", "age", "class", "grade", "school", "college", "university"]
        has_details = any(detail in words for detail in personal_details)
        if has_details:
            analysis["strengths"].append("✓ Personal details included")
        else:
            analysis["improvements"].append("• Add relevant personal details (age, education, etc.)")
            analysis["examples"]["details_suggestion"] = "Example: 'I am 20 years old and studying in...' or 'I work as...'"
        
        # Family information
        family_words = ["family", "parents", "mother", "father", "brother", "sister", "live with"]
        has_family = any(word in words for word in family_words)
        if has_family:
            analysis["strengths"].append("✓ Family information shared")
        else:
            analysis["improvements"].append("• Consider mentioning family background")
            analysis["examples"]["family_suggestion"] = "Example: 'I live with my family of four members...'"
        
        # Interests/hobbies
        interest_words = ["enjoy", "like", "love", "hobby", "interest", "passion", "play", "read", "music"]
        has_interests = any(word in words for word in interest_words)
        if has_interests:
            analysis["strengths"].append("✓ Personal interests mentioned")
        else:
            analysis["improvements"].append("• Share your interests or hobbies")
            analysis["examples"]["interests_suggestion"] = "Example: 'I enjoy playing cricket and reading books...'"
        
        # Closing
        closing_words = ["thank you", "thanks", "pleasure", "nice meeting"]
        has_closing = any(word in words for word in closing_words)
        if has_closing:
            analysis["strengths"].append("✓ Polite closing used")
        else:
            analysis["improvements"].append("• End with a polite closing")
            analysis["examples"]["closing_suggestion"] = "Example: 'Thank you for listening' or 'Nice meeting you all'"
        
        return analysis
    
    def generate_speech_rate_analysis_with_examples(self, wpm: float, score: int) -> Dict:
        """Generate detailed speech rate analysis with examples."""
        analysis = {
            "current_rate": f"{wpm:.1f} WPM",
            "optimal_range": "111-140 WPM",
            "score": f"{score}/10 points",
            "assessment": "",
            "suggestions": []
        }
        
        if 111 <= wpm <= 140:
            analysis["assessment"] = "Excellent pace - clear and easy to follow"
            analysis["suggestions"].append("✓ Maintain this natural speaking pace")
        elif 90 <= wpm < 111:
            analysis["assessment"] = "Slightly slow - consider speeding up a bit"
            analysis["suggestions"].extend([
                "• Practice speaking with more confidence",
                "• Try to reduce long pauses between words",
                "• Example: Record yourself and gradually increase pace"
            ])
        elif 140 < wpm <= 180:
            analysis["assessment"] = "Slightly fast - consider slowing down"
            analysis["suggestions"].extend([
                "• Take brief pauses between sentences",
                "• Focus on clear pronunciation",
                "• Example: Practice with a metronome or count 'one-Mississippi' between sentences"
            ])
        elif wpm > 180:
            analysis["assessment"] = "Too fast - significantly slow down"
            analysis["suggestions"].extend([
                "• Practice speaking much slower",
                "• Focus on enunciating each word clearly",
                "• Take deliberate pauses between thoughts"
            ])
        else:
            analysis["assessment"] = "Too slow - try to speak more fluently"
            analysis["suggestions"].extend([
                "• Practice speaking with more confidence",
                "• Reduce hesitation and long pauses",
                "• Prepare key points in advance"
            ])
        
        return analysis
    
    def generate_language_analysis_with_examples(self, text: str, ttr: float, grammar_score: int) -> Dict:
        """Generate detailed language and grammar analysis."""
        analysis = {
            "vocabulary_richness": f"TTR: {ttr:.3f}",
            "grammar_quality": f"{grammar_score}/10 points",
            "strengths": [],
            "improvements": [],
            "examples": {}
        }
        
        # TTR Analysis
        if ttr >= 0.7:
            analysis["strengths"].append("✓ Excellent vocabulary variety")
        elif ttr >= 0.5:
            analysis["strengths"].append("✓ Good vocabulary usage")
        else:
            analysis["improvements"].append("• Use more varied vocabulary")
            analysis["examples"]["vocabulary"] = "Instead of repeating 'good', try: excellent, wonderful, fantastic, great"
        
        # Grammar patterns analysis
        words = text.lower().split()
        
        # Check for common grammar issues
        grammar_issues = []
        
        # Subject-verb agreement (simplified check)
        if "i are" in text.lower() or "he are" in text.lower() or "she are" in text.lower():
            grammar_issues.append("Subject-verb agreement error")
            analysis["examples"]["grammar_fix"] = "Use 'I am' instead of 'I are'"
        
        # Article usage
        if any(phrase in text.lower() for phrase in ["i am student", "i work in company", "i live in house"]):
            grammar_issues.append("Missing articles")
            analysis["examples"]["articles"] = "Use 'I am a student' instead of 'I am student'"
        
        # Sentence structure
        sentences = text.split('.')
        short_sentences = [s for s in sentences if len(s.split()) < 4 and s.strip()]
        if len(short_sentences) > len(sentences) * 0.3:
            analysis["improvements"].append("• Use more complex sentence structures")
            analysis["examples"]["sentence_structure"] = "Combine: 'I like cricket. I play daily.' → 'I like cricket and play it daily.'"
        
        if grammar_issues:
            analysis["improvements"].extend([f"• Fix: {issue}" for issue in grammar_issues])
        else:
            analysis["strengths"].append("✓ Good grammar usage")
        
        return analysis
    
    def generate_clarity_analysis_with_examples(self, text: str, filler_count: int, filler_rate: float, score: int) -> Dict:
        """Generate detailed clarity analysis with examples."""
        analysis = {
            "score": f"{score}/15 points",
            "filler_words_count": filler_count,
            "filler_rate": f"{filler_rate:.1f}%",
            "assessment": "",
            "strengths": [],
            "improvements": [],
            "examples": {}
        }
        
        # Performance assessment
        if filler_rate <= 6:
            analysis["assessment"] = "Excellent clarity - very few filler words"
            analysis["strengths"].append("✓ Minimal use of filler words")
            analysis["strengths"].append("✓ Clear and fluent speech")
        elif filler_rate <= 9:
            analysis["assessment"] = "Good clarity with room for minor improvement"
            analysis["strengths"].append("✓ Generally clear speech")
            analysis["improvements"].append("• Reduce filler words slightly")
        elif filler_rate <= 12:
            analysis["assessment"] = "Fair clarity - noticeable filler words"
            analysis["improvements"].extend([
                "• Practice reducing filler words",
                "• Pause instead of using 'um', 'uh', 'like'"
            ])
        else:
            analysis["assessment"] = "Needs improvement - too many filler words"
            analysis["improvements"].extend([
                "• Significantly reduce filler words",
                "• Practice speaking more deliberately",
                "• Use pauses for thinking time"
            ])
        
        # Common filler words found
        filler_words = ["um", "uh", "like", "you know", "actually", "basically", "literally"]
        found_fillers = [word for word in filler_words if word in text.lower()]
        
        if found_fillers:
            analysis["examples"]["found_fillers"] = f"Detected: {', '.join(found_fillers)}"
            analysis["examples"]["improvement_tip"] = "Replace filler words with brief pauses or remove them entirely"
        
        # Suggestions based on score
        if score < 10:
            analysis["examples"]["practice_suggestion"] = "Practice: Record yourself speaking and count filler words. Aim for less than 5% filler rate."
        
        return analysis
    
    def generate_engagement_analysis_with_examples(self, text: str, sentiment_score: float, score: int) -> Dict:
        """Generate detailed engagement analysis with examples."""
        analysis = {
            "score": f"{score}/15 points",
            "sentiment_score": f"{sentiment_score:.2f}",
            "sentiment_range": "0.0 (negative) to 1.0 (positive)",
            "assessment": "",
            "strengths": [],
            "improvements": [],
            "examples": {}
        }
        
        # Sentiment assessment
        if sentiment_score >= 0.9:
            analysis["assessment"] = "Excellent positivity and enthusiasm"
            analysis["strengths"].extend([
                "✓ Very positive and engaging tone",
                "✓ Enthusiastic delivery"
            ])
        elif sentiment_score >= 0.7:
            analysis["assessment"] = "Good positive engagement"
            analysis["strengths"].append("✓ Generally positive and engaging")
        elif sentiment_score >= 0.5:
            analysis["assessment"] = "Neutral tone - could be more engaging"
            analysis["improvements"].append("• Add more enthusiasm and positivity")
        else:
            analysis["assessment"] = "Low engagement - needs more positivity"
            analysis["improvements"].extend([
                "• Use more positive language",
                "• Show enthusiasm about your interests",
                "• Smile while speaking (it shows in your voice)"
            ])
        
        # Analyze positive/negative words
        positive_words = ["love", "enjoy", "excited", "happy", "wonderful", "great", "amazing", "fantastic"]
        negative_words = ["hate", "dislike", "boring", "terrible", "awful", "bad"]
        
        found_positive = [word for word in positive_words if word in text.lower()]
        found_negative = [word for word in negative_words if word in text.lower()]
        
        if found_positive:
            analysis["strengths"].append(f"✓ Uses positive words: {', '.join(found_positive)}")
        
        if found_negative:
            analysis["improvements"].append(f"• Consider replacing negative words: {', '.join(found_negative)}")
            analysis["examples"]["positive_alternatives"] = "Instead of 'I don't like', try 'I prefer' or 'I'm more interested in'"
        
        # Engagement suggestions
        if score < 10:
            analysis["examples"]["engagement_tips"] = [
                "Use words like: enjoy, love, excited, passionate",
                "Share what makes you happy or proud",
                "Express enthusiasm about your goals and interests"
            ]
        
        # Check for personal connection elements
        connection_words = ["share", "connect", "meet", "together", "community", "friends"]
        has_connection = any(word in text.lower() for word in connection_words)
        
        if has_connection:
            analysis["strengths"].append("✓ Shows desire to connect with others")
        else:
            analysis["improvements"].append("• Express interest in connecting with your audience")
            analysis["examples"]["connection_suggestion"] = "Example: 'I look forward to getting to know all of you' or 'I'm excited to be part of this group'"
        
        return analysis

# Initialize scorer
scorer = CommunicationScorer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health_check():
    """Health check endpoint for deployment platforms"""
    return jsonify({
        "status": "healthy",
        "service": "Nirmaan AI Communication Scorer",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "port": os.environ.get('PORT', 'not set'),
        "environment": os.environ.get('FLASK_ENV', 'not set')
    }), 200

@app.route('/test')
def test_endpoint():
    """Simple test endpoint to verify the app is running"""
    return jsonify({
        "message": "Nirmaan AI Communication Scorer is running!",
        "status": "success",
        "timestamp": datetime.utcnow().isoformat()
    }), 200

@app.route('/api/score', methods=['POST'])
def score_transcript():
    try:
        data = request.get_json()
        transcript = data.get('transcript', '').strip()
        duration = data.get('duration_seconds')
        
        if not transcript:
            return jsonify({"error": "No transcript provided"}), 400
        
        # Score the transcript
        results = scorer.score_transcript(transcript, duration)
        
        return jsonify(results)
    
    except Exception as e:
        logger.error(f"Error scoring transcript: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/embeddings', methods=['POST'])
def generate_embeddings():
    """Generate embeddings for provided text and store in Cosmos DB."""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        # Process and store embeddings
        result = scorer.process_and_store_transcript(text, data.get('metadata', {}))
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/similar', methods=['POST'])
def find_similar():
    """Find similar texts based on embeddings."""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        top_k = data.get('top_k', 5)
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        # Generate embeddings for the query text
        embeddings = scorer.generate_azure_openai_embeddings(text)
        if not embeddings:
            return jsonify({"error": "Failed to generate embeddings"}), 500
        
        # Find similar embeddings
        similar_results = scorer.retrieve_similar_embeddings(embeddings, top_k)
        
        return jsonify({
            "query_text": text,
            "similar_texts": similar_results
        })
    
    except Exception as e:
        logger.error(f"Error finding similar texts: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/transcribe', methods=['POST'])
def transcribe_audio():
    """Transcribe audio using Azure Speech Services and generate embeddings."""
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({"error": "No audio file selected"}), 400
        
        # Get file extension
        file_extension = os.path.splitext(audio_file.filename)[1].lower()
        # Azure Speech Services natively supports these formats
        azure_supported_formats = ['.wav', '.mp3', '.ogg', '.flac', '.aac', '.mp4', '.wma']
        
        if file_extension not in azure_supported_formats:
            return jsonify({
                "error": f"Unsupported audio format: {file_extension}. Azure Speech Services supports: {', '.join(azure_supported_formats)}"
            }), 400
        
        # Save the uploaded file temporarily with original extension
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            audio_file.save(temp_file.name)
            temp_file_path = temp_file.name
        
        try:
            # Use the original file directly - Azure Speech Services handles format conversion
            audio_path = temp_file_path
            logger.info(f"Using original {file_extension} file for Azure Speech Services transcription")
            
            # Configure Azure Speech Service
            speech_config = SpeechConfig(
                subscription=os.getenv('AZURE_SPEECH_KEY'),
                region=os.getenv('AZURE_SPEECH_REGION')
            )
            speech_config.speech_recognition_language = "en-US"
            
            # Set additional properties for better recognition
            speech_config.set_property_by_name("SpeechServiceConnection_InitialSilenceTimeoutMs", "5000")
            speech_config.set_property_by_name("SpeechServiceConnection_EndSilenceTimeoutMs", "1000")
            
            # Try different approaches for audio recognition
            result = None
            
            # Method 1: Try with push stream (more robust for various formats)
            try:
                from azure.cognitiveservices.speech.audio import PushAudioInputStream, AudioStreamFormat
                
                # Read audio file as binary
                with open(audio_path, 'rb') as audio_file:
                    audio_data = audio_file.read()
                
                # Create push stream
                stream_format = AudioStreamFormat(compressed_stream_format=AudioStreamFormat.CompressedStreamFormat.MP3)
                if file_extension == '.wav':
                    stream_format = AudioStreamFormat(samples_per_second=16000, bits_per_sample=16, channels=1)
                elif file_extension == '.flac':
                    stream_format = AudioStreamFormat(compressed_stream_format=AudioStreamFormat.CompressedStreamFormat.FLAC)
                elif file_extension == '.ogg':
                    stream_format = AudioStreamFormat(compressed_stream_format=AudioStreamFormat.CompressedStreamFormat.OGG_OPUS)
                
                push_stream = PushAudioInputStream(stream_format)
                audio_config = AudioConfig(stream=push_stream)
                
                # Create recognizer
                recognizer = SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
                
                # Push audio data
                push_stream.write(audio_data)
                push_stream.close()
                
                # Perform recognition
                result = recognizer.recognize_once()
                logger.info("Used push stream method for recognition")
                
            except Exception as stream_error:
                logger.warning(f"Push stream method failed: {stream_error}")
                
                # Method 2: Fallback to file-based recognition
                try:
                    audio_config = AudioConfig(filename=audio_path)
                    recognizer = SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
                    result = recognizer.recognize_once()
                    logger.info("Used file-based method for recognition")
                    
                except Exception as file_error:
                    logger.error(f"File-based method also failed: {file_error}")
                    raise file_error
            
            if result.reason.name == 'RecognizedSpeech':
                transcript = result.text
                
                # Generate embeddings for the transcript
                embeddings_result = scorer.process_and_store_transcript(
                    transcript,
                    metadata={
                        "source": "audio_transcription",
                        "filename": audio_file.filename,
                        "original_format": file_extension,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                )
                
                return jsonify({
                    "success": True,
                    "transcript": transcript,
                    "embeddings_info": embeddings_result
                })
            elif result.reason.name == 'NoMatch':
                return jsonify({"error": "No speech could be recognized from the audio"}), 400
            elif result.reason.name == 'Canceled':
                cancellation = result.cancellation_details
                logger.error(f"Speech recognition canceled: {cancellation.reason}, {cancellation.error_details}")
                return jsonify({"error": f"Speech recognition failed: {cancellation.error_details}"}), 500
            else:
                return jsonify({"error": f"Speech recognition failed with reason: {result.reason.name}"}), 500
                
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
    
    except Exception as e:
        logger.error(f"Error transcribing audio: {e}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/api/record-and-process', methods=['POST'])
def record_and_process():
    """
    Complete flow: Record -> Azure Storage -> Speech-to-Text -> Embeddings -> Cosmos DB
    """
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({"error": "No audio file selected"}), 400
        
        # Generate unique filename based on actual file type
        import uuid
        from datetime import datetime
        
        # Detect file extension from filename or content type
        original_filename = audio_file.filename or 'recording'
        file_extension = os.path.splitext(original_filename)[1] or '.webm'
        
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        unique_id = str(uuid.uuid4())[:8]
        blob_name = f"recordings/{timestamp}_{unique_id}{file_extension}"
        
        # Step 1: Store in Azure Storage
        try:
            from azure.storage.blob import BlobServiceClient
            
            # Initialize Azure Storage client
            storage_connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
            if not storage_connection_string:
                logger.warning("Azure Storage not configured, skipping storage step")
                storage_info = {"stored": False, "reason": "Not configured"}
            else:
                blob_service_client = BlobServiceClient.from_connection_string(storage_connection_string)
                container_name = os.getenv('AZURE_STORAGE_CONTAINER', 'audio-recordings')
                
                # Upload to blob storage
                blob_client = blob_service_client.get_blob_client(
                    container=container_name, 
                    blob=blob_name
                )
                
                audio_file.seek(0)  # Reset file pointer
                blob_client.upload_blob(audio_file.read(), overwrite=True)
                
                storage_info = {
                    "stored": True,
                    "blob_name": blob_name,
                    "container": container_name,
                    "url": blob_client.url
                }
                logger.info(f"Audio stored in Azure Storage: {blob_name}")
                
        except Exception as storage_error:
            logger.warning(f"Azure Storage upload failed: {storage_error}")
            storage_info = {"stored": False, "reason": str(storage_error)}
        
        # Step 2: Save temporarily for speech processing
        audio_file.seek(0)  # Reset file pointer
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            audio_file.save(temp_file.name)
            temp_file_path = temp_file.name
        
        try:
            # Step 3: Convert to Azure-supported format and transcribe
            # WebM with Opus is supported by Azure Speech Services
            
            # Configure Azure Speech Service
            speech_config = SpeechConfig(
                subscription=os.getenv('AZURE_SPEECH_KEY'),
                region=os.getenv('AZURE_SPEECH_REGION')
            )
            speech_config.speech_recognition_language = "en-US"
            
            # Set properties for better recognition
            speech_config.set_property_by_name("SpeechServiceConnection_InitialSilenceTimeoutMs", "5000")
            speech_config.set_property_by_name("SpeechServiceConnection_EndSilenceTimeoutMs", "1000")
            
            # Process audio file (should be WAV from frontend)
            logger.info(f"Processing {file_extension} audio file for Azure Speech Services")
            
            # Use a separate scope to ensure proper cleanup
            result = None
            try:
                # Configure audio input - WAV files should work directly
                audio_config = AudioConfig(filename=temp_file_path)
                
                # Create recognizer and perform recognition
                recognizer = SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
                
                # Perform recognition
                result = recognizer.recognize_once()
                logger.info(f"Azure Speech Services recognition completed with reason: {result.reason.name}")
                
                # Explicitly cleanup to release file handles
                del recognizer
                del audio_config
                
                # Force garbage collection
                import gc
                gc.collect()
                
                # Add a small delay to ensure file handles are released
                import time
                time.sleep(0.1)
                
            except Exception as recognition_error:
                logger.error(f"Speech recognition error: {recognition_error}")
                raise recognition_error
            
            if result and result.reason.name == 'RecognizedSpeech':
                transcript = result.text
                
                # Calculate duration from file size (rough estimate)
                file_size = os.path.getsize(temp_file_path)
                duration_seconds = max(1.0, file_size / (16000 * 2))  # Rough estimate, minimum 1 second
                
                # Step 4: Generate embeddings and store in Cosmos DB
                embeddings_result = scorer.process_and_store_transcript(
                    transcript,
                    metadata={
                        "source": "web_recording",
                        "blob_name": blob_name,
                        "storage_info": storage_info,
                        "duration_seconds": duration_seconds,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                )
                
                return jsonify({
                    "success": True,
                    "transcript": transcript,
                    "duration": duration_seconds,
                    "storage_info": storage_info,
                    "embeddings_info": embeddings_result,
                    "processing_flow": {
                        "azure_storage": storage_info["stored"],
                        "speech_to_text": True,
                        "embeddings_created": embeddings_result.get("success", False),
                        "cosmos_db_stored": embeddings_result.get("success", False)
                    }
                })
                
            elif result and result.reason.name == 'NoMatch':
                return jsonify({"error": "No speech could be recognized from the recording"}), 400
            elif result and result.reason.name == 'Canceled':
                cancellation = result.cancellation_details
                logger.error(f"Speech recognition canceled: {cancellation.reason}, {cancellation.error_details}")
                return jsonify({"error": f"Speech recognition failed: {cancellation.error_details}"}), 500
            elif result:
                return jsonify({"error": f"Speech recognition failed with reason: {result.reason.name}"}), 500
            else:
                return jsonify({"error": "Speech recognition failed - no result obtained"}), 500
                
        finally:
            # Clean up temporary file with retry mechanism
            if os.path.exists(temp_file_path):
                import time
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        os.unlink(temp_file_path)
                        break
                    except PermissionError as e:
                        if attempt < max_retries - 1:
                            logger.warning(f"File deletion attempt {attempt + 1} failed: {e}. Retrying...")
                            time.sleep(0.5)  # Wait 500ms before retry
                        else:
                            logger.error(f"Failed to delete temporary file after {max_retries} attempts: {e}")
                            # Don't raise the error - just log it and continue
    
    except Exception as e:
        logger.error(f"Error in record-and-process flow: {e}")
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500

@app.route('/api/sample-embeddings', methods=['POST'])
def create_sample_embeddings():
    """Create embeddings for the sample text provided in the requirements."""
    sample_text = """Hello everyone, myself Muskan, studying in class 8th B section from Christ Public School. 
I am 13 years old. I live with my family. There are 3 people in my family, me, my mother and my father.
One special thing about my family is that they are very kind hearted to everyone and soft spoken. One thing I really enjoy is play, playing cricket and taking wickets.
A fun fact about me is that I see in mirror and talk by myself. One thing people don't know about me is that I once stole a toy from one of my cousin.
 My favorite subject is science because it is very interesting. Through science I can explore the whole world and make the discoveries and improve the lives of others. 
Thank you for listening."""
    
    try:
        # Process and store embeddings for the sample text
        result = scorer.process_and_store_transcript(
            sample_text,
            metadata={
                "source": "sample_data",
                "student_name": "Muskan",
                "school": "Christ Public School",
                "class": "8th B",
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        return jsonify({
            "message": "Sample embeddings created successfully",
            "sample_text": sample_text,
            "result": result
        })
    
    except Exception as e:
        logger.error(f"Error creating sample embeddings: {e}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
