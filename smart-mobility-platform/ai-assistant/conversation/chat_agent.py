"""
Conversational AI Agent for Smart Mobility Platform
RAG-powered chatbot for operators and stakeholders
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import json
import re

# LangChain components
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.callbacks import AsyncCallbackHandler

# Internal imports
from ..rag.knowledge_base import MobilityKnowledgeBase

logger = logging.getLogger(__name__)

class MobilityCallbackHandler(AsyncCallbackHandler):
    """Custom callback handler for mobility chat agent"""
    
    def __init__(self):
        self.conversation_log = []
    
    async def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        """Called when LLM starts running"""
        logger.info("LLM processing started")
    
    async def on_llm_end(self, response, **kwargs: Any) -> None:
        """Called when LLM ends running"""
        logger.info("LLM processing completed")
    
    async def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        """Called when LLM encounters an error"""
        logger.error(f"LLM error: {error}")

class MobilityChatAgent:
    """Conversational AI agent for smart mobility platform"""
    
    def __init__(self, 
                 knowledge_base: MobilityKnowledgeBase,
                 openai_api_key: str = None,
                 model_name: str = "gpt-3.5-turbo",
                 temperature: float = 0.3,
                 max_tokens: int = 1000):
        
        self.knowledge_base = knowledge_base
        self.openai_api_key = openai_api_key
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize components
        self.llm = None
        self.conversation_chain = None
        self.memory = None
        self.callback_handler = MobilityCallbackHandler()
        
        # Conversation context
        self.user_sessions = {}  # Store user conversation sessions
        
        # Intent classification patterns
        self.intent_patterns = {
            'system_status': [
                r'what.*status', r'how.*system', r'current.*condition',
                r'system.*health', r'is.*working', r'performance'
            ],
            'prediction_query': [
                r'predict.*', r'forecast.*', r'expect.*', r'will.*happen',
                r'future.*', r'tomorrow.*', r'next.*hour'
            ],
            'anomaly_inquiry': [
                r'anomal.*', r'unusual.*', r'problem.*', r'issue.*',
                r'alert.*', r'warning.*', r'error.*'
            ],
            'optimization_request': [
                r'optimize.*', r'improve.*', r'better.*', r'efficient.*',
                r'schedule.*', r'dispatch.*'
            ],
            'historical_analysis': [
                r'history.*', r'past.*', r'previous.*', r'trend.*',
                r'analytics.*', r'report.*'
            ],
            'operational_guidance': [
                r'how.*do', r'procedure.*', r'guideline.*', r'best.*practice',
                r'should.*', r'recommend.*'
            ]
        }
        
        # System prompts
        self.system_prompts = {
            'general': """You are an AI assistant for a Smart Mobility Platform. You help transportation operators, city planners, and stakeholders with:

1. Real-time system monitoring and status updates
2. Predictive analytics and forecasting
3. Anomaly detection and incident response
4. Operational procedures and best practices
5. Performance optimization recommendations
6. Historical data analysis and insights

You have access to:
- Real-time vehicle and passenger data
- Historical transportation patterns
- System performance metrics
- Operational procedures and guidelines
- Predictive models for demand and congestion

Always provide accurate, actionable information. If you're unsure about something, say so and suggest how to get the needed information. Focus on practical solutions and operational excellence.""",
            
            'operator': """You are an AI assistant specifically designed for transportation operators. Your role is to:

1. Provide real-time operational insights
2. Alert about anomalies and incidents
3. Recommend immediate actions for issues
4. Guide through operational procedures
5. Optimize day-to-day operations

Keep responses concise and action-oriented. Always prioritize safety and service reliability.""",
            
            'analyst': """You are an AI assistant for data analysts and city planners working with transportation data. You help with:

1. Data analysis and interpretation
2. Trend identification and forecasting
3. Performance measurement and KPIs
4. Strategic planning insights
5. Report generation and visualization

Provide detailed analytical insights with supporting data when available."""
        }
    
    async def initialize(self):
        """Initialize the chat agent"""
        try:
            logger.info("Initializing chat agent...")
            
            # Initialize LLM
            if self.openai_api_key:
                self.llm = ChatOpenAI(
                    openai_api_key=self.openai_api_key,
                    model_name=self.model_name,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    callbacks=[self.callback_handler]
                )
            else:
                # Use a mock LLM for development
                self.llm = self._create_mock_llm()
            
            # Initialize memory
            self.memory = ConversationBufferWindowMemory(
                k=10,  # Keep last 10 exchanges
                return_messages=True
            )
            
            logger.info("Chat agent initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize chat agent: {e}")
            raise
    
    def _create_mock_llm(self):
        """Create mock LLM for development/testing"""
        class MockLLM:
            def __init__(self):
                self.responses = {
                    'system_status': "The smart mobility system is currently operating normally. All key components are functional with 98.5% uptime. There are no critical alerts at this time.",
                    'prediction_query': "Based on current patterns, I predict moderate congestion during evening rush hour (5-7 PM) with peak demand on Route A. Passenger volume is expected to increase by 15% compared to yesterday.",
                    'anomaly_inquiry': "I've detected 2 minor anomalies in the past hour: unusual passenger surge at Central Station (+40% above normal) and slight delay on Route B (5 minutes behind schedule). Both are being monitored.",
                    'optimization_request': "To optimize current operations, I recommend: 1) Deploy additional vehicle on Route A, 2) Adjust schedule frequency during peak hours, 3) Implement dynamic pricing to distribute demand.",
                    'historical_analysis': "Historical analysis shows consistent patterns: 25% higher ridership on Fridays, peak congestion between 8-9 AM and 5-6 PM, and weather-related 15% demand increase during rain.",
                    'operational_guidance': "For handling high congestion: 1) Immediately assess affected routes, 2) Deploy additional vehicles if available, 3) Update passenger information systems, 4) Consider alternative routing options.",
                    'default': "I'm here to help with your smart mobility platform questions. I can provide information about system status, predictions, anomalies, optimization recommendations, and operational guidance."
                }
            
            async def apredict(self, prompt):
                # Simple intent detection based on keywords
                prompt_lower = prompt.lower()
                
                for intent, patterns in self.intent_patterns.items():
                    if any(re.search(pattern, prompt_lower) for pattern in patterns):
                        return self.responses.get(intent, self.responses['default'])
                
                return self.responses['default']
            
            def predict(self, prompt):
                return asyncio.run(self.apredict(prompt))
        
        return MockLLM()
    
    async def chat(self, message: str, 
                   user_id: str = "default",
                   user_role: str = "general",
                   context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a chat message and return response"""
        try:
            # Get or create user session
            session = self._get_user_session(user_id, user_role)
            
            # Classify intent
            intent = await self._classify_intent(message)
            
            # Get relevant knowledge
            knowledge_context = await self._get_knowledge_context(message, intent)
            
            # Get real-time context if needed
            real_time_context = await self._get_real_time_context(message, intent)
            
            # Build enhanced prompt
            enhanced_prompt = await self._build_enhanced_prompt(
                message, intent, knowledge_context, real_time_context, user_role
            )
            
            # Generate response
            response = await self._generate_response(enhanced_prompt, session)
            
            # Update conversation history
            session['history'].append({
                'timestamp': datetime.utcnow().isoformat(),
                'user_message': message,
                'intent': intent,
                'response': response,
                'context_used': {
                    'knowledge_docs': len(knowledge_context.get('relevant_documents', [])),
                    'real_time_data': bool(real_time_context)
                }
            })
            
            # Prepare response
            chat_response = {
                'response': response,
                'intent': intent,
                'confidence': 0.85,  # Would be calculated based on actual model
                'context_sources': self._get_context_sources(knowledge_context, real_time_context),
                'suggestions': await self._generate_suggestions(intent, context),
                'timestamp': datetime.utcnow().isoformat()
            }
            
            return chat_response
            
        except Exception as e:
            logger.error(f"Chat processing failed: {e}")
            return {
                'response': "I apologize, but I'm experiencing technical difficulties. Please try again or contact support if the issue persists.",
                'intent': 'error',
                'confidence': 0.0,
                'context_sources': [],
                'suggestions': [],
                'timestamp': datetime.utcnow().isoformat(),
                'error': str(e)
            }
    
    def _get_user_session(self, user_id: str, user_role: str) -> Dict[str, Any]:
        """Get or create user session"""
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = {
                'user_id': user_id,
                'user_role': user_role,
                'created_at': datetime.utcnow().isoformat(),
                'history': [],
                'preferences': {},
                'context': {}
            }
        
        return self.user_sessions[user_id]
    
    async def _classify_intent(self, message: str) -> str:
        """Classify user intent from message"""
        try:
            message_lower = message.lower()
            
            # Check patterns for each intent
            for intent, patterns in self.intent_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, message_lower):
                        return intent
            
            # Default intent
            return 'general_inquiry'
            
        except Exception as e:
            logger.error(f"Intent classification failed: {e}")
            return 'general_inquiry'
    
    async def _get_knowledge_context(self, message: str, intent: str) -> Dict[str, Any]:
        """Get relevant knowledge base context"""
        try:
            # Determine context type based on intent
            context_type_map = {
                'system_status': 'operational',
                'anomaly_inquiry': 'operational',
                'operational_guidance': 'operational',
                'prediction_query': 'technical',
                'optimization_request': 'technical',
                'historical_analysis': 'performance'
            }
            
            context_type = context_type_map.get(intent, 'general')
            
            # Get contextual information from knowledge base
            context = await self.knowledge_base.get_contextual_information(
                message, context_type
            )
            
            return context
            
        except Exception as e:
            logger.error(f"Failed to get knowledge context: {e}")
            return {}
    
    async def _get_real_time_context(self, message: str, intent: str) -> Dict[str, Any]:
        """Get real-time system context if relevant"""
        try:
            # Only get real-time context for certain intents
            real_time_intents = ['system_status', 'anomaly_inquiry', 'prediction_query']
            
            if intent not in real_time_intents:
                return {}
            
            # This would fetch real-time data from APIs or database
            # For now, return mock data
            return {
                'current_time': datetime.utcnow().isoformat(),
                'system_health': 'operational',
                'active_alerts': 2,
                'average_congestion': 45
            }
            
        except Exception as e:
            logger.error(f"Failed to get real-time context: {e}")
            return {}
    
    async def _build_enhanced_prompt(self, message: str, intent: str,
                                   knowledge_context: Dict[str, Any],
                                   real_time_context: Dict[str, Any],
                                   user_role: str) -> str:
        """Build enhanced prompt with context"""
        try:
            # Base system prompt
            system_prompt = self.system_prompts.get(user_role, self.system_prompts['general'])
            
            # Add knowledge context
            context_info = []
            
            if knowledge_context.get('relevant_documents'):
                context_info.append("Relevant Documentation:")
                for doc in knowledge_context['relevant_documents'][:3]:  # Top 3 docs
                    context_info.append(f"- {doc['content'][:200]}...")
            
            if real_time_context:
                context_info.append(f"\nCurrent System Status:")
                for key, value in real_time_context.items():
                    context_info.append(f"- {key}: {value}")
            
            # Build final prompt
            enhanced_prompt = f"""
{system_prompt}

Current Context:
{chr(10).join(context_info) if context_info else "No specific context available."}

User Intent: {intent}
User Role: {user_role}

User Question: {message}

Please provide a helpful, accurate response based on the available context and your knowledge of smart mobility systems.
"""
            
            return enhanced_prompt
            
        except Exception as e:
            logger.error(f"Failed to build enhanced prompt: {e}")
            return message
    
    async def _generate_response(self, prompt: str, session: Dict[str, Any]) -> str:
        """Generate AI response"""
        try:
            # Use the LLM to generate response
            if hasattr(self.llm, 'apredict'):
                response = await self.llm.apredict(prompt)
            else:
                response = self.llm.predict(prompt)
            
            return response
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return "I apologize, but I'm unable to process your request right now. Please try again later."
    
    def _get_context_sources(self, knowledge_context: Dict[str, Any],
                           real_time_context: Dict[str, Any]) -> List[str]:
        """Get list of context sources used"""
        sources = []
        
        if knowledge_context.get('relevant_documents'):
            sources.append('Knowledge Base')
        
        if real_time_context:
            sources.append('Real-time System Data')
        
        if knowledge_context.get('historical_context'):
            sources.append('Historical Analysis')
        
        return sources
    
    async def _generate_suggestions(self, intent: str, 
                                  context: Optional[Dict[str, Any]] = None) -> List[str]:
        """Generate follow-up suggestions"""
        try:
            suggestions_map = {
                'system_status': [
                    "Show me current congestion levels",
                    "Are there any active alerts?",
                    "What's the overall system performance?"
                ],
                'prediction_query': [
                    "Show me tomorrow's demand forecast",
                    "When will congestion peak today?",
                    "Predict passenger flows for Route A"
                ],
                'anomaly_inquiry': [
                    "Show me recent anomalies",
                    "What caused the last incident?",
                    "How can I prevent similar issues?"
                ],
                'optimization_request': [
                    "Optimize current vehicle dispatch",
                    "Suggest schedule improvements",
                    "How can I reduce wait times?"
                ],
                'operational_guidance': [
                    "Show me the emergency procedures",
                    "What are today's operational priorities?",
                    "How do I handle a vehicle breakdown?"
                ]
            }
            
            return suggestions_map.get(intent, [
                "Tell me about system performance",
                "Show me current alerts",
                "Help me optimize operations"
            ])
            
        except Exception as e:
            logger.error(f"Failed to generate suggestions: {e}")
            return []
    
    async def get_conversation_history(self, user_id: str, 
                                     limit: int = 20) -> List[Dict[str, Any]]:
        """Get conversation history for a user"""
        try:
            session = self.user_sessions.get(user_id, {})
            history = session.get('history', [])
            
            # Return recent conversations
            return history[-limit:] if len(history) > limit else history
            
        except Exception as e:
            logger.error(f"Failed to get conversation history: {e}")
            return []
    
    async def clear_conversation(self, user_id: str) -> bool:
        """Clear conversation history for a user"""
        try:
            if user_id in self.user_sessions:
                self.user_sessions[user_id]['history'] = []
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to clear conversation: {e}")
            return False
    
    async def get_chat_analytics(self) -> Dict[str, Any]:
        """Get chat analytics and statistics"""
        try:
            total_sessions = len(self.user_sessions)
            total_messages = sum(
                len(session.get('history', [])) 
                for session in self.user_sessions.values()
            )
            
            # Intent distribution
            intent_counts = {}
            for session in self.user_sessions.values():
                for exchange in session.get('history', []):
                    intent = exchange.get('intent', 'unknown')
                    intent_counts[intent] = intent_counts.get(intent, 0) + 1
            
            # User role distribution
            role_counts = {}
            for session in self.user_sessions.values():
                role = session.get('user_role', 'unknown')
                role_counts[role] = role_counts.get(role, 0) + 1
            
            return {
                'total_sessions': total_sessions,
                'total_messages': total_messages,
                'intent_distribution': intent_counts,
                'user_role_distribution': role_counts,
                'average_messages_per_session': total_messages / max(total_sessions, 1),
                'generated_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get chat analytics: {e}")
            return {}
    
    async def handle_feedback(self, user_id: str, message_id: str,
                            feedback: str, rating: int) -> bool:
        """Handle user feedback on responses"""
        try:
            session = self.user_sessions.get(user_id)
            if not session:
                return False
            
            # Find the message and add feedback
            for exchange in session['history']:
                if exchange.get('id') == message_id:
                    exchange['feedback'] = {
                        'rating': rating,
                        'comment': feedback,
                        'timestamp': datetime.utcnow().isoformat()
                    }
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to handle feedback: {e}")
            return False

# Example usage
if __name__ == "__main__":
    async def main():
        # Initialize knowledge base
        kb = MobilityKnowledgeBase()
        await kb.initialize()
        
        # Initialize chat agent
        agent = MobilityChatAgent(knowledge_base=kb)
        await agent.initialize()
        
        # Test conversation
        response = await agent.chat(
            message="What's the current system status?",
            user_id="test_user",
            user_role="operator"
        )
        
        print(f"Response: {response['response']}")
        print(f"Intent: {response['intent']}")
        print(f"Sources: {response['context_sources']}")
        print(f"Suggestions: {response['suggestions']}")
        
        # Get analytics
        analytics = await agent.get_chat_analytics()
        print(f"Analytics: {analytics}")
        
        # Cleanup
        await kb.cleanup()
    
    # Run example
    asyncio.run(main())