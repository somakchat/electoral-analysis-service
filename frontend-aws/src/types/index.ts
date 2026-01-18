/**
 * Political Strategy Maker - TypeScript Types
 */

// ============= Chat Types =============

export interface Message {
  role: 'user' | 'assistant';
  content: string;
  timestamp?: string;
}

export interface AgentActivity {
  status: 'idle' | 'working' | 'done';
  task: string;
}

// ============= Response Types =============

export interface Citation {
  chunk_id?: string;
  doc_id?: string;
  source_path?: string;
  source?: string;
  score?: number;
  relevance_score?: number;
  text?: string;
  content?: string;
}

export interface Interaction {
  type: 'follow_up' | 'clarification' | 'refinement' | 'confirmation';
  message: string;
  options: InteractionOption[];
  priority?: 'suggested' | 'blocking';
}

export interface InteractionOption {
  id: string;
  label: string;
}

// ============= Strategy Types =============

export interface SWOTAnalysis {
  strengths: string[];
  weaknesses: string[];
  opportunities: string[];
  threats: string[];
}

export interface VoterSegment {
  segment_name: string;
  population_share: number | string;
  current_support: string;
  persuadability: string;
  strategy: string;
}

export interface Scenario {
  name: string;
  projected_vote_share: string;
  outcome: string;
  probability?: number;
  key_factors?: string[];
}

export interface Strategy {
  executive_summary?: string;
  swot_analysis?: SWOTAnalysis;
  voter_segments?: VoterSegment[];
  scenarios?: Scenario[];
  priority_actions?: string[];
  risk_factors?: string[];
  timeline?: TimelineItem[];
  budget_allocation?: BudgetItem[];
}

export interface TimelineItem {
  phase: string;
  duration: string;
  activities: string[];
}

export interface BudgetItem {
  category: string;
  percentage: number;
  description: string;
}

// ============= API Response Types =============

export interface StrategyResponse {
  response_id: string;
  answer: string;
  citations: Citation[];
  agents_used: string[];
  confidence: number;
  memory_stored: boolean;
  strategy?: Strategy;
  interactions?: Interaction[];
  needs_clarification?: boolean;
  conversation_context?: Record<string, unknown>;
}

export interface IngestResponse {
  document_id: string;
  chunks_indexed: number;
  entities_extracted: number;
  file_name: string;
  index_status?: {
    local_index?: { status: string; chunks_indexed: number };
    opensearch?: { status: string; chunks_indexed: number };
    knowledge_graph?: { status: string; facts_added: number };
  };
}

export interface FeedbackResponse {
  feedback_id: string;
  status: string;
  message: string;
  correction_applied: boolean;
  learning_updated: boolean;
}

export interface FeedbackStats {
  total_feedback: number;
  total_corrections: number;
  average_rating: number;
}

// ============= WebSocket Message Types =============

export interface WSMessage {
  type: 'ack' | 'agent_activity' | 'stream' | 'final_response' | 'error' | 'closed';
  [key: string]: unknown;
}

export interface WSAckMessage extends WSMessage {
  type: 'ack';
  message: string;
  session_id: string;
  response_id: string;
}

export interface WSAgentActivityMessage extends WSMessage {
  type: 'agent_activity';
  agent: string;
  status: 'working' | 'done';
  task: string;
}

export interface WSStreamMessage extends WSMessage {
  type: 'stream';
  chunk: string;
  done: boolean;
}

export interface WSFinalResponseMessage extends WSMessage {
  type: 'final_response';
  response_id: string;
  answer: string;
  citations: Citation[];
  agents_used: string[];
  confidence: number;
  memory_stored: boolean;
  strategy?: Strategy;
  interactions?: Interaction[];
}

export interface WSErrorMessage extends WSMessage {
  type: 'error';
  message: string;
}

// ============= Config Types =============

export interface AppConfig {
  apiUrl: string;
  wsUrl: string;
  environment: 'development' | 'staging' | 'production';
}

// ============= Agent Types =============

export interface AgentInfo {
  icon: string;
  name: string;
  key: string;
  description?: string;
}

export const AGENTS: AgentInfo[] = [
  { icon: '🔍', name: 'Intelligence Agent', key: 'intelligence', description: 'Gathers and analyzes political intelligence' },
  { icon: '📊', name: 'Voter Analyst', key: 'voter', description: 'Analyzes voter demographics and behavior' },
  { icon: '⚔️', name: 'Opposition Research', key: 'opposition', description: 'Researches opposition strategies' },
  { icon: '🎯', name: 'Ground Strategy', key: 'ground', description: 'Plans ground-level campaign activities' },
  { icon: '💰', name: 'Resource Optimizer', key: 'resource', description: 'Optimizes resource allocation' },
  { icon: '💭', name: 'Sentiment Decoder', key: 'sentiment', description: 'Analyzes public sentiment' },
  { icon: '📈', name: 'Data Scientist', key: 'data', description: 'Performs data analysis and predictions' },
  { icon: '📝', name: 'Strategic Reporter', key: 'reporter', description: 'Compiles strategic reports' }
];

