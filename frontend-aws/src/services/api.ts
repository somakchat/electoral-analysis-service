/**
 * API Service for Political Strategy Maker
 */

import type {
  IngestResponse,
  FeedbackResponse,
  FeedbackStats,
  StrategyResponse
} from '../types';

// AWS Production Endpoints
const PROD_API_URL = 'https://5rk2pj3nei.execute-api.us-east-1.amazonaws.com/prod';

// Get API URL from environment or use production default
const API_URL = import.meta.env.VITE_API_URL || PROD_API_URL;

/**
 * Base fetch wrapper with error handling
 */
async function fetchApi<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const url = `${API_URL}${endpoint}`;
  
  const defaultHeaders: HeadersInit = {
    'Content-Type': 'application/json',
  };

  const response = await fetch(url, {
    ...options,
    headers: {
      ...defaultHeaders,
      ...options.headers,
    },
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`API Error (${response.status}): ${errorText}`);
  }

  return response.json();
}

// ============= Health Check =============

export async function checkHealth(): Promise<{ status: string; version: string }> {
  return fetchApi('/health');
}

// ============= Document Ingestion =============

export async function ingestDocument(
  file: File,
  extractEntities: boolean = true
): Promise<IngestResponse> {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('extract_entities', extractEntities.toString());

  const url = `${API_URL}/ingest`;
  
  const response = await fetch(url, {
    method: 'POST',
    body: formData,
    // Note: Don't set Content-Type header for FormData
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Ingest Error (${response.status}): ${errorText}`);
  }

  return response.json();
}

// ============= Chat (REST) =============

export interface ChatRequest {
  session_id: string;
  query: string;
  constituency?: string;
  party?: string;
  use_crewai?: boolean;
}

export async function sendChatMessage(request: ChatRequest): Promise<StrategyResponse> {
  return fetchApi('/chat', {
    method: 'POST',
    body: JSON.stringify(request),
  });
}

export async function sendInteractiveChat(request: ChatRequest): Promise<StrategyResponse> {
  return fetchApi('/chat/interactive', {
    method: 'POST',
    body: JSON.stringify({
      ...request,
      apply_learnings: true,
    }),
  });
}

export async function quickAnalysis(request: ChatRequest): Promise<StrategyResponse> {
  return fetchApi('/quick-analysis', {
    method: 'POST',
    body: JSON.stringify(request),
  });
}

// ============= RAG Endpoints =============

export interface RAGQueryRequest {
  query: string;
  use_llm?: boolean;
}

export async function queryRAG(request: RAGQueryRequest): Promise<unknown> {
  return fetchApi('/rag/query', {
    method: 'POST',
    body: JSON.stringify(request),
  });
}

export async function getConstituency(name: string): Promise<unknown> {
  return fetchApi(`/rag/constituency/${encodeURIComponent(name)}`);
}

export async function listConstituencies(params?: {
  district?: string;
  pc?: string;
  winner_2021?: string;
  predicted_2026?: string;
  race_rating?: string;
}): Promise<unknown> {
  const queryParams = new URLSearchParams();
  if (params) {
    Object.entries(params).forEach(([key, value]) => {
      if (value) queryParams.append(key, value);
    });
  }
  const queryString = queryParams.toString();
  return fetchApi(`/rag/constituencies${queryString ? `?${queryString}` : ''}`);
}

export async function getPredictions(): Promise<unknown> {
  return fetchApi('/rag/predictions');
}

export async function getSwingAnalysis(): Promise<unknown> {
  return fetchApi('/rag/swing-analysis');
}

export async function getDistrictSummary(name: string): Promise<unknown> {
  return fetchApi(`/rag/district/${encodeURIComponent(name)}`);
}

export async function searchRAG(query: string, topK: number = 5): Promise<unknown> {
  return fetchApi(`/rag/search?q=${encodeURIComponent(query)}&top_k=${topK}`);
}

// ============= Memory =============

export async function getSessionMemory(sessionId: string): Promise<unknown> {
  return fetchApi(`/memory/${encodeURIComponent(sessionId)}`);
}

export async function getEntitiesByType(entityType: string): Promise<unknown> {
  return fetchApi(`/entities/${encodeURIComponent(entityType)}`);
}

export async function searchEntities(query: string, limit: number = 10): Promise<unknown> {
  return fetchApi(`/entities/search/${encodeURIComponent(query)}?limit=${limit}`);
}

// ============= Feedback =============

export interface FeedbackRequest {
  session_id: string;
  response_id: string;
  feedback_type: 'rating' | 'correction' | 'addition' | 'clarification' | 'disagreement' | 'confirmation' | 'comment';
  rating?: number;
  original_text?: string;
  corrected_text?: string;
  comment?: string;
  entity_type?: string;
  entity_name?: string;
}

export async function submitFeedback(request: FeedbackRequest): Promise<FeedbackResponse> {
  return fetchApi('/feedback', {
    method: 'POST',
    body: JSON.stringify(request),
  });
}

export async function getFeedbackStats(): Promise<FeedbackStats> {
  return fetchApi('/feedback/stats');
}

export async function getAllLearnings(): Promise<unknown> {
  return fetchApi('/feedback/learnings');
}

export async function getEntityLearnings(entityName: string): Promise<unknown> {
  return fetchApi(`/feedback/learnings/${encodeURIComponent(entityName)}`);
}

// ============= Knowledge Updates =============

export interface KnowledgeUpdateRequest {
  entity_type: string;
  entity_name: string;
  field: string;
  old_value?: string;
  new_value: string;
  reason: string;
}

export async function updateKnowledge(request: KnowledgeUpdateRequest): Promise<unknown> {
  return fetchApi('/knowledge/update', {
    method: 'POST',
    body: JSON.stringify(request),
  });
}

export async function correctKnowledge(
  entityName: string,
  originalText: string,
  correctedText: string,
  reason?: string
): Promise<FeedbackResponse> {
  const formData = new FormData();
  formData.append('entity_name', entityName);
  formData.append('original_text', originalText);
  formData.append('corrected_text', correctedText);
  if (reason) formData.append('reason', reason);

  const response = await fetch(`${API_URL}/knowledge/correct`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Knowledge Correct Error (${response.status}): ${errorText}`);
  }

  return response.json();
}

// ============= Query Understanding =============

export async function understandQuery(query: string): Promise<unknown> {
  const formData = new FormData();
  formData.append('query', query);

  const response = await fetch(`${API_URL}/query/understand`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Query Understand Error (${response.status}): ${errorText}`);
  }

  return response.json();
}

export async function listIntents(): Promise<unknown> {
  return fetchApi('/query/intents');
}

// ============= Export all =============

export default {
  checkHealth,
  ingestDocument,
  sendChatMessage,
  sendInteractiveChat,
  quickAnalysis,
  queryRAG,
  getConstituency,
  listConstituencies,
  getPredictions,
  getSwingAnalysis,
  getDistrictSummary,
  searchRAG,
  getSessionMemory,
  getEntitiesByType,
  searchEntities,
  submitFeedback,
  getFeedbackStats,
  getAllLearnings,
  getEntityLearnings,
  updateKnowledge,
  correctKnowledge,
  understandQuery,
  listIntents,
};

