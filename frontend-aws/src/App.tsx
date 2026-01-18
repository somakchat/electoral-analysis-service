import { useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

// Components
import Header from './components/Header';
import ChatInterface from './components/ChatInterface';
import StrategyResults from './components/StrategyResults';
import DocumentUpload from './components/DocumentUpload';
import FeedbackPanel from './components/FeedbackPanel';
import Sidebar from './components/Sidebar';

// Hooks & Types
import { useWebSocket } from './hooks/useWebSocket';
import { useSession } from './hooks/useSession';
import type { Message, AgentActivity, StrategyResponse, WSMessage } from './types';

// AWS Production Endpoints
const PROD_API_URL = 'https://5rk2pj3nei.execute-api.us-east-1.amazonaws.com/prod';
const PROD_WS_URL = 'wss://yq1l37m37g.execute-api.us-east-1.amazonaws.com/prod';

// Config - Uses production URLs by default
const API_URL = import.meta.env.VITE_API_URL || PROD_API_URL;
const WS_URL = import.meta.env.VITE_WS_URL || PROD_WS_URL;

export default function App() {
  const { sessionId, resetSession } = useSession();
  const [messages, setMessages] = useState<Message[]>([]);
  const [, setAgentActivities] = useState<Record<string, AgentActivity>>({});
  const [currentResponse, setCurrentResponse] = useState<StrategyResponse | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [constituency, setConstituency] = useState('');
  const [party, setParty] = useState('');

  // WebSocket message handler
  const handleMessage = useCallback((data: WSMessage) => {
    switch (data.type) {
      case 'ack':
        console.log('[App] Query acknowledged:', data.message);
        break;
      
      case 'agent_activity': {
        const agentData = data as unknown as { agent: string; status: string; task: string };
        setAgentActivities(prev => ({
          ...prev,
          [agentData.agent]: {
            status: agentData.status as 'idle' | 'working' | 'done',
            task: agentData.task || 'Working...'
          }
        }));
        break;
      }
      
      case 'stream':
        break;
      
      case 'final_response': {
        const responseData = data as unknown as StrategyResponse;
        setMessages(prev => [...prev, { 
          role: 'assistant', 
          content: responseData.answer,
          timestamp: new Date().toISOString()
        }]);
        setCurrentResponse(responseData);
        setIsProcessing(false);
        break;
      }
      
      case 'error':
        console.error('[App] WebSocket error:', data.message);
        setIsProcessing(false);
        break;
      
      case 'closed':
        console.log('[App] WebSocket closed');
        break;
    }
  }, []);

  const { sendMessage, isConnected, reconnect } = useWebSocket(WS_URL, handleMessage, {
    onOpen: () => console.log('[App] WebSocket connected'),
    onClose: () => console.log('[App] WebSocket disconnected'),
  });

  // Send query via WebSocket
  const handleSendQuery = useCallback((query: string) => {
    if (!query.trim() || isProcessing) return;

    setIsProcessing(true);
    setAgentActivities({});
    setMessages(prev => [...prev, { 
      role: 'user', 
      content: query,
      timestamp: new Date().toISOString()
    }]);

    sendMessage({
      action: 'chat',
      session_id: sessionId,
      query,
      constituency: constituency || undefined,
      party: party || undefined,
      workflow: 'comprehensive_strategy',
      depth: 'micro',
      include_scenarios: true,
      stream: true
    });
  }, [sessionId, constituency, party, isProcessing, sendMessage]);

  // Handle follow-up questions
  const handleFollowUp = useCallback((query: string) => {
    handleSendQuery(query);
  }, [handleSendQuery]);

  // Reset session and clear state
  const handleNewSession = useCallback(() => {
    resetSession();
    setMessages([]);
    setAgentActivities({});
    setCurrentResponse(null);
  }, [resetSession]);

  // Handle document upload completion
  const handleUploadComplete = useCallback((result: unknown) => {
    console.log('[App] Document uploaded:', result);
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100">
      {/* Header */}
      <Header 
        sessionId={sessionId}
        messageCount={messages.length}
        isConnected={isConnected}
        onNewSession={handleNewSession}
        onReconnect={reconnect}
        onToggleSidebar={() => {}}
      />
      
      {/* Main Content */}
      <main className="container mx-auto px-4 py-6 max-w-6xl">
        {/* Chat Interface - Full Width */}
        <div className="space-y-6">
          <ChatInterface
            messages={messages}
            isProcessing={isProcessing}
            constituency={constituency}
            party={party}
            onConstituencyChange={setConstituency}
            onPartyChange={setParty}
            onSendQuery={handleSendQuery}
          />
          
          {/* Strategy Results */}
          <AnimatePresence>
            {currentResponse && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                transition={{ duration: 0.3 }}
              >
                <StrategyResults 
                  response={currentResponse}
                  onFollowUp={handleFollowUp}
                />
              </motion.div>
            )}
          </AnimatePresence>
          
          {/* Feedback Panel */}
          {currentResponse && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.2 }}
            >
              <FeedbackPanel 
                responseId={currentResponse.response_id}
                sessionId={sessionId}
                apiUrl={API_URL}
              />
            </motion.div>
          )}
          
          {/* Configuration and Quick Links - Below Chat */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="md:col-span-2">
              <Sidebar
                constituency={constituency}
                party={party}
                onConstituencyChange={setConstituency}
                onPartyChange={setParty}
              />
            </div>
            
            <div>
              <DocumentUpload 
                apiUrl={API_URL}
                onUploadComplete={handleUploadComplete}
              />
            </div>
          </div>
        </div>
      </main>
      
      {/* Footer */}
      <footer className="text-center py-6 text-gray-500 text-sm border-t border-gray-200 bg-white/50">
        <p>Political Strategy Maker v1.0 | Powered by Multi-Agent AI</p>
      </footer>
    </div>
  );
}
