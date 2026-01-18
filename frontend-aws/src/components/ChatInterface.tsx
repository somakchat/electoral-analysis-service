import { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Send, Settings2, Loader2, User, Bot } from 'lucide-react';
import type { Message } from '../types';

interface ChatInterfaceProps {
  messages: Message[];
  isProcessing: boolean;
  constituency: string;
  party: string;
  onConstituencyChange: (value: string) => void;
  onPartyChange: (value: string) => void;
  onSendQuery: (query: string) => void;
}

export default function ChatInterface({
  messages,
  isProcessing,
  constituency,
  party,
  onConstituencyChange,
  onPartyChange,
  onSendQuery
}: ChatInterfaceProps) {
  const [input, setInput] = useState('');
  const [showOptions, setShowOptions] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`;
    }
  }, [input]);

  const handleSubmit = () => {
    if (input.trim() && !isProcessing) {
      onSendQuery(input);
      setInput('');
      if (textareaRef.current) {
        textareaRef.current.style.height = 'auto';
      }
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  // Quick action suggestions
  const quickActions = [
    { label: '🎯 Swing Seats', query: 'What are the swing seats in West Bengal?' },
    { label: '📊 TMC Analysis', query: 'What is TMC\'s current position?' },
    { label: '📈 BJP Strategy', query: 'What strategic actions should BJP take?' },
    { label: '🗺️ District View', query: 'Analyze the key districts' },
  ];

  return (
    <div className="bg-white rounded-2xl shadow-lg overflow-hidden border border-gray-100">
      {/* Header */}
      <div className="p-4 bg-gradient-to-r from-blue-500 to-indigo-600 flex items-center justify-between">
        <h2 className="text-xl font-bold text-white flex items-center gap-2">
          <span className="text-2xl">💬</span>
          Strategy Chat
        </h2>
        <button
          onClick={() => setShowOptions(!showOptions)}
          className={`p-2 rounded-lg transition-colors ${
            showOptions 
              ? 'bg-white/30 text-white' 
              : 'hover:bg-white/20 text-white/80 hover:text-white'
          }`}
          title="Advanced options"
        >
          <Settings2 className="w-5 h-5" />
        </button>
      </div>
      
      {/* Advanced Options Panel */}
      <AnimatePresence>
        {showOptions && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="border-b border-gray-100 overflow-hidden"
          >
            <div className="p-4 grid grid-cols-2 gap-4 bg-gray-50">
              <div>
                <label className="text-sm text-gray-600 font-medium block mb-1.5">
                  Constituency (optional)
                </label>
                <input
                  type="text"
                  value={constituency}
                  onChange={(e) => onConstituencyChange(e.target.value)}
                  placeholder="e.g., Nandigram"
                  className="w-full px-3 py-2 rounded-lg border border-gray-200 focus:border-blue-500 focus:ring-2 focus:ring-blue-200 text-gray-800"
                />
              </div>
              <div>
                <label className="text-sm text-gray-600 font-medium block mb-1.5">
                  Party (optional)
                </label>
                <input
                  type="text"
                  value={party}
                  onChange={(e) => onPartyChange(e.target.value)}
                  placeholder="e.g., BJP, TMC"
                  className="w-full px-3 py-2 rounded-lg border border-gray-200 focus:border-blue-500 focus:ring-2 focus:ring-blue-200 text-gray-800"
                />
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
      
      {/* Messages Container */}
      <div className="h-[450px] overflow-y-auto p-4 space-y-4 bg-gradient-to-b from-gray-50 to-white">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center">
            <div className="text-6xl mb-4">🗳️</div>
            <h3 className="text-xl font-bold text-gray-800 mb-2">
              Start Your Strategy Session
            </h3>
            <p className="text-gray-500 max-w-md mb-6">
              Ask strategic questions about West Bengal politics, election analysis, 
              voter demographics, and campaign planning.
            </p>
            
            {/* Quick Actions */}
            <div className="flex flex-wrap justify-center gap-2">
              {quickActions.map((action, idx) => (
                <button
                  key={idx}
                  onClick={() => onSendQuery(action.query)}
                  className="px-4 py-2 text-sm rounded-full bg-gradient-to-r from-blue-50 to-indigo-50 hover:from-blue-100 hover:to-indigo-100 
                           text-blue-700 border border-blue-200 hover:border-blue-300
                           transition-all duration-200 shadow-sm hover:shadow"
                >
                  {action.label}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <>
            {messages.map((msg, idx) => (
              <motion.div
                key={idx}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.2 }}
                className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div className={`flex items-start gap-2 max-w-[85%] ${
                  msg.role === 'user' ? 'flex-row-reverse' : ''
                }`}>
                  {/* Avatar */}
                  <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center shadow-md ${
                    msg.role === 'user' 
                      ? 'bg-gradient-to-br from-blue-500 to-indigo-600' 
                      : 'bg-gradient-to-br from-emerald-500 to-teal-600'
                  }`}>
                    {msg.role === 'user' ? (
                      <User className="w-4 h-4 text-white" />
                    ) : (
                      <Bot className="w-4 h-4 text-white" />
                    )}
                  </div>
                  
                  {/* Message Bubble */}
                  <div className={`rounded-2xl px-4 py-3 shadow-sm ${
                    msg.role === 'user' 
                      ? 'bg-gradient-to-r from-blue-500 to-indigo-600 text-white' 
                      : 'bg-white border border-gray-100 text-gray-800'
                  }`}>
                    <p className="whitespace-pre-wrap text-sm leading-relaxed">
                      {msg.content}
                    </p>
                    {msg.timestamp && (
                      <p className={`text-xs mt-2 ${msg.role === 'user' ? 'text-white/70' : 'text-gray-400'}`}>
                        {new Date(msg.timestamp).toLocaleTimeString()}
                      </p>
                    )}
                  </div>
                </div>
              </motion.div>
            ))}
            
            {/* Processing Indicator */}
            {isProcessing && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="flex items-center gap-2"
              >
                <div className="w-8 h-8 rounded-full bg-gradient-to-br from-emerald-500 to-teal-600 flex items-center justify-center shadow-md">
                  <Bot className="w-4 h-4 text-white" />
                </div>
                <div className="flex items-center gap-2 bg-white rounded-2xl px-4 py-3 shadow-sm border border-gray-100">
                  <Loader2 className="w-4 h-4 animate-spin text-blue-500" />
                  <span className="text-sm text-gray-600">Analyzing and generating strategy...</span>
                </div>
              </motion.div>
            )}
            
            <div ref={messagesEndRef} />
          </>
        )}
      </div>
      
      {/* Input Area */}
      <div className="p-4 border-t border-gray-100 bg-white">
        <div className="flex gap-3">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask your strategy question... (Press Enter to send)"
            className="flex-1 px-4 py-3 rounded-xl border border-gray-200 focus:border-blue-500 focus:ring-2 focus:ring-blue-200 resize-none min-h-[48px] max-h-[200px] text-gray-800 placeholder-gray-400"
            rows={1}
            disabled={isProcessing}
          />
          <motion.button
            onClick={handleSubmit}
            disabled={isProcessing || !input.trim()}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            className="flex items-center justify-center gap-2 px-6 py-3 rounded-xl bg-gradient-to-r from-blue-500 to-indigo-600 hover:from-blue-600 hover:to-indigo-700 text-white font-medium shadow-md hover:shadow-lg disabled:opacity-50 disabled:cursor-not-allowed transition-all self-end"
          >
            {isProcessing ? (
              <Loader2 className="w-5 h-5 animate-spin" />
            ) : (
              <>
                <Send className="w-5 h-5" />
                <span className="hidden sm:inline">Send</span>
              </>
            )}
          </motion.button>
        </div>
      </div>
    </div>
  );
}
