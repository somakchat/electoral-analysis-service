import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  ChevronDown, 
  ChevronUp, 
  Target, 
  Users, 
  AlertTriangle,
  TrendingUp,
  FileText,
  Quote,
  Zap
} from 'lucide-react';
import type { StrategyResponse, Citation } from '../types';

interface StrategyResultsProps {
  response: StrategyResponse;
  onFollowUp: (query: string) => void;
}

interface ExpandableSectionProps {
  title: string;
  icon: React.ReactNode;
  defaultExpanded?: boolean;
  children: React.ReactNode;
  badge?: string;
}

function ExpandableSection({ title, icon, defaultExpanded = false, children, badge }: ExpandableSectionProps) {
  const [isExpanded, setIsExpanded] = useState(defaultExpanded);
  
  return (
    <div className="bg-gray-50 rounded-xl p-4 border border-gray-100">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center justify-between text-left"
      >
        <div className="flex items-center gap-2">
          {icon}
          <span className="font-semibold text-gray-800">{title}</span>
          {badge && (
            <span className="text-xs bg-blue-100 text-blue-600 px-2 py-0.5 rounded-full font-medium">
              {badge}
            </span>
          )}
        </div>
        {isExpanded ? (
          <ChevronUp className="w-5 h-5 text-gray-400" />
        ) : (
          <ChevronDown className="w-5 h-5 text-gray-400" />
        )}
      </button>
      
      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="pt-4">{children}</div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export default function StrategyResults({ response, onFollowUp }: StrategyResultsProps) {
  const { strategy, citations, agents_used, confidence, interactions } = response;

  // Quick actions
  const quickActions = [
    { label: '🎯 Swing Seats', query: 'What are the swing seats in West Bengal?' },
    { label: '📊 TMC Analysis', query: 'What is TMC\'s current position?' },
    { label: '📈 BJP Strategy', query: 'What strategic actions should BJP take?' },
    { label: '🗺️ District View', query: 'Analyze the key districts' },
  ];

  return (
    <div className="bg-white rounded-2xl shadow-lg overflow-hidden border border-gray-100">
      {/* Header */}
      <div className="p-4 bg-gradient-to-r from-purple-500 to-pink-500">
        <h2 className="text-xl font-bold text-white flex items-center gap-2">
          <span className="text-2xl">📋</span>
          Strategy Analysis
        </h2>
      </div>
      
      {/* Metrics Row */}
      <div className="grid grid-cols-4 gap-4 p-4 border-b border-gray-100 bg-gray-50">
        <div className="bg-white rounded-xl p-3 text-center shadow-sm border border-gray-100">
          <div className="text-2xl font-bold text-rose-500">{agents_used?.length || 1}</div>
          <div className="text-xs text-gray-500 font-medium">Agents Used</div>
        </div>
        <div className="bg-white rounded-xl p-3 text-center shadow-sm border border-gray-100">
          <div className="text-2xl font-bold text-rose-500">{Math.round((confidence || 0) * 100)}%</div>
          <div className="text-xs text-gray-500 font-medium">Confidence</div>
        </div>
        <div className="bg-white rounded-xl p-3 text-center shadow-sm border border-gray-100">
          <div className="text-2xl font-bold text-rose-500">{citations?.length || 0}</div>
          <div className="text-xs text-gray-500 font-medium">Citations</div>
        </div>
        <div className="bg-white rounded-xl p-3 text-center shadow-sm border border-gray-100">
          <div className="text-2xl font-bold text-emerald-500">{response.memory_stored ? '✅' : '❌'}</div>
          <div className="text-xs text-gray-500 font-medium">Saved</div>
        </div>
      </div>
      
      {/* Strategy Content */}
      <div className="p-4 space-y-3">
        {/* Executive Summary */}
        {strategy?.executive_summary && (
          <ExpandableSection
            title="Executive Summary"
            icon={<FileText className="w-5 h-5 text-purple-500" />}
            defaultExpanded={true}
          >
            <p className="text-gray-700 text-sm leading-relaxed whitespace-pre-wrap">
              {strategy.executive_summary}
            </p>
          </ExpandableSection>
        )}
        
        {/* SWOT Analysis */}
        {strategy?.swot_analysis && (
          <ExpandableSection
            title="SWOT Analysis"
            icon={<Target className="w-5 h-5 text-emerald-500" />}
          >
            <div className="grid grid-cols-2 gap-4">
              {/* Strengths */}
              <div className="bg-emerald-50 rounded-lg p-3 border border-emerald-200">
                <h4 className="font-semibold text-emerald-700 mb-2">💪 Strengths</h4>
                <ul className="space-y-1">
                  {strategy.swot_analysis.strengths?.map((item, i) => (
                    <li key={i} className="text-sm text-gray-700 flex items-start gap-2">
                      <span className="text-emerald-500">•</span>
                      {item}
                    </li>
                  ))}
                </ul>
              </div>
              
              {/* Weaknesses */}
              <div className="bg-amber-50 rounded-lg p-3 border border-amber-200">
                <h4 className="font-semibold text-amber-700 mb-2">⚠️ Weaknesses</h4>
                <ul className="space-y-1">
                  {strategy.swot_analysis.weaknesses?.map((item, i) => (
                    <li key={i} className="text-sm text-gray-700 flex items-start gap-2">
                      <span className="text-amber-500">•</span>
                      {item}
                    </li>
                  ))}
                </ul>
              </div>
              
              {/* Opportunities */}
              <div className="bg-blue-50 rounded-lg p-3 border border-blue-200">
                <h4 className="font-semibold text-blue-700 mb-2">🎯 Opportunities</h4>
                <ul className="space-y-1">
                  {strategy.swot_analysis.opportunities?.map((item, i) => (
                    <li key={i} className="text-sm text-gray-700 flex items-start gap-2">
                      <span className="text-blue-500">•</span>
                      {item}
                    </li>
                  ))}
                </ul>
              </div>
              
              {/* Threats */}
              <div className="bg-rose-50 rounded-lg p-3 border border-rose-200">
                <h4 className="font-semibold text-rose-700 mb-2">🚨 Threats</h4>
                <ul className="space-y-1">
                  {strategy.swot_analysis.threats?.map((item, i) => (
                    <li key={i} className="text-sm text-gray-700 flex items-start gap-2">
                      <span className="text-rose-500">•</span>
                      {item}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </ExpandableSection>
        )}
        
        {/* Voter Segments */}
        {strategy?.voter_segments && strategy.voter_segments.length > 0 && (
          <ExpandableSection
            title="Voter Segments"
            icon={<Users className="w-5 h-5 text-blue-500" />}
            badge={`${strategy.voter_segments.length} segments`}
          >
            <div className="space-y-3">
              {strategy.voter_segments.slice(0, 5).map((segment, i) => (
                <div key={i} className="bg-white rounded-lg p-3 border border-gray-200 shadow-sm">
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-semibold text-gray-800">{segment.segment_name}</span>
                    <span className="text-xs text-gray-500 bg-gray-100 px-2 py-1 rounded-full">
                      {segment.population_share}% of voters
                    </span>
                  </div>
                  <div className="grid grid-cols-3 gap-2 text-xs">
                    <div>
                      <span className="text-gray-500">Support:</span>
                      <span className="text-gray-700 ml-1 font-medium">{segment.current_support}</span>
                    </div>
                    <div>
                      <span className="text-gray-500">Persuadability:</span>
                      <span className="text-gray-700 ml-1 font-medium">{segment.persuadability}</span>
                    </div>
                    <div className="col-span-3 mt-1">
                      <span className="text-gray-500">Strategy:</span>
                      <span className="text-gray-700 ml-1">{segment.strategy}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </ExpandableSection>
        )}
        
        {/* Scenarios */}
        {strategy?.scenarios && strategy.scenarios.length > 0 && (
          <ExpandableSection
            title="Election Scenarios"
            icon={<TrendingUp className="w-5 h-5 text-orange-500" />}
            badge={`${strategy.scenarios.length} scenarios`}
          >
            <div className="space-y-3">
              {strategy.scenarios.map((scenario, i) => (
                <div key={i} className="bg-white rounded-lg p-3 border border-gray-200 shadow-sm">
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-semibold text-gray-800">{scenario.name}</span>
                    <span className="text-lg font-bold text-rose-500">
                      {scenario.projected_vote_share}
                    </span>
                  </div>
                  <p className="text-sm text-gray-600">{scenario.outcome}</p>
                </div>
              ))}
            </div>
          </ExpandableSection>
        )}
        
        {/* Priority Actions */}
        {strategy?.priority_actions && strategy.priority_actions.length > 0 && (
          <ExpandableSection
            title="Priority Actions"
            icon={<Zap className="w-5 h-5 text-yellow-500" />}
            badge={`${strategy.priority_actions.length} actions`}
          >
            <ol className="space-y-2">
              {strategy.priority_actions.map((action, i) => (
                <li key={i} className="flex items-start gap-3 text-sm text-gray-700">
                  <span className="flex-shrink-0 w-6 h-6 rounded-full bg-yellow-100 text-yellow-700 text-xs font-bold flex items-center justify-center border border-yellow-300">
                    {i + 1}
                  </span>
                  {action}
                </li>
              ))}
            </ol>
          </ExpandableSection>
        )}
        
        {/* Risk Factors */}
        {strategy?.risk_factors && strategy.risk_factors.length > 0 && (
          <ExpandableSection
            title="Risk Factors"
            icon={<AlertTriangle className="w-5 h-5 text-rose-500" />}
            badge={`${strategy.risk_factors.length} risks`}
          >
            <div className="space-y-2">
              {strategy.risk_factors.map((risk, i) => (
                <div key={i} className="flex items-start gap-2 text-sm bg-rose-50 rounded-lg p-3 border border-rose-200">
                  <AlertTriangle className="w-4 h-4 text-rose-500 flex-shrink-0 mt-0.5" />
                  <span className="text-gray-700">{risk}</span>
                </div>
              ))}
            </div>
          </ExpandableSection>
        )}
        
        {/* Citations */}
        {citations && citations.length > 0 && (
          <ExpandableSection
            title="Citations"
            icon={<Quote className="w-5 h-5 text-indigo-500" />}
            badge={`${citations.length} sources`}
          >
            <div className="space-y-3 max-h-80 overflow-y-auto">
              {citations.slice(0, 10).map((cite: Citation, i: number) => (
                <div key={i} className="bg-white rounded-lg p-4 border border-gray-200 shadow-sm">
                  <div className="flex items-center justify-between mb-2">
                    <code className="text-sm font-semibold text-indigo-600 bg-indigo-50 px-2 py-1 rounded">
                      [{i + 1}] {cite.chunk_id || cite.doc_id || 'N/A'}
                    </code>
                    <span className="text-sm font-medium text-gray-500 bg-gray-100 px-2 py-1 rounded">
                      Score: {(cite.score || cite.relevance_score || 0).toFixed(3)}
                    </span>
                  </div>
                  <p className="text-sm text-blue-600 font-medium mb-2">
                    📄 Source: {cite.source_path || cite.source || 'Unknown'}
                  </p>
                  <p className="text-sm text-gray-700 leading-relaxed bg-gray-50 p-3 rounded-lg border border-gray-100">
                    {(cite.text || cite.content || '').slice(0, 400)}
                    {(cite.text || cite.content || '').length > 400 ? '...' : ''}
                  </p>
                </div>
              ))}
            </div>
          </ExpandableSection>
        )}
      </div>
      
      {/* Interactive Follow-ups */}
      {interactions && interactions.length > 0 && (
        <div className="p-4 border-t border-gray-100 bg-gradient-to-r from-purple-50 to-pink-50">
          <h3 className="text-sm font-semibold text-gray-800 mb-3">🔄 Continue the Conversation</h3>
          <div className="space-y-3">
            {interactions.map((interaction, i) => (
              <div key={i}>
                <p className="text-sm text-gray-600 mb-2">{interaction.message}</p>
                <div className="flex flex-wrap gap-2">
                  {interaction.options.slice(0, 4).map((opt) => (
                    <button
                      key={opt.id}
                      onClick={() => onFollowUp(opt.label || opt.id)}
                      className="px-3 py-1.5 text-xs rounded-lg bg-purple-100 hover:bg-purple-200 
                               text-purple-700 border border-purple-200 transition-colors font-medium"
                    >
                      {opt.label}
                    </button>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
      
      {/* Quick Actions */}
      <div className="p-4 border-t border-gray-100 bg-gray-50">
        <h3 className="text-sm font-semibold text-gray-800 mb-3">⚡ Quick Actions</h3>
        <div className="flex flex-wrap gap-2">
          {quickActions.map((action, i) => (
            <button
              key={i}
              onClick={() => onFollowUp(action.query)}
              className="px-4 py-2 text-sm rounded-full bg-white hover:bg-gray-100 
                       text-gray-700 border border-gray-200 hover:border-gray-300
                       shadow-sm transition-all duration-200"
            >
              {action.label}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
