import { motion } from 'framer-motion';
import type { AgentActivity } from '../types';
import { AGENTS } from '../types';

interface AgentPanelProps {
  activities: Record<string, AgentActivity>;
  isProcessing?: boolean;
}

export default function AgentPanel({ activities, isProcessing = false }: AgentPanelProps) {
  // Calculate progress
  const totalAgents = AGENTS.length;
  const doneAgents = Object.values(activities).filter(a => a.status === 'done').length;
  const workingAgents = Object.values(activities).filter(a => a.status === 'working').length;
  const progress = totalAgents > 0 ? (doneAgents / totalAgents) * 100 : 0;

  return (
    <div className="glass-panel rounded-2xl overflow-hidden">
      {/* Header */}
      <div className="p-4 border-b border-white/10">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-xl font-display font-bold text-white flex items-center gap-2">
            <span className="text-2xl">🤖</span>
            Agent Activity
          </h2>
          {isProcessing && (
            <span className="text-xs text-rose-400 bg-rose-500/10 px-2 py-1 rounded-full">
              Processing
            </span>
          )}
        </div>
        
        {/* Progress Bar */}
        {isProcessing && (
          <div className="progress-bar">
            <motion.div
              className="progress-bar-fill"
              initial={{ width: 0 }}
              animate={{ width: `${Math.max(progress, workingAgents > 0 ? 10 : 0)}%` }}
              transition={{ duration: 0.3 }}
            />
          </div>
        )}
      </div>
      
      {/* Agent List */}
      <div className="p-4 space-y-2 max-h-[400px] overflow-y-auto scrollbar-thin">
        {AGENTS.map(({ icon, name, description }, index) => {
          const activity = activities[name] || { status: 'idle', task: 'Waiting...' };
          const isWorking = activity.status === 'working';
          const isDone = activity.status === 'done';
          
          return (
            <motion.div
              key={name}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.05 }}
              className={`agent-card ${
                isWorking 
                  ? 'agent-card-working animate-pulse-glow' 
                  : isDone 
                  ? 'agent-card-done' 
                  : 'agent-card-idle'
              }`}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="text-lg">{icon}</span>
                  <div>
                    <span className="text-white font-medium text-sm">{name}</span>
                    {description && (
                      <p className="text-xs text-slate-500 hidden sm:block">{description}</p>
                    )}
                  </div>
                </div>
                <span className="text-lg">
                  {isWorking ? (
                    <motion.span
                      animate={{ rotate: 360 }}
                      transition={{ repeat: Infinity, duration: 2, ease: 'linear' }}
                      className="inline-block"
                    >
                      ⏳
                    </motion.span>
                  ) : isDone ? (
                    <motion.span
                      initial={{ scale: 0 }}
                      animate={{ scale: 1 }}
                      transition={{ type: 'spring', stiffness: 500 }}
                    >
                      ✅
                    </motion.span>
                  ) : (
                    '💤'
                  )}
                </span>
              </div>
              
              {/* Task description */}
              <motion.p
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className={`text-xs mt-1 truncate ${
                  isWorking ? 'text-rose-300' : isDone ? 'text-emerald-300' : 'text-slate-500'
                }`}
              >
                {activity.task.slice(0, 60)}{activity.task.length > 60 ? '...' : ''}
              </motion.p>
            </motion.div>
          );
        })}
      </div>
      
      {/* Summary */}
      {Object.keys(activities).length > 0 && (
        <div className="p-4 border-t border-white/10 bg-black/20">
          <div className="flex justify-between text-xs text-slate-400">
            <span>
              {workingAgents > 0 && `${workingAgents} working`}
              {workingAgents > 0 && doneAgents > 0 && ' • '}
              {doneAgents > 0 && `${doneAgents} completed`}
            </span>
            <span>{Math.round(progress)}% complete</span>
          </div>
        </div>
      )}
    </div>
  );
}

