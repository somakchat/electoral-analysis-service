import { motion } from 'framer-motion';
import { 
  Wifi, 
  WifiOff, 
  RefreshCw,
  RotateCcw
} from 'lucide-react';

interface HeaderProps {
  sessionId: string;
  messageCount: number;
  isConnected: boolean;
  onNewSession: () => void;
  onReconnect: () => void;
  onToggleSidebar: () => void;
}

export default function Header({
  sessionId,
  isConnected,
  onNewSession,
  onReconnect,
}: HeaderProps) {
  return (
    <header className="sticky top-0 z-50 bg-white shadow-sm border-b border-gray-100">
      <div className="container mx-auto px-4 max-w-6xl">
        <div className="flex items-center justify-between h-16">
          {/* Left section - Logo */}
          <div className="flex items-center gap-3">
            <motion.div
              initial={{ rotate: -10 }}
              animate={{ rotate: 0 }}
              className="text-3xl"
            >
              🗳️
            </motion.div>
            <div>
              <h1 className="text-xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
                Political Strategy Maker
              </h1>
              <p className="text-xs text-gray-500 hidden sm:block">
                Multi-Agent AI Strategy System
              </p>
            </div>
          </div>
          
          {/* Right section - Status & Actions */}
          <div className="flex items-center gap-3">
            {/* Connection Status */}
            <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full ${
              isConnected 
                ? 'bg-emerald-50 border border-emerald-200' 
                : 'bg-rose-50 border border-rose-200'
            }`}>
              {isConnected ? (
                <>
                  <Wifi className="w-4 h-4 text-emerald-500" />
                  <span className="text-xs text-emerald-600 font-medium hidden sm:inline">Connected</span>
                </>
              ) : (
                <>
                  <WifiOff className="w-4 h-4 text-rose-500" />
                  <span className="text-xs text-rose-600 font-medium hidden sm:inline">Disconnected</span>
                </>
              )}
            </div>
            
            {/* Session ID */}
            <div className="hidden md:flex items-center gap-2 px-3 py-1.5 rounded-full bg-indigo-50 border border-indigo-200">
              <span className="text-xs text-gray-600">
                Session: <code className="text-indigo-600 font-semibold">{sessionId.slice(0, 8)}</code>
              </span>
            </div>
            
            {/* Reconnect Button */}
            {!isConnected && (
              <motion.button
                onClick={onReconnect}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                className="p-2 rounded-lg bg-amber-100 hover:bg-amber-200 text-amber-600 transition-colors"
                title="Reconnect"
              >
                <RefreshCw className="w-4 h-4" />
              </motion.button>
            )}
            
            {/* New Session Button */}
            <motion.button
              onClick={onNewSession}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="flex items-center gap-2 px-4 py-2 rounded-lg bg-gradient-to-r from-blue-500 to-indigo-600 hover:from-blue-600 hover:to-indigo-700 text-white text-sm font-medium shadow-md hover:shadow-lg transition-all"
            >
              <RotateCcw className="w-4 h-4" />
              <span className="hidden sm:inline">New Session</span>
            </motion.button>
          </div>
        </div>
      </div>
    </header>
  );
}
