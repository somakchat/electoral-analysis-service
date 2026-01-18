import { motion } from 'framer-motion';
import { 
  Settings, 
  ExternalLink,
  MapPin,
  Users
} from 'lucide-react';

interface SidebarProps {
  constituency: string;
  party: string;
  onConstituencyChange: (value: string) => void;
  onPartyChange: (value: string) => void;
}

export default function Sidebar({
  constituency,
  party,
  onConstituencyChange,
  onPartyChange,
}: SidebarProps) {
  return (
    <div className="space-y-4">
      {/* Configuration Panel */}
      <motion.div 
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white rounded-2xl shadow-lg overflow-hidden border border-gray-100"
      >
        <div className="p-4 bg-gradient-to-r from-blue-500 to-indigo-600">
          <h3 className="font-bold text-white flex items-center gap-2">
            <Settings className="w-5 h-5" />
            Configuration
          </h3>
        </div>
        
        <div className="p-4 space-y-4">
          <div>
            <label className="text-sm text-gray-600 font-medium flex items-center gap-1.5 mb-2">
              <MapPin className="w-4 h-4 text-blue-500" />
              Default Constituency
            </label>
            <input
              type="text"
              value={constituency}
              onChange={(e) => onConstituencyChange(e.target.value)}
              placeholder="e.g., Nandigram, Karimpur"
              className="w-full px-4 py-2.5 rounded-lg border border-gray-200 focus:border-blue-500 focus:ring-2 focus:ring-blue-200 transition-all text-gray-800 placeholder-gray-400"
            />
          </div>
          
          <div>
            <label className="text-sm text-gray-600 font-medium flex items-center gap-1.5 mb-2">
              <Users className="w-4 h-4 text-indigo-500" />
              Default Party
            </label>
            <select
              value={party}
              onChange={(e) => onPartyChange(e.target.value)}
              className="w-full px-4 py-2.5 rounded-lg border border-gray-200 focus:border-blue-500 focus:ring-2 focus:ring-blue-200 transition-all text-gray-800"
            >
              <option value="">Select party...</option>
              <option value="BJP">BJP</option>
              <option value="TMC">TMC (Trinamool)</option>
              <option value="INC">INC (Congress)</option>
              <option value="CPM">CPM (Left)</option>
              <option value="AITC">AITC</option>
            </select>
          </div>
        </div>
      </motion.div>
      
      {/* Quick Links */}
      <motion.div 
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="bg-white rounded-2xl shadow-lg p-4 border border-gray-100"
      >
        <h3 className="font-bold text-gray-800 text-sm mb-3 flex items-center gap-2">
          <ExternalLink className="w-4 h-4 text-emerald-500" />
          Quick Links
        </h3>
        <div className="space-y-1">
          <a
            href="#predictions"
            className="flex items-center justify-between p-3 rounded-lg bg-gradient-to-r from-emerald-50 to-teal-50 hover:from-emerald-100 hover:to-teal-100 text-sm text-gray-700 hover:text-emerald-700 transition-all group"
          >
            <span>📊 2026 Predictions</span>
            <ExternalLink className="w-4 h-4 opacity-0 group-hover:opacity-100 transition-opacity" />
          </a>
          <a
            href="#swing-analysis"
            className="flex items-center justify-between p-3 rounded-lg bg-gradient-to-r from-blue-50 to-indigo-50 hover:from-blue-100 hover:to-indigo-100 text-sm text-gray-700 hover:text-blue-700 transition-all group"
          >
            <span>📈 Swing Analysis</span>
            <ExternalLink className="w-4 h-4 opacity-0 group-hover:opacity-100 transition-opacity" />
          </a>
          <a
            href="#constituencies"
            className="flex items-center justify-between p-3 rounded-lg bg-gradient-to-r from-purple-50 to-pink-50 hover:from-purple-100 hover:to-pink-100 text-sm text-gray-700 hover:text-purple-700 transition-all group"
          >
            <span>🗺️ Constituencies</span>
            <ExternalLink className="w-4 h-4 opacity-0 group-hover:opacity-100 transition-opacity" />
          </a>
        </div>
      </motion.div>
    </div>
  );
}
