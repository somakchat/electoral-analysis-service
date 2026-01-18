import { useState, useCallback, useEffect } from 'react';
import { v4 as uuidv4 } from 'uuid';

interface UseSessionReturn {
  sessionId: string;
  resetSession: () => void;
  getSessionAge: () => number;
}

const SESSION_STORAGE_KEY = 'political-strategy-session-id';
const SESSION_TIMESTAMP_KEY = 'political-strategy-session-timestamp';
const SESSION_MAX_AGE = 24 * 60 * 60 * 1000; // 24 hours

export function useSession(): UseSessionReturn {
  const [sessionId, setSessionId] = useState<string>(() => {
    // Try to restore session from localStorage
    if (typeof window !== 'undefined') {
      const storedSessionId = localStorage.getItem(SESSION_STORAGE_KEY);
      const storedTimestamp = localStorage.getItem(SESSION_TIMESTAMP_KEY);
      
      if (storedSessionId && storedTimestamp) {
        const age = Date.now() - parseInt(storedTimestamp, 10);
        // Reuse session if less than 24 hours old
        if (age < SESSION_MAX_AGE) {
          return storedSessionId;
        }
      }
    }
    
    // Create new session
    const newSessionId = uuidv4();
    if (typeof window !== 'undefined') {
      localStorage.setItem(SESSION_STORAGE_KEY, newSessionId);
      localStorage.setItem(SESSION_TIMESTAMP_KEY, Date.now().toString());
    }
    return newSessionId;
  });

  // Persist session changes
  useEffect(() => {
    if (typeof window !== 'undefined') {
      localStorage.setItem(SESSION_STORAGE_KEY, sessionId);
      localStorage.setItem(SESSION_TIMESTAMP_KEY, Date.now().toString());
    }
  }, [sessionId]);

  const resetSession = useCallback(() => {
    const newSessionId = uuidv4();
    setSessionId(newSessionId);
    if (typeof window !== 'undefined') {
      localStorage.setItem(SESSION_STORAGE_KEY, newSessionId);
      localStorage.setItem(SESSION_TIMESTAMP_KEY, Date.now().toString());
    }
    console.log('[Session] New session created:', newSessionId.slice(0, 8));
  }, []);

  const getSessionAge = useCallback(() => {
    if (typeof window !== 'undefined') {
      const storedTimestamp = localStorage.getItem(SESSION_TIMESTAMP_KEY);
      if (storedTimestamp) {
        return Date.now() - parseInt(storedTimestamp, 10);
      }
    }
    return 0;
  }, []);

  return { sessionId, resetSession, getSessionAge };
}

export default useSession;

