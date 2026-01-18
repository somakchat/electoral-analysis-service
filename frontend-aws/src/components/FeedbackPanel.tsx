import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  ThumbsUp,
  MessageSquare, 
  Edit3, 
  Send,
  Loader2,
  Check,
  Star
} from 'lucide-react';
import { submitFeedback, type FeedbackRequest } from '../services/api';

interface FeedbackPanelProps {
  responseId: string;
  sessionId: string;
  apiUrl: string;
}

type FeedbackType = 'rating' | 'correction' | 'addition' | 'disagreement' | 'comment';

export default function FeedbackPanel({ responseId, sessionId }: FeedbackPanelProps) {
  const [feedbackType, setFeedbackType] = useState<FeedbackType | null>(null);
  const [rating, setRating] = useState<number>(0);
  const [hoverRating, setHoverRating] = useState<number>(0);
  const [originalText, setOriginalText] = useState('');
  const [correctedText, setCorrectedText] = useState('');
  const [entityName, setEntityName] = useState('');
  const [comment, setComment] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitted, setSubmitted] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async () => {
    if (!feedbackType) return;
    
    setIsSubmitting(true);
    setError(null);

    const request: FeedbackRequest = {
      session_id: sessionId,
      response_id: responseId,
      feedback_type: feedbackType,
    };

    if (feedbackType === 'rating') {
      request.rating = rating;
    } else if (feedbackType === 'correction') {
      request.original_text = originalText;
      request.corrected_text = correctedText;
      request.entity_name = entityName;
    } else if (feedbackType === 'addition') {
      request.corrected_text = correctedText;
      request.entity_name = entityName;
    } else {
      request.comment = comment;
    }

    try {
      await submitFeedback(request);
      setSubmitted(true);
      
      // Reset after 3 seconds
      setTimeout(() => {
        setSubmitted(false);
        setFeedbackType(null);
        setRating(0);
        setOriginalText('');
        setCorrectedText('');
        setEntityName('');
        setComment('');
      }, 3000);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to submit feedback');
    } finally {
      setIsSubmitting(false);
    }
  };

  const feedbackOptions = [
    { type: 'rating' as FeedbackType, icon: Star, label: 'Rate', color: 'text-yellow-500' },
    { type: 'correction' as FeedbackType, icon: Edit3, label: 'Correct', color: 'text-rose-500' },
    { type: 'addition' as FeedbackType, icon: ThumbsUp, label: 'Add Info', color: 'text-emerald-500' },
    { type: 'comment' as FeedbackType, icon: MessageSquare, label: 'Comment', color: 'text-blue-500' },
  ];

  if (submitted) {
    return (
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        className="bg-white rounded-2xl shadow-lg p-6 text-center border border-gray-100"
      >
        <motion.div
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          transition={{ type: 'spring', stiffness: 500 }}
          className="w-16 h-16 rounded-full bg-emerald-100 flex items-center justify-center mx-auto mb-4"
        >
          <Check className="w-8 h-8 text-emerald-500" />
        </motion.div>
        <h3 className="text-lg font-bold text-gray-800 mb-2">Thank You!</h3>
        <p className="text-sm text-gray-500">Your feedback helps improve future responses.</p>
      </motion.div>
    );
  }

  return (
    <div className="bg-white rounded-2xl shadow-lg overflow-hidden border border-gray-100">
      {/* Header */}
      <div className="p-4 bg-gradient-to-r from-teal-500 to-cyan-500">
        <h2 className="text-xl font-bold text-white flex items-center gap-2">
          <span className="text-2xl">💬</span>
          Provide Feedback
        </h2>
        <p className="text-sm text-white/80 mt-1">Help us improve by sharing your thoughts</p>
      </div>
      
      {/* Feedback Type Selection */}
      <div className="p-4">
        <div className="grid grid-cols-4 gap-2">
          {feedbackOptions.map(({ type, icon: Icon, label, color }) => (
            <button
              key={type}
              onClick={() => setFeedbackType(feedbackType === type ? null : type)}
              className={`p-3 rounded-xl text-center transition-all ${
                feedbackType === type
                  ? 'bg-teal-50 border-teal-300 shadow-sm'
                  : 'bg-gray-50 hover:bg-gray-100 border-gray-200'
              } border`}
            >
              <Icon className={`w-5 h-5 mx-auto mb-1 ${
                feedbackType === type ? color : 'text-gray-400'
              }`} />
              <span className={`text-xs font-medium ${
                feedbackType === type ? 'text-gray-800' : 'text-gray-500'
              }`}>
                {label}
              </span>
            </button>
          ))}
        </div>
      </div>
      
      {/* Feedback Form */}
      <AnimatePresence>
        {feedbackType && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="overflow-hidden"
          >
            <div className="p-4 pt-0 space-y-4">
              {/* Rating Input */}
              {feedbackType === 'rating' && (
                <div>
                  <label className="text-sm text-gray-600 font-medium block mb-2">Rate this response</label>
                  <div className="flex items-center gap-1">
                    {[1, 2, 3, 4, 5].map((value) => (
                      <button
                        key={value}
                        onClick={() => setRating(value)}
                        onMouseEnter={() => setHoverRating(value)}
                        onMouseLeave={() => setHoverRating(0)}
                        className="p-1 transition-transform hover:scale-110"
                      >
                        <Star
                          className={`w-8 h-8 transition-colors ${
                            value <= (hoverRating || rating)
                              ? 'text-yellow-400 fill-yellow-400'
                              : 'text-gray-300'
                          }`}
                        />
                      </button>
                    ))}
                    <span className="ml-3 text-lg">
                      {rating > 0 ? ['😞', '😐', '🙂', '😊', '🤩'][rating - 1] : ''}
                    </span>
                  </div>
                </div>
              )}
              
              {/* Correction Inputs */}
              {feedbackType === 'correction' && (
                <>
                  <div>
                    <label className="text-sm text-gray-600 font-medium block mb-1.5">What was incorrect?</label>
                    <input
                      type="text"
                      value={originalText}
                      onChange={(e) => setOriginalText(e.target.value)}
                      placeholder="e.g., 'BJP won 8 seats'"
                      className="w-full px-4 py-2.5 rounded-lg border border-gray-200 focus:border-teal-500 focus:ring-2 focus:ring-teal-200 text-gray-800 text-sm"
                    />
                  </div>
                  <div>
                    <label className="text-sm text-gray-600 font-medium block mb-1.5">What is correct?</label>
                    <input
                      type="text"
                      value={correctedText}
                      onChange={(e) => setCorrectedText(e.target.value)}
                      placeholder="e.g., 'BJP won 9 seats'"
                      className="w-full px-4 py-2.5 rounded-lg border border-gray-200 focus:border-teal-500 focus:ring-2 focus:ring-teal-200 text-gray-800 text-sm"
                    />
                  </div>
                  <div>
                    <label className="text-sm text-gray-600 font-medium block mb-1.5">Related entity (optional)</label>
                    <input
                      type="text"
                      value={entityName}
                      onChange={(e) => setEntityName(e.target.value)}
                      placeholder="e.g., BANKURA constituency"
                      className="w-full px-4 py-2.5 rounded-lg border border-gray-200 focus:border-teal-500 focus:ring-2 focus:ring-teal-200 text-gray-800 text-sm"
                    />
                  </div>
                </>
              )}
              
              {/* Addition Inputs */}
              {feedbackType === 'addition' && (
                <>
                  <div>
                    <label className="text-sm text-gray-600 font-medium block mb-1.5">New information to add</label>
                    <textarea
                      value={correctedText}
                      onChange={(e) => setCorrectedText(e.target.value)}
                      placeholder="Enter additional information that should be included..."
                      className="w-full px-4 py-2.5 rounded-lg border border-gray-200 focus:border-teal-500 focus:ring-2 focus:ring-teal-200 text-gray-800 text-sm resize-none"
                      rows={3}
                    />
                  </div>
                  <div>
                    <label className="text-sm text-gray-600 font-medium block mb-1.5">Related entity (optional)</label>
                    <input
                      type="text"
                      value={entityName}
                      onChange={(e) => setEntityName(e.target.value)}
                      placeholder="e.g., NANDIGRAM constituency"
                      className="w-full px-4 py-2.5 rounded-lg border border-gray-200 focus:border-teal-500 focus:ring-2 focus:ring-teal-200 text-gray-800 text-sm"
                    />
                  </div>
                </>
              )}
              
              {/* Comment Input */}
              {(feedbackType === 'comment' || feedbackType === 'disagreement') && (
                <div>
                  <label className="text-sm text-gray-600 font-medium block mb-1.5">Your feedback</label>
                  <textarea
                    value={comment}
                    onChange={(e) => setComment(e.target.value)}
                    placeholder="Share your thoughts..."
                    className="w-full px-4 py-2.5 rounded-lg border border-gray-200 focus:border-teal-500 focus:ring-2 focus:ring-teal-200 text-gray-800 text-sm resize-none"
                    rows={3}
                  />
                </div>
              )}
              
              {/* Error Message */}
              {error && (
                <p className="text-sm text-rose-600 bg-rose-50 rounded-lg p-3 border border-rose-200">
                  {error}
                </p>
              )}
              
              {/* Submit Button */}
              <button
                onClick={handleSubmit}
                disabled={isSubmitting || (feedbackType === 'rating' && rating === 0)}
                className="w-full flex items-center justify-center gap-2 px-4 py-3 rounded-lg bg-gradient-to-r from-teal-500 to-cyan-500 hover:from-teal-600 hover:to-cyan-600 text-white font-medium shadow-md disabled:opacity-50 disabled:cursor-not-allowed transition-all"
              >
                {isSubmitting ? (
                  <Loader2 className="w-5 h-5 animate-spin" />
                ) : (
                  <>
                    <Send className="w-5 h-5" />
                    Submit Feedback
                  </>
                )}
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
