import { useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Upload, FileText, Check, X, Loader2, AlertCircle } from 'lucide-react';
import type { IngestResponse } from '../types';

interface DocumentUploadProps {
  apiUrl: string;
  onUploadComplete: (result: IngestResponse) => void;
}

interface UploadedFile {
  file: File;
  status: 'pending' | 'uploading' | 'success' | 'error';
  result?: IngestResponse;
  error?: string;
}

const ALLOWED_EXTENSIONS = ['.xlsx', '.xls', '.xlsm', '.docx', '.pdf', '.txt'];
const MAX_FILE_SIZE = 50 * 1024 * 1024; // 50MB

export default function DocumentUpload({ apiUrl, onUploadComplete }: DocumentUploadProps) {
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);

  const validateFile = (file: File): string | null => {
    const ext = '.' + file.name.split('.').pop()?.toLowerCase();
    if (!ALLOWED_EXTENSIONS.includes(ext)) {
      return `Unsupported file type. Allowed: ${ALLOWED_EXTENSIONS.join(', ')}`;
    }
    if (file.size > MAX_FILE_SIZE) {
      return `File too large. Maximum size: ${MAX_FILE_SIZE / 1024 / 1024}MB`;
    }
    return null;
  };

  const handleFiles = useCallback((files: FileList | File[]) => {
    const newFiles: UploadedFile[] = Array.from(files).map(file => {
      const error = validateFile(file);
      return {
        file,
        status: error ? 'error' : 'pending',
        error
      } as UploadedFile;
    });
    
    setUploadedFiles(prev => [...prev, ...newFiles]);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files) {
      handleFiles(e.dataTransfer.files);
    }
  }, [handleFiles]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const uploadFile = async (uploadedFile: UploadedFile, index: number) => {
    const formData = new FormData();
    formData.append('file', uploadedFile.file);
    formData.append('extract_entities', 'true');

    setUploadedFiles(prev => prev.map((f, i) => 
      i === index ? { ...f, status: 'uploading' } : f
    ));

    try {
      const response = await fetch(`${apiUrl}/ingest`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText);
      }

      const result: IngestResponse = await response.json();
      
      setUploadedFiles(prev => prev.map((f, i) => 
        i === index ? { ...f, status: 'success', result } : f
      ));
      
      onUploadComplete(result);
    } catch (error) {
      setUploadedFiles(prev => prev.map((f, i) => 
        i === index ? { 
          ...f, 
          status: 'error', 
          error: error instanceof Error ? error.message : 'Upload failed' 
        } : f
      ));
    }
  };

  const handleUploadAll = async () => {
    setIsUploading(true);
    
    for (let i = 0; i < uploadedFiles.length; i++) {
      if (uploadedFiles[i].status === 'pending') {
        await uploadFile(uploadedFiles[i], i);
      }
    }
    
    setIsUploading(false);
  };

  const removeFile = (index: number) => {
    setUploadedFiles(prev => prev.filter((_, i) => i !== index));
  };

  const clearCompleted = () => {
    setUploadedFiles(prev => prev.filter(f => f.status !== 'success'));
  };

  const pendingCount = uploadedFiles.filter(f => f.status === 'pending').length;
  const successCount = uploadedFiles.filter(f => f.status === 'success').length;

  return (
    <motion.div 
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.2 }}
      className="bg-white rounded-2xl shadow-lg overflow-hidden border border-gray-100"
    >
      {/* Header */}
      <div className="p-4 bg-gradient-to-r from-amber-500 to-orange-500">
        <h2 className="text-lg font-bold text-white flex items-center gap-2">
          <span className="text-xl">📁</span>
          Document Upload
        </h2>
      </div>
      
      {/* Drop Zone */}
      <div className="p-4">
        <div
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          className={`border-2 border-dashed rounded-xl p-6 text-center transition-all duration-200 ${
            isDragging
              ? 'border-amber-500 bg-amber-50'
              : 'border-gray-200 hover:border-amber-300 hover:bg-amber-50/50'
          }`}
        >
          <Upload className={`w-10 h-10 mx-auto mb-3 ${
            isDragging ? 'text-amber-500' : 'text-gray-400'
          }`} />
          <p className="text-sm text-gray-600 mb-2">
            Drag & drop files here, or{' '}
            <label className="text-amber-600 hover:text-amber-700 cursor-pointer font-medium">
              browse
              <input
                type="file"
                multiple
                accept={ALLOWED_EXTENSIONS.join(',')}
                onChange={(e) => e.target.files && handleFiles(e.target.files)}
                className="hidden"
              />
            </label>
          </p>
          <p className="text-xs text-gray-400">
            Supported: {ALLOWED_EXTENSIONS.join(', ')} (max 50MB)
          </p>
        </div>
      </div>
      
      {/* File List */}
      {uploadedFiles.length > 0 && (
        <div className="px-4 pb-4">
          <div className="space-y-2 max-h-48 overflow-y-auto">
            <AnimatePresence>
              {uploadedFiles.map((uploadedFile, index) => (
                <motion.div
                  key={`${uploadedFile.file.name}-${index}`}
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  className={`flex items-center gap-3 p-3 rounded-lg ${
                    uploadedFile.status === 'error'
                      ? 'bg-rose-50 border border-rose-200'
                      : uploadedFile.status === 'success'
                      ? 'bg-emerald-50 border border-emerald-200'
                      : 'bg-gray-50 border border-gray-200'
                  }`}
                >
                  <FileText className={`w-5 h-5 flex-shrink-0 ${
                    uploadedFile.status === 'error'
                      ? 'text-rose-500'
                      : uploadedFile.status === 'success'
                      ? 'text-emerald-500'
                      : 'text-gray-400'
                  }`} />
                  
                  <div className="flex-1 min-w-0">
                    <p className="text-sm text-gray-800 truncate font-medium">
                      {uploadedFile.file.name}
                    </p>
                    {uploadedFile.status === 'success' && uploadedFile.result && (
                      <p className="text-xs text-emerald-600">
                        {uploadedFile.result.chunks_indexed} chunks, {uploadedFile.result.entities_extracted} entities
                      </p>
                    )}
                    {uploadedFile.status === 'error' && uploadedFile.error && (
                      <p className="text-xs text-rose-600 truncate">
                        {uploadedFile.error}
                      </p>
                    )}
                    {uploadedFile.status === 'uploading' && (
                      <p className="text-xs text-amber-600">Uploading...</p>
                    )}
                  </div>
                  
                  <div className="flex-shrink-0">
                    {uploadedFile.status === 'uploading' ? (
                      <Loader2 className="w-5 h-5 text-amber-500 animate-spin" />
                    ) : uploadedFile.status === 'success' ? (
                      <Check className="w-5 h-5 text-emerald-500" />
                    ) : uploadedFile.status === 'error' ? (
                      <AlertCircle className="w-5 h-5 text-rose-500" />
                    ) : (
                      <button
                        onClick={() => removeFile(index)}
                        className="p-1 hover:bg-gray-200 rounded transition-colors"
                      >
                        <X className="w-4 h-4 text-gray-500" />
                      </button>
                    )}
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>
          </div>
          
          {/* Actions */}
          <div className="flex gap-2 mt-4">
            {pendingCount > 0 && (
              <button
                onClick={handleUploadAll}
                disabled={isUploading}
                className="flex-1 flex items-center justify-center gap-2 px-4 py-2 rounded-lg bg-gradient-to-r from-amber-500 to-orange-500 hover:from-amber-600 hover:to-orange-600 text-white font-medium shadow-md disabled:opacity-50 transition-all"
              >
                {isUploading ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Upload className="w-4 h-4" />
                )}
                Upload {pendingCount} file{pendingCount > 1 ? 's' : ''}
              </button>
            )}
            {successCount > 0 && (
              <button
                onClick={clearCompleted}
                className="flex items-center gap-2 px-4 py-2 rounded-lg bg-gray-100 hover:bg-gray-200 text-gray-700 font-medium transition-colors"
              >
                <Check className="w-4 h-4" />
                Clear Done
              </button>
            )}
          </div>
        </div>
      )}
    </motion.div>
  );
}
