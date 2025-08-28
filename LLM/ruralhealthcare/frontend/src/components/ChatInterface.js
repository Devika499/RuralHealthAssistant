import React, { useState, useRef, useEffect } from 'react';
import {
  Box,
  Paper,
  TextField,
  Button,
  Typography,
  IconButton,
  CircularProgress,
  Alert,
} from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import MicIcon from '@mui/icons-material/Mic';
import { useAuth } from '../hooks/useAuth';
import { useLanguage } from '../hooks/useLanguage';
import { useSpeech } from '../hooks/useSpeech';
import { useChat } from '../hooks/useChat';
import MessageBubble from './MessageBubble';

const ChatInterface = () => {
  const { user } = useAuth();
  const { language } = useLanguage();
  const { messages, isLoading, error, sendMessage } = useChat();
  const { isListening, startListening, stopListening } = useSpeech();
  const [inputText, setInputText] = useState('');
  const messagesEndRef = useRef(null);

  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages]);

  const handleSend = async () => {
    if (!inputText.trim()) return;
    await sendMessage(inputText);
    setInputText('');
  };

  return (
    <Box sx={{ p: 3 }}>
      <Paper elevation={3} sx={{ p: 3, height: 'calc(100vh - 150px)' }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
          <Typography variant="h5">Chat with AI Doctor</Typography>
        </Box>

        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}

        <Box
          sx={{
            height: 'calc(100% - 120px)',
            overflow: 'auto',
            mb: 2,
            display: 'flex',
            flexDirection: 'column',
          }}
        >
          {messages.map((message, index) => (
            <MessageBubble
              key={index}
              message={message}
              isUser={message.sender === 'user'}
            />
          ))}
          <div ref={messagesEndRef} />
        </Box>

        <Box sx={{ display: 'flex', gap: 1 }}>
          <TextField
            fullWidth
            variant="outlined"
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            placeholder="Type your message..."
            disabled={isLoading}
            onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && handleSend()}
          />
          <IconButton
            color="primary"
            onClick={() => isListening ? stopListening() : startListening()}
            disabled={isLoading}
          >
            <MicIcon sx={{ color: isListening ? 'red' : 'inherit' }} />
          </IconButton>
          <Button
            variant="contained"
            color="primary"
            onClick={handleSend}
            disabled={isLoading || !inputText.trim()}
            endIcon={isLoading ? <CircularProgress size={20} /> : <SendIcon />}
          >
            Send
          </Button>
        </Box>
      </Paper>
    </Box>
  );
};

export default ChatInterface;
