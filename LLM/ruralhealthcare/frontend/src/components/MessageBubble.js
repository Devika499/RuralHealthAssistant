import React from 'react';
import { Box, Paper, Typography, IconButton, CircularProgress } from '@mui/material';
import { useSpeechSynthesis } from '../hooks/useSpeechSynthesis';

const MessageBubble = ({ message, isUser }) => {
  const { speak } = useSpeechSynthesis();

  const handleSpeak = () => {
    speak(message.text, message.language);
  };

  return (
    <Box
      sx={{
        mb: 2,
        display: 'flex',
        justifyContent: isUser ? 'flex-end' : 'flex-start',
        alignItems: 'flex-end',
      }}
    >
      <Paper
        elevation={1}
        sx={{
          p: 2,
          maxWidth: '80%',
          backgroundColor: isUser ? '#e3f2fd' : '#fff',
          borderRadius: '12px',
          border: isUser ? 'none' : '1px solid #e0e0e0',
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          <Typography variant="body1" sx={{ flexGrow: 1 }}>
            {message.text}
          </Typography>
          <IconButton
            size="small"
            onClick={handleSpeak}
            sx={{ ml: 1 }}
          >
            <CircularProgress size={20} />
          </IconButton>
        </Box>
        <Typography
          variant="caption"
          sx={{
            color: 'text.secondary',
            mt: 0.5,
            fontSize: '0.75rem',
          }}
        >
          {new Date(message.timestamp).toLocaleString()}
        </Typography>
      </Paper>
    </Box>
  );
};

export default MessageBubble;
