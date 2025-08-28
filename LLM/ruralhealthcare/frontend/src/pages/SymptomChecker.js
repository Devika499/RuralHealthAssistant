import React, { useState } from 'react';
import { Box, Typography, TextField, Button, Paper, List, ListItem, ListItemText, CircularProgress, MenuItem } from '@mui/material';
import { useTranslation } from 'react-i18next';

const languageOptions = [
  { value: 'en', label: 'English' },
  { value: 'hi', label: 'Hindi' },
  { value: 'te', label: 'Telugu' },
  { value: 'ta', label: 'Tamil' },
  { value: 'kn', label: 'Kannada' },
  { value: 'bn', label: 'Bengali' },
  { value: 'mr', label: 'Marathi' },
  { value: 'pa', label: 'Punjabi' },
  { value: 'gu', label: 'Gujarati' },
  { value: 'ml', label: 'Malayalam' },
  { value: 'or', label: 'Odia' },
  // Add more as needed
];

const getToken = () => localStorage.getItem('token');

async function startSymptomSession(symptom, language) {
  const token = getToken();
  const res = await fetch('/symptom/start', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify({ symptom, language }),
  });
  if (!res.ok) throw new Error('Failed to start session');
  return await res.json();
}

async function sendSymptomAnswer(session_id, answer) {
  const token = getToken();
  const res = await fetch('/symptom/answer', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify({ session_id, answer }),
  });
  if (!res.ok) throw new Error('Failed to send answer');
  return await res.json();
}

async function finishSymptomSession(session_id) {
  const token = getToken();
  const res = await fetch('/symptom/finish', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify({ session_id }),
  });
  if (!res.ok) throw new Error('Failed to finish session');
  return await res.json();
}

const SymptomChecker = () => {
  const { t } = useTranslation();
  const [language, setLanguage] = useState('te'); // Default to Telugu, or use navigator.language
  const [symptom, setSymptom] = useState('');
  const [sessionId, setSessionId] = useState(null);
  const [questions, setQuestions] = useState([]);
  const [answers, setAnswers] = useState([]);
  const [currentQ, setCurrentQ] = useState(0);
  const [input, setInput] = useState('');
  const [chat, setChat] = useState([]);
  const [loading, setLoading] = useState(false);
  const [advice, setAdvice] = useState(null);
  const [error, setError] = useState('');

  // Start session
  const handleStart = async () => {
    setLoading(true);
    setError('');
    try {
      const res = await startSymptomSession(symptom, language);
      setSessionId(res.session_id);
      setQuestions(res.questions); // These are in native language
      setChat([{ sender: 'user', text: symptom }]);
    } catch (err) {
      setError(err.message);
    }
    setLoading(false);
  };

  // Handle answer to a question
  const handleAnswer = async () => {
    setLoading(true);
    setError('');
    setChat((prev) => [
      ...prev,
      { sender: 'bot', text: questions[currentQ] },
      { sender: 'user', text: input },
    ]);
    setAnswers((prev) => [...prev, input]);
    try {
      if (currentQ < questions.length - 1) {
        await sendSymptomAnswer(sessionId, input);
        setCurrentQ(currentQ + 1);
      } else {
        const res = await finishSymptomSession(sessionId);
        setAdvice(res.advice);
      }
    } catch (err) {
      setError(err.message);
    }
    setInput('');
    setLoading(false);
  };

  // Reset session
  const handleReset = () => {
    setSymptom('');
    setSessionId(null);
    setQuestions([]);
    setAnswers([]);
    setCurrentQ(0);
    setInput('');
    setChat([]);
    setAdvice(null);
    setError('');
  };

  return (
    <Box sx={{ maxWidth: 600, mx: 'auto', mt: 4 }}>
      <Typography variant="h5" gutterBottom>
        {t('symptom_checker')}
      </Typography>
      <Typography variant="body2" color="text.secondary" gutterBottom>
        {t('language')}: {language.toUpperCase()}
      </Typography>
      <TextField
        select
        label={t('select_language')}
        value={language}
        onChange={e => setLanguage(e.target.value)}
        sx={{ mb: 2, width: 200 }}
      >
        {languageOptions.map(opt => (
          <MenuItem key={opt.value} value={opt.value}>{opt.label}</MenuItem>
        ))}
      </TextField>
      <Paper sx={{ minHeight: 200, mb: 2, p: 2 }}>
        <List>
          {chat.map((msg, idx) => (
            <ListItem key={idx}>
              <ListItemText
                primary={msg.text}
                secondary={msg.sender === 'user' ? 'You' : 'Assistant'}
                sx={{ textAlign: msg.sender === 'user' ? 'right' : 'left' }}
              />
            </ListItem>
          ))}
        </List>
        {advice && (
          <Box sx={{ mt: 2 }}>
            <Typography variant="subtitle1" color="primary">
              {t('advice')}:
            </Typography>
            <Typography>{advice}</Typography>
          </Box>
        )}
      </Paper>
      {error && <Typography color="error" sx={{ mb: 2 }}>{error}</Typography>}
      {!sessionId ? (
        <Box component="form" onSubmit={e => { e.preventDefault(); handleStart(); }}>
          <TextField
            label={t('describe_symptom')}
            value={symptom}
            onChange={e => setSymptom(e.target.value)}
            fullWidth
            required
            sx={{ mb: 2 }}
          />
          <Button type="submit" variant="contained" disabled={loading || !symptom}>
            {loading ? <CircularProgress size={24} /> : t('start_symptom_check')}
          </Button>
        </Box>
      ) : !advice ? (
        <Box component="form" onSubmit={e => { e.preventDefault(); handleAnswer(); }}>
          {loading ? (
            <Typography sx={{ mb: 1 }} color="text.secondary">
              {t('translating_question')}...
            </Typography>
          ) : (
            <Typography sx={{ mb: 1 }}>{questions[currentQ]}</Typography>
          )}
          <TextField
            label={t('your_answer')}
            value={input}
            onChange={e => setInput(e.target.value)}
            fullWidth
            required
            sx={{ mb: 2 }}
            disabled={loading}
          />
          <Button type="submit" variant="contained" disabled={loading || !input}>
            {loading ? <CircularProgress size={24} /> : t('submit_answer')}
          </Button>
        </Box>
      ) : (
        <Button variant="outlined" onClick={handleReset} sx={{ mt: 2 }}>
          {t('start_new_check')}
        </Button>
      )}
    </Box>
  );
};

export default SymptomChecker; 